#!/usr/bin/env python3
"""
Backtest simulator wired to the current trading algorithm.

This file is standalone and does not modify existing project modules.

Primary market data source: defeatbeta-api (daily OHLC).
Install with: pip install defeatbeta-api
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import io
from pathlib import Path
import re
from typing import List

import numpy as np
import pandas as pd

from Signaux import f_signal_ichimoku, f_signal_RSI
from decision import f_vendre_ou_acheter


@dataclass
class Trade:
    timestamp: str
    action: str
    price: float
    quantity: float
    confidence: float
    cash_after: float


def synthetic_ohlc(bars: int, start_price: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rets = rng.normal(loc=0.0, scale=0.001, size=bars)
    close = start_price * np.cumprod(1.0 + rets)
    open_ = np.concatenate(([close[0]], close[:-1]))
    idx = pd.date_range(end=datetime.now(), periods=bars, freq="min")
    return pd.DataFrame({"Open": open_, "Close": close}, index=idx)


def _period_start(period: str, end_dt: datetime) -> datetime:
    match = re.fullmatch(r"(\d+)([dmyDMy])", period.strip())
    if not match:
        return datetime(1970, 1, 1)
    value = int(match.group(1))
    unit = match.group(2).lower()
    if unit == "d":
        return end_dt - timedelta(days=value)
    if unit == "m":
        return end_dt - timedelta(days=30 * value)
    if unit == "y":
        return end_dt - timedelta(days=365 * value)
    return datetime(1970, 1, 1)


def _load_from_defeatbeta(ticker: str, period: str) -> pd.DataFrame:
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from defeatbeta_api.data.ticker import Ticker
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "defeatbeta-api is not installed. Install it with: pip install defeatbeta-api"
        ) from exc

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        raw = Ticker(ticker).price()
    if raw is None or raw.empty:
        raise ValueError(f"Defeat Beta API returned no data for {ticker}")

    expected_cols = {"report_date", "open", "close"}
    if not expected_cols.issubset(set(raw.columns)):
        missing = expected_cols.difference(set(raw.columns))
        raise ValueError(f"Defeat Beta API price payload missing columns: {sorted(missing)}")

    df = raw[["report_date", "open", "close"]].copy()
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["report_date", "open", "close"]).sort_values("report_date")
    if df.empty:
        raise ValueError(f"Defeat Beta API returned only invalid rows for {ticker}")

    end_dt = df["report_date"].max().to_pydatetime()
    start_dt = _period_start(period, end_dt)
    df = df[df["report_date"] >= start_dt]
    if df.empty:
        raise ValueError(
            f"Defeat Beta API returned data for {ticker}, but nothing matched period={period}"
        )

    return df.set_index("report_date").rename(columns={"open": "Open", "close": "Close"})[["Open", "Close"]]


def download_ohlc(ticker: str, period: str, interval: str, synthetic_bars: int = 0) -> pd.DataFrame:
    source_error = ""
    try:
        if interval.lower() not in {"1d", "1day", "d"}:
            print(
                f"Note: defeatbeta-api provides daily prices; interval={interval} is ignored and daily bars are used."
            )
        return _load_from_defeatbeta(ticker, period)
    except Exception as exc:
        source_error = str(exc)
        df = pd.DataFrame()

    if df.empty:
        try:
            close_df = pd.read_csv("DataC.csv")
            open_df = pd.read_csv("DataO.csv")
            close_vals = pd.to_numeric(close_df.iloc[:, 0], errors="coerce").dropna().reset_index(drop=True)
            open_vals = pd.to_numeric(open_df.iloc[:, 0], errors="coerce").dropna().reset_index(drop=True)
            rows = min(len(close_vals), len(open_vals))
            if rows > 0:
                idx = pd.date_range(end=datetime.now(), periods=rows, freq="min")
                return pd.DataFrame(
                    {"Open": open_vals.iloc[:rows].to_numpy(), "Close": close_vals.iloc[:rows].to_numpy()},
                    index=idx,
                )
        except Exception:
            pass
        if synthetic_bars > 0:
            return synthetic_ohlc(synthetic_bars)
        raise ValueError(
            f"No market data returned for {ticker} from defeatbeta-api and local fallback files DataC.csv/DataO.csv were unusable. "
            f"Primary source error: {source_error}. "
            "You can also pass --synthetic-bars N to test the simulator pipeline offline."
        )
    return df


def build_algo_input(df: pd.DataFrame) -> List[List[float]]:
    # Existing code expects rows shaped as [close, open].
    return [[float(row.Close), float(row.Open)] for row in df.itertuples()]


def external_signal_component() -> float:
    # The original pipeline mixes a Yahoo-based signal; keep neutral in simulator mode.
    return 0.0


def compute_global_signal(price_window: List[List[float]], cached_yf_component: float) -> float:
    coeff_ichi, coeff_rsi, coeff_yf = 1.0, 1.0, 1.0
    max_signal = coeff_ichi + coeff_rsi + coeff_yf
    return (
        coeff_ichi * f_signal_ichimoku(price_window)
        + coeff_rsi * f_signal_RSI(price_window)
        + coeff_yf * cached_yf_component
    ) / max_signal


def run_simulation(
    ticker: str,
    df: pd.DataFrame,
    initial_cash: float,
    fee_pct: float,
    warmup_bars: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if len(df) <= warmup_bars:
        raise ValueError(
            f"Not enough bars: got {len(df)} rows, need > {warmup_bars}. "
            "Use a larger period or a smaller interval."
        )

    algo_prices = build_algo_input(df)
    cached_yf_component = external_signal_component()

    cash = float(initial_cash)
    position_qty = 0.0
    signal_history: List[float] = []
    trades: List[Trade] = []
    equity_rows = []

    for idx in range(warmup_bars, len(algo_prices)):
        window = algo_prices[: idx + 1]
        close_price = window[-1][0]
        ts = df.index[idx]

        try:
            global_signal = compute_global_signal(window, cached_yf_component)
        except Exception:
            # Skip bar if indicators cannot be computed.
            continue

        signal_history.append(global_signal)
        if len(signal_history) > 5:
            signal_history = signal_history[-5:]

        with contextlib.redirect_stdout(io.StringIO()):
            action, confidence = f_vendre_ou_acheter(signal_history)

        if action == "buy" and position_qty == 0.0 and cash > 0.0:
            # Invest all available cash.
            position_qty = (cash * (1.0 - fee_pct)) / close_price
            cash = 0.0
            trades.append(
                Trade(
                    timestamp=str(ts),
                    action="buy",
                    price=close_price,
                    quantity=position_qty,
                    confidence=float(confidence),
                    cash_after=cash,
                )
            )
        elif action == "sell" and position_qty > 0.0:
            cash = position_qty * close_price * (1.0 - fee_pct)
            trades.append(
                Trade(
                    timestamp=str(ts),
                    action="sell",
                    price=close_price,
                    quantity=position_qty,
                    confidence=float(confidence),
                    cash_after=cash,
                )
            )
            position_qty = 0.0

        equity = cash + position_qty * close_price
        equity_rows.append(
            {
                "timestamp": ts,
                "price": close_price,
                "decision": action,
                "confidence": float(confidence),
                "equity": equity,
                "cash": cash,
                "position_qty": position_qty,
            }
        )

    if not equity_rows:
        raise RuntimeError("No simulation rows produced (indicators likely failed on all bars).")

    equity_df = pd.DataFrame(equity_rows).set_index("timestamp")
    trades_df = pd.DataFrame([asdict(t) for t in trades])

    first_price = float(equity_df["price"].iloc[0])
    last_price = float(equity_df["price"].iloc[-1])
    buy_hold_equity = initial_cash * (last_price / first_price)

    summary = {
        "ticker": ticker,
        "bars_used": int(len(equity_df)),
        "start_time": str(equity_df.index[0]),
        "end_time": str(equity_df.index[-1]),
        "initial_cash": float(initial_cash),
        "final_equity": float(equity_df["equity"].iloc[-1]),
        "algo_return_pct": float((equity_df["equity"].iloc[-1] / initial_cash - 1.0) * 100.0),
        "buy_hold_return_pct": float((buy_hold_equity / initial_cash - 1.0) * 100.0),
        "trade_count": int(len(trades_df)),
    }
    return trades_df, equity_df, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simulator for the current trading algorithm.")
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol (default: AAPL)")
    parser.add_argument("--period", default="2y", help="Lookback period (e.g. 30d, 6m, 2y)")
    parser.add_argument("--interval", default="1d", help="Bar interval (defeatbeta-api currently supports daily bars)")
    parser.add_argument("--initial-cash", type=float, default=10_000.0, help="Initial cash")
    parser.add_argument("--fee-pct", type=float, default=0.001, help="Fee per trade (e.g. 0.001 = 0.1%)")
    parser.add_argument(
        "--warmup-bars",
        type=int,
        default=120,
        help="Bars before strategy starts making decisions (default: 120)",
    )
    parser.add_argument(
        "--out-dir",
        default="simulation_output",
        help="Directory for CSV outputs (default: simulation_output)",
    )
    parser.add_argument(
        "--synthetic-bars",
        type=int,
        default=0,
        help="If > 0, use synthetic OHLC bars only when live/local data cannot be loaded.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = download_ohlc(args.ticker, args.period, args.interval, synthetic_bars=args.synthetic_bars)
    trades_df, equity_df, summary = run_simulation(
        ticker=args.ticker,
        df=df,
        initial_cash=args.initial_cash,
        fee_pct=args.fee_pct,
        warmup_bars=args.warmup_bars,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{args.ticker}_{args.period}_{args.interval}_{stamp}"

    equity_path = out_dir / f"{base}_equity.csv"
    trades_path = out_dir / f"{base}_trades.csv"
    summary_path = out_dir / f"{base}_summary.csv"

    equity_df.to_csv(equity_path)
    trades_df.to_csv(trades_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print(f"Ticker: {summary['ticker']}")
    print(f"Bars simulated: {summary['bars_used']}")
    print(f"Initial cash: {summary['initial_cash']:.2f}")
    print(f"Final equity: {summary['final_equity']:.2f}")
    print(f"Algorithm return: {summary['algo_return_pct']:.2f}%")
    print(f"Buy & hold return: {summary['buy_hold_return_pct']:.2f}%")
    print(f"Trades: {summary['trade_count']}")
    print(f"Saved equity curve: {equity_path}")
    print(f"Saved trades: {trades_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
