#!/usr/bin/env python3
"""
Plot simulation outputs (price data, trades, and returns).

By default, uses the most recently modified *_summary.csv file in simulation_output/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pandas.errors import EmptyDataError


def resolve_base(summary_path: str | None, base: str | None, out_dir: str) -> tuple[Path, Path]:
    output_dir = Path(out_dir)
    if base:
        b = Path(base)
        if b.suffix:
            b = b.with_suffix("")
        return output_dir, b

    if summary_path:
        s = Path(summary_path)
        if not s.exists():
            raise FileNotFoundError(f"Summary file not found: {s}")
        if not s.name.endswith("_summary.csv"):
            raise ValueError("Provided --summary must end with '_summary.csv'")
        base_name = s.name[: -len("_summary.csv")]
        return s.parent, s.parent / base_name

    summaries = list(output_dir.glob("*_summary.csv"))
    if not summaries:
        raise FileNotFoundError(f"No *_summary.csv found in {output_dir}")
    latest = max(summaries, key=lambda p: p.stat().st_mtime)
    base_name = latest.name[: -len("_summary.csv")]
    return latest.parent, latest.parent / base_name


def load_frames(base_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_file = Path(str(base_path) + "_summary.csv")
    equity_file = Path(str(base_path) + "_equity.csv")
    trades_file = Path(str(base_path) + "_trades.csv")

    for f in (summary_file, equity_file, trades_file):
        if not f.exists():
            raise FileNotFoundError(f"Missing expected file: {f}")

    summary = pd.read_csv(summary_file)
    equity = pd.read_csv(equity_file)
    try:
        trades = pd.read_csv(trades_file)
    except EmptyDataError:
        trades = pd.DataFrame(columns=["timestamp", "action", "price", "quantity", "confidence", "cash_after"])

    equity["timestamp"] = pd.to_datetime(equity["timestamp"], errors="coerce")
    equity = equity.dropna(subset=["timestamp"]).sort_values("timestamp")
    if equity.empty:
        raise ValueError(f"Equity file has no valid timestamps: {equity_file}")

    if "timestamp" in trades.columns:
        trades["timestamp"] = pd.to_datetime(trades["timestamp"], errors="coerce")
        trades = trades.dropna(subset=["timestamp"]).sort_values("timestamp")

    return summary, equity, trades


def trade_markers(equity: pd.DataFrame, trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty or "action" not in trades.columns:
        return pd.DataFrame(), pd.DataFrame()
    merged = trades.merge(
        equity[["timestamp", "price"]],
        on="timestamp",
        how="left",
        suffixes=("_trade", "_equity"),
    )
    if "price_trade" in merged.columns:
        merged["plot_price"] = merged["price_trade"].fillna(merged["price"])
    else:
        merged["plot_price"] = merged["price"]
    buys = merged[merged["action"].str.lower() == "buy"]
    sells = merged[merged["action"].str.lower() == "sell"]
    return buys, sells


def build_plot(summary: pd.DataFrame, equity: pd.DataFrame, trades: pd.DataFrame, output_png: Path) -> None:
    initial_cash = float(summary["initial_cash"].iloc[0]) if "initial_cash" in summary.columns else float(equity["equity"].iloc[0])
    ticker = str(summary["ticker"].iloc[0]) if "ticker" in summary.columns else "N/A"

    equity = equity.copy()
    equity["algo_return_pct"] = (equity["equity"] / initial_cash - 1.0) * 100.0
    equity["buyhold_equity"] = initial_cash * (equity["price"] / equity["price"].iloc[0])
    equity["buyhold_return_pct"] = (equity["buyhold_equity"] / initial_cash - 1.0) * 100.0

    buys, sells = trade_markers(equity, trades)

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    axes[0].plot(equity["timestamp"], equity["price"], label="Price", color="#1f77b4", linewidth=1.5)
    if not buys.empty:
        axes[0].scatter(buys["timestamp"], buys["plot_price"], marker="^", color="green", s=70, label="Buy")
    if not sells.empty:
        axes[0].scatter(sells["timestamp"], sells["plot_price"], marker="v", color="red", s=70, label="Sell")
    axes[0].set_title(f"{ticker} Price + Trades")
    axes[0].set_ylabel("Price")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(equity["timestamp"], equity["equity"], label="Strategy Equity", color="#ff7f0e", linewidth=1.8)
    axes[1].plot(equity["timestamp"], equity["buyhold_equity"], label="Buy & Hold Equity", color="#2ca02c", linewidth=1.4, linestyle="--")
    axes[1].set_title("Portfolio Value")
    axes[1].set_ylabel("Equity")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    axes[2].plot(equity["timestamp"], equity["algo_return_pct"], label="Strategy Return %", color="#d62728", linewidth=1.8)
    axes[2].plot(equity["timestamp"], equity["buyhold_return_pct"], label="Buy & Hold Return %", color="#9467bd", linewidth=1.4, linestyle="--")
    axes[2].axhline(0, color="black", linewidth=0.8, alpha=0.5)
    axes[2].set_title("Cumulative Returns")
    axes[2].set_ylabel("Return (%)")
    axes[2].set_xlabel("Time")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")

    fig.suptitle(
        f"Simulation Results: {ticker} | Trades={int(summary['trade_count'].iloc[0]) if 'trade_count' in summary.columns else 0}",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot simulation output files.")
    parser.add_argument("--summary", help="Path to a *_summary.csv file")
    parser.add_argument("--base", help="Base path without suffix (e.g. simulation_output/AAPL_2y_1d_20260218_121609)")
    parser.add_argument("--out-dir", default="simulation_output", help="Directory containing simulation CSV files")
    parser.add_argument("--output", help="Path to output PNG (default: <base>_plot.png)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, base_path = resolve_base(args.summary, args.base, args.out_dir)
    summary, equity, trades = load_frames(base_path)
    output_png = Path(args.output) if args.output else Path(str(base_path) + "_plot.png")
    build_plot(summary, equity, trades, output_png)
    print(f"Saved plot: {output_png}")


if __name__ == "__main__":
    main()
