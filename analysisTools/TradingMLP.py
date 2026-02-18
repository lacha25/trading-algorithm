from __future__ import annotations

import contextlib
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "NVDA",
    "TSLA",
    "JPM",
    "XOM",
    "UNH",
    "JNJ",
    "V",
    "HD",
    "SPY",
    "QQQ",
    "BAC",
    "WMT",
    "PG",
    "KO",
    "PEP",
    "AVGO",
    "COST",
    "ADBE",
    "CRM",
    "ORCL",
    "CSCO",
    "NFLX",
    "AMD",
    "INTC",
    "MU",
    "PFE",
    "MRK",
    "CVX",
    "ABBV",
    "MCD",
    "NKE",
    "T",
    "DIS",
    "IBM",
    "CAT",
    "BA",
    "GS",
    "MS",
    "BLK",
    "GE",
    "F",
    "GM",
    "UBER",
    "PYPL",
    "SQ",
]


_TICKER_CLASS = None


def _ticker_class():
    global _TICKER_CLASS
    if _TICKER_CLASS is None:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from defeatbeta_api.data.ticker import Ticker

        _TICKER_CLASS = Ticker
    return _TICKER_CLASS


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_down = down.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_up / (avg_down + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    unique = np.unique(y_true)
    if len(unique) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _fetch_history(ticker: str, cache_dir: Path, refresh_cache: bool) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{ticker}.csv"
    if cache_path.exists() and not refresh_cache:
        cached = pd.read_csv(cache_path)
        cached["report_date"] = pd.to_datetime(cached["report_date"], errors="coerce")
        for col in ["open", "high", "low", "close", "volume"]:
            cached[col] = pd.to_numeric(cached[col], errors="coerce")
        cached = cached.dropna().sort_values("report_date").reset_index(drop=True)
        if not cached.empty:
            return cached

    ticker_cls = _ticker_class()
    data = pd.DataFrame()
    for attempt in range(3):
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                data = ticker_cls(ticker).price()
            if not data.empty:
                break
        except Exception:
            data = pd.DataFrame()
        time.sleep(0.5 + attempt)

    if data.empty:
        raise ValueError(f"No history available for ticker {ticker}")

    required = {"report_date", "open", "high", "low", "close", "volume"}
    if not required.issubset(set(data.columns)):
        raise ValueError(f"Unexpected columns for ticker {ticker}: {list(data.columns)}")

    data = data[list(required)].copy()
    data["report_date"] = pd.to_datetime(data["report_date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna().sort_values("report_date").reset_index(drop=True)
    if data.empty:
        raise ValueError(f"History became empty after cleanup for ticker {ticker}")

    data.to_csv(cache_path, index=False)
    return data


def _build_features(df: pd.DataFrame, ticker: str, horizon: int) -> pd.DataFrame:
    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    frame = pd.DataFrame({"date": df["report_date"]})
    frame["ret1"] = close.pct_change(1)
    frame["ret2"] = close.pct_change(2)
    frame["ret5"] = close.pct_change(5)
    frame["ret10"] = close.pct_change(10)
    frame["ret20"] = close.pct_change(20)
    frame["co_return"] = (close - open_) / (open_ + 1e-12)
    frame["hl_range"] = (high - low) / (close + 1e-12)
    frame["vol5"] = frame["ret1"].rolling(5).std()
    frame["vol20"] = frame["ret1"].rolling(20).std()
    frame["sma10_ratio"] = (close / close.rolling(10).mean()) - 1.0
    frame["sma50_ratio"] = (close / close.rolling(50).mean()) - 1.0
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    frame["macd"] = macd / (close + 1e-12)
    frame["macd_hist"] = (macd - macd_signal) / (close + 1e-12)
    frame["rsi14"] = _rsi(close) / 100.0
    log_volume = np.log1p(volume)
    frame["volume_z20"] = (log_volume - log_volume.rolling(20).mean()) / (log_volume.rolling(20).std() + 1e-12)
    frame["forward_return"] = close.shift(-horizon) / close - 1.0
    frame["target"] = (frame["forward_return"] > 0.0).astype(np.float32)
    frame["ticker"] = ticker
    frame = frame.dropna().reset_index(drop=True)
    return frame


class TradingMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


@dataclass
class TradingMLPConfig:
    tickers: List[str]
    horizon: int = 1
    train_end: str = "2023-01-01"
    val_end: str = "2025-01-01"
    epochs: int = 35
    patience: int = 6
    batch_size: int = 2048
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    trade_cost: float = 4e-4
    cache_dir: str = "analysisTools/cache"
    model_path: str = "model_weights_trading_mlp.pth"
    report_path: str = "mlp_trading_report.json"
    test_daily_path: str = "mlp_trading_test_daily.csv"
    refresh_cache: bool = False


def _daily_strategy_metrics(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    low_threshold: float,
    high_threshold: float,
    horizon: int,
    trade_cost: float,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    pos = np.where(probabilities >= high_threshold, 1, np.where(probabilities <= low_threshold, -1, 0))
    gross = pos * frame["forward_return"].to_numpy()
    net = np.where(pos != 0, gross - trade_cost, 0.0)
    daily = pd.DataFrame({"date": frame["date"].to_numpy(), "strategy_return": net})
    daily = daily.groupby("date", as_index=False)["strategy_return"].mean()
    daily["equity"] = (1.0 + daily["strategy_return"]).cumprod()

    mean_ret = float(daily["strategy_return"].mean())
    std_ret = float(daily["strategy_return"].std())
    ann_factor = float(np.sqrt(252.0 / max(1, horizon)))
    sharpe = 0.0 if std_ret == 0.0 else float((mean_ret / std_ret) * ann_factor)
    coverage = float((pos != 0).mean())
    active = net[pos != 0]
    win_rate = float((active > 0).mean()) if active.size > 0 else 0.0

    metrics = {
        "daily_mean_return": mean_ret,
        "daily_sharpe": sharpe,
        "cumulative_return": float(daily["equity"].iloc[-1] - 1.0) if not daily.empty else 0.0,
        "coverage": coverage,
        "trade_win_rate": win_rate,
    }
    return metrics, daily


def _tune_thresholds(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    horizon: int,
    trade_cost: float,
) -> Tuple[float, float, Dict[str, float]]:
    best_low, best_high = 0.40, 0.54
    best_metrics, _ = _daily_strategy_metrics(frame, probabilities, best_low, best_high, horizon, trade_cost)
    best_sharpe = best_metrics["daily_sharpe"]

    low_grid = [0.40, 0.42, 0.44, 0.46, 0.48]
    high_grid = [0.52, 0.54, 0.56, 0.58, 0.60]
    for low in low_grid:
        for high in high_grid:
            if low >= high:
                continue
            metrics, _ = _daily_strategy_metrics(frame, probabilities, low, high, horizon, trade_cost)
            if metrics["daily_sharpe"] > best_sharpe:
                best_sharpe = metrics["daily_sharpe"]
                best_low = low
                best_high = high
                best_metrics = metrics

    return best_low, best_high, best_metrics


def train_trading_mlp(config: TradingMLPConfig | None = None) -> Dict[str, object]:
    cfg = config or TradingMLPConfig(tickers=list(DEFAULT_TICKERS))
    cache_dir = Path(cfg.cache_dir)

    per_ticker_frames = []
    used_tickers = []
    for ticker in cfg.tickers:
        try:
            raw_df = _fetch_history(ticker, cache_dir=cache_dir, refresh_cache=cfg.refresh_cache)
            feat_df = _build_features(raw_df, ticker=ticker, horizon=cfg.horizon)
            if len(feat_df) < 1000:
                continue
            used_tickers.append(ticker)
            per_ticker_frames.append(feat_df)
        except Exception:
            continue

    if not per_ticker_frames:
        raise RuntimeError("No valid ticker data available for training.")

    all_df = pd.concat(per_ticker_frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    for ticker in sorted(all_df["ticker"].unique()):
        all_df[f"ticker_{ticker}"] = (all_df["ticker"] == ticker).astype(np.float32)

    train_end = pd.Timestamp(cfg.train_end)
    val_end = pd.Timestamp(cfg.val_end)
    train_df = all_df[all_df["date"] < train_end]
    val_df = all_df[(all_df["date"] >= train_end) & (all_df["date"] < val_end)]
    test_df = all_df[all_df["date"] >= val_end]
    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError("Train/validation/test split produced an empty partition.")

    feature_cols = [c for c in all_df.columns if c not in {"date", "ticker", "target", "forward_return"}]
    x_train = train_df[feature_cols].to_numpy(np.float32)
    x_val = val_df[feature_cols].to_numpy(np.float32)
    x_test = test_df[feature_cols].to_numpy(np.float32)
    y_train = train_df["target"].to_numpy(np.float32)
    y_val = val_df["target"].to_numpy(np.float32)
    y_test = test_df["target"].to_numpy(np.float32)

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0) + 1e-8
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    recency_days = (train_df["date"] - train_df["date"].min()).dt.days.to_numpy(np.float32)
    recency_weights = 1.0 + 3.0 * (recency_days / max(1.0, float(recency_days.max())))

    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    sampler = WeightedRandomSampler(
        weights=torch.tensor(recency_weights, dtype=torch.float32),
        num_samples=len(recency_weights),
        replacement=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=sampler)

    model = TradingMLP(input_dim=x_train.shape[1])
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / (y_train.sum() + 1e-8)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    x_val_t = torch.tensor(x_val)
    y_val_t = torch.tensor(y_val)
    best_state = None
    best_val_loss = float("inf")
    patience_left = cfg.patience

    for _ in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(x_val_t), y_val_t).item())
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_left = cfg.patience
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid model state.")
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        val_probs = torch.sigmoid(model(torch.tensor(x_val))).cpu().numpy()
        test_probs = torch.sigmoid(model(torch.tensor(x_test))).cpu().numpy()

    low_thr, high_thr, val_strategy_metrics = _tune_thresholds(
        val_df, val_probs, horizon=cfg.horizon, trade_cost=cfg.trade_cost
    )
    test_strategy_metrics, test_daily = _daily_strategy_metrics(
        test_df, test_probs, low_thr, high_thr, horizon=cfg.horizon, trade_cost=cfg.trade_cost
    )

    report = {
        "tickers_requested": len(cfg.tickers),
        "tickers_used": len(used_tickers),
        "horizon": cfg.horizon,
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "test_samples": int(len(test_df)),
        "classification_test": {
            "accuracy": float(accuracy_score(y_test, (test_probs >= 0.5).astype(np.int32))),
            "f1": float(f1_score(y_test, (test_probs >= 0.5).astype(np.int32))),
            "auc": _safe_auc(y_test, test_probs),
            "base_positive_rate": float(y_test.mean()),
        },
        "thresholds": {"low": float(low_thr), "high": float(high_thr)},
        "strategy_validation": val_strategy_metrics,
        "strategy_test": test_strategy_metrics,
    }

    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_columns": feature_cols,
            "feature_mean": mean.tolist(),
            "feature_std": std.tolist(),
            "horizon": cfg.horizon,
            "threshold_low": float(low_thr),
            "threshold_high": float(high_thr),
            "tickers": used_tickers,
        },
        cfg.model_path,
    )

    Path(cfg.test_daily_path).parent.mkdir(parents=True, exist_ok=True)
    test_daily.to_csv(cfg.test_daily_path, index=False)
    with open(cfg.report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report
