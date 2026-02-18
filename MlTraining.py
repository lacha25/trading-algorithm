import argparse
import json

from analysisTools.TradingMLP import DEFAULT_TICKERS, TradingMLPConfig, train_trading_mlp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLP models for trading.")
    parser.add_argument(
        "--tickers",
        default=",".join(DEFAULT_TICKERS),
        help="Comma-separated ticker list.",
    )
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon in trading days.")
    parser.add_argument("--train-end", default="2023-01-01", help="Train split end date (YYYY-MM-DD).")
    parser.add_argument("--val-end", default="2025-01-01", help="Validation split end date (YYYY-MM-DD).")
    parser.add_argument("--epochs", type=int, default=35, help="Maximum training epochs.")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience.")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=8e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--trade-cost", type=float, default=4e-4, help="Per-trade cost.")
    parser.add_argument("--cache-dir", default="analysisTools/cache", help="Cache directory for price data.")
    parser.add_argument("--model-path", default="model_weights_trading_mlp.pth", help="Output model weights path.")
    parser.add_argument("--report-path", default="mlp_trading_report.json", help="Output metrics report path.")
    parser.add_argument(
        "--test-daily-path",
        default="mlp_trading_test_daily.csv",
        help="Output CSV path for test daily strategy returns.",
    )
    parser.add_argument("--refresh-cache", action="store_true", help="Refresh cached market data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = [ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()]
    config = TradingMLPConfig(
        tickers=tickers,
        horizon=args.horizon,
        train_end=args.train_end,
        val_end=args.val_end,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        trade_cost=args.trade_cost,
        cache_dir=args.cache_dir,
        model_path=args.model_path,
        report_path=args.report_path,
        test_daily_path=args.test_daily_path,
        refresh_cache=args.refresh_cache,
    )
    report = train_trading_mlp(config)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
