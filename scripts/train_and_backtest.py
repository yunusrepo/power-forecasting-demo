from pathlib import Path

import src
from src.config import ExperimentConfig
from src.synthetic_data import generate_synthetic_data
from src.features import build_features
from src.model import train_model
from src.backtest import run_backtest


def main() -> None:
    cfg = ExperimentConfig()

    print("Generating synthetic data...")
    df = generate_synthetic_data(num_days=cfg.num_days, seed=cfg.seed)

    print("Building features...")
    feat = build_features(df)

    print("Training model...")
    result = train_model(
        feat,
        target_col=cfg.target_col,
        train_fraction=cfg.train_fraction,
    )

    print("Model metrics:")
    for k, v in result.metrics.items():
        print(f"  {k}: {v:.4f}")

    print("Running backtest...")
    bt = run_backtest(
        feat,
        model=result.model,
        threshold=cfg.forecast_threshold,
    )

    print("Backtest stats:")
    for k, v in bt.stats.items():
        print(f"  {k}: {v:.4f}")

    # To save csv outputs for quick eye check
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    bt.pnl.to_csv(out_dir / "pnl.csv")
    bt.equity.to_csv(out_dir / "equity.csv")
    print(f"Saved PnL and equity to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
