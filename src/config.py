from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    num_days: int = 365
    seed: int = 42
    target_col: str = "price"
    train_fraction: float = 0.7
    forecast_threshold: float = 5.0
