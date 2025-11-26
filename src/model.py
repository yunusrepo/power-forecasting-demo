from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score


@dataclass
class ModelResult:
    model: GradientBoostingRegressor
    metrics: Dict[str, float]


def train_model(
    feat: pd.DataFrame,
    target_col: str = "price",
    train_fraction: float = 0.7,
) -> ModelResult:
    """
    Train a gradient boosting regression model on the feature DataFrame.

    Parameters
    ----------
    feat : pd.DataFrame
        Feature matrix including the target column.
    target_col : str
        Name of the target column.
    train_fraction : float
        Fraction of samples used for training, the rest is test.

    Returns
    -------
    ModelResult
        Model and basic evaluation metrics on the test set.
    """
    X = feat.drop(columns=[target_col])
    y = feat[target_col]

    split_idx = int(len(feat) * train_fraction)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return ModelResult(
        model=model,
        metrics={"mae": float(mae), "r2": float(r2)},
    )
