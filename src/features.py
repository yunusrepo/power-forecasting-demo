from __future__ import annotations

from typing import Iterable
import pandas as pd


def build_features(
    df: pd.DataFrame,
    lags: Iterable[int] = (1, 2, 3, 24),
    rolling_windows: Iterable[int] = (3, 24),
) -> pd.DataFrame:
    """
    Build lag and rolling features from a base time series.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with columns ["price", "load", "temp"].
    lags : Iterable[int]
        Lag steps to include for each base column.
    rolling_windows : Iterable[int]
        Window sizes for rolling means.

    Returns
    -------
    pd.DataFrame
        Feature-enriched DataFrame, with NaN rows dropped.
    """
    feat = df.copy()

    # Lagged features
    base_cols = ["price", "load", "temp"]
    for lag in lags:
        for col in base_cols:
            feat[f"{col}_lag{lag}"] = feat[col].shift(lag)

    # Rolling statistics on price and load
    for win in rolling_windows:
        feat[f"price_roll_mean_{win}"] = feat["price"].shift(1).rolling(win).mean()
        feat[f"load_roll_mean_{win}"] = feat["load"].shift(1).rolling(win).mean()

    # Drop initial NaN rows created by lags and rolling
    feat = feat.dropna()
    return feat
