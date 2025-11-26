from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


@dataclass
class BacktestResult:
    pnl: pd.Series
    equity: pd.Series
    stats: Dict[str, float]


def run_backtest(
    feat: pd.DataFrame,
    model: GradientBoostingRegressor,
    threshold: float = 5.0,
) -> BacktestResult:
    """
    Run a simple long/short backtest based on model forecasts.

    Strategy:
        If forecast > current_price + threshold -> go long 1 unit
        If forecast < current_price - threshold -> go short 1 unit
        Otherwise -> stay flat

    PnL is computed as position_t * (price_{t+1} - price_t).

    Parameters
    ----------
    feat : pd.DataFrame
        Feature DataFrame containing the target "price".
    model : GradientBoostingRegressor
        Trained model used to generate forecasts.
    threshold : float
        Entry threshold in price units.

    Returns
    -------
    BacktestResult
        PnL series, equity curve, and summary statistics.
    """
    if "price" not in feat.columns:
        raise ValueError("feat must contain a 'price' column for the target")

    X = feat.drop(columns=["price"])
    y = feat["price"]

    forecasts = model.predict(X)
    prices = y.values

    positions = np.zeros(len(prices))
    for t in range(len(prices) - 1):
        if forecasts[t] > prices[t] + threshold:
            positions[t] = 1.0
        elif forecasts[t] < prices[t] - threshold:
            positions[t] = -1.0
        else:
            positions[t] = 0.0

    pnl = positions[:-1] * (prices[1:] - prices[:-1])
    pnl_series = pd.Series(pnl, index=feat.index[:-1], name="pnl")
    equity = pnl_series.cumsum()
    equity.name = "equity"

    stats = {
        "pnl_sum": float(pnl_series.sum()),
        "pnl_mean": float(pnl_series.mean()),
        "pnl_std": float(pnl_series.std()),
        "num_trades": int(np.count_nonzero(positions)),
    }

    return BacktestResult(pnl=pnl_series, equity=equity, stats=stats)
