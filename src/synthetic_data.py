import numpy as np
import pandas as pd


def generate_synthetic_data(
    num_days: int = 365,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic hourly time series with price, load, and temperature.

    The series has:
    - seasonal yearly component
    - daily demand pattern
    - a loose relationship between temperature, load, and price

    Parameters
    ----------
    num_days : int
        Number of days to simulate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by hourly timestamps with columns:
        ["price", "load", "temp"].
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=num_days * 24, freq="h")
    hours = np.arange(len(idx))

    # Yearly seasonal term
    year_term = 10 * np.sin(2 * np.pi * hours / (24 * 365))

    # Daily intraday load shape
    day_term = 15 * np.sin(2 * np.pi * (hours % 24) / 24 - np.pi / 2)

    # Temperature with slow seasonal changes and noise
    temp = (
        10
        + 10 * np.sin(2 * np.pi * (hours - 6) / (24 * 365))
        + rng.normal(0, 2, size=len(idx))
    )

    # Load relates to day pattern and temperature
    load = (
        100
        + day_term
        - 0.7 * (temp - 10)
        + rng.normal(0, 5, size=len(idx))
    )

    # Price loosely linked to load and a seasonal term
    price = (
        30
        + 0.8 * load
        + year_term
        + rng.normal(0, 8, size=len(idx))
    )

    df = pd.DataFrame(
        {"price": price, "load": load, "temp": temp},
        index=idx,
    )
    df.index.name = "timestamp"
    return df
