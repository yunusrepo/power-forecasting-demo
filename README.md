# Power Forecasting Demo

This repository contains a compact, self-contained example of a short-term
power price forecasting and backtesting pipeline using fully synthetic data.

The goal is to illustrate my general modelling and engineering style:
data generation, feature engineering, model training, and a simple
signal-based backtest, written in clean, production-oriented Python.

## Structure

- `src/synthetic_data.py` – generates a synthetic hourly time series with
  load, temperature, and price.
- `src/features.py` – builds lag and rolling-window features.
- `src/model.py` – trains a gradient boosting model and reports metrics.
- `src/backtest.py` – runs a simple rule-based trading backtest based on
  the model forecasts.
- `scripts/train_and_backtest.py` – end-to-end script that ties everything
  together and prints results.

All data is synthetic and does not reflect any real markets or
proprietary trading logic.

## Quick start

```bash
pip install -r requirements.txt
python scripts/train_and_backtest.py
