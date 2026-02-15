"""Baseline forecasting model."""

from __future__ import annotations

import pandas as pd


def persistence_forecast(frame: pd.DataFrame, target_col: str = "value") -> pd.Series:
    """Return one-step persistence forecast from prior observation."""
    return frame[target_col].shift(1).bfill()


def persistence_predict_from_train(
    train_frame: pd.DataFrame,
    target_col: str,
    horizon: int,
) -> float:
    """Predict future value as the most recent observed training value."""
    _ = horizon
    return float(train_frame[target_col].iloc[-1])
