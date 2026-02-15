"""Baseline forecasting model."""

from __future__ import annotations

import pandas as pd


def persistence_forecast(frame: pd.DataFrame, target_col: str = "value") -> pd.Series:
    """Return one-step persistence forecast from prior observation."""
    return frame[target_col].shift(1).bfill()
