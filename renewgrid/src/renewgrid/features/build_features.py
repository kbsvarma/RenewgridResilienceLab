"""Feature engineering helpers."""

from __future__ import annotations

import pandas as pd


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add basic calendar features derived from the date column."""
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["dayofweek"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month
    return data
