"""Feature engineering helpers."""

from __future__ import annotations

import pandas as pd


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add basic calendar features derived from the date column."""
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["dow"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month
    return data


def build_feature_frame(frame: pd.DataFrame, target_col: str = "demand_mw_avg") -> pd.DataFrame:
    """Build leakage-safe daily features with lags and rolling means.

    The daily timestep represents average power over the UTC day.
    """
    data = add_time_features(frame).sort_values("date").reset_index(drop=True)

    series_targets = [target_col]
    if "solar_gen" in data.columns and "wind_gen" in data.columns:
        series_targets.extend(["solar_gen", "wind_gen"])
    if "solar_gen_mwh" in data.columns and "wind_gen_mwh" in data.columns:
        series_targets.extend(["solar_gen_mwh", "wind_gen_mwh"])
    if "solar_cf" in data.columns and "wind_cf" in data.columns:
        series_targets.extend(["solar_cf", "wind_cf"])

    for col in series_targets:
        data[f"{col}_lag_1"] = data[col].shift(1)
        data[f"{col}_lag_7"] = data[col].shift(7)
        data[f"{col}_roll_7"] = data[col].shift(1).rolling(7).mean()
        data[f"{col}_roll_14"] = data[col].shift(1).rolling(14).mean()

    return data
