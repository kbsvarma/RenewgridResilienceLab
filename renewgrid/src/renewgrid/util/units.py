"""Unit conventions and helpers for daily power/energy values."""

from __future__ import annotations

import pandas as pd

ScalarOrSeries = float | pd.Series
HOURS_PER_DAY = 24.0


def mw_avg_to_mwh(mw_avg: ScalarOrSeries) -> ScalarOrSeries:
    """Convert daily average power (MW) to daily energy (MWh)."""
    return mw_avg * HOURS_PER_DAY


def mwh_to_mw_avg(mwh: ScalarOrSeries) -> ScalarOrSeries:
    """Convert daily energy (MWh) to daily average power (MW)."""
    return mwh / HOURS_PER_DAY


def assert_daily_mw_avg(df: pd.DataFrame, col: str = "demand_mw_avg") -> None:
    """Validate daily average MW column contract for Phase 1/2 datasets."""
    if col not in df.columns:
        raise ValueError(f"Missing required daily MW-average column: {col}")

    series = pd.to_numeric(df[col], errors="coerce")
    if series.isna().any():
        raise ValueError(f"Column {col} contains non-numeric or null values.")

    if (series < 0).any():
        raise ValueError(f"Column {col} contains negative values, invalid for demand MW average.")
