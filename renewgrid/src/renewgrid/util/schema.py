"""Schema contracts for daily modeling datasets."""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {"date", "demand_mw_avg"}
OPTIONAL_COLUMNS = {"solar_cf", "wind_cf", "solar_gen_mwh", "wind_gen_mwh"}


def validate_daily_frame(df: pd.DataFrame, strict: bool = False) -> None:
    """Validate daily dataset contract for Phase 0/1 artifacts."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("date column contains unparseable values")

    if dates.duplicated().any():
        raise ValueError("date column contains duplicate entries")

    if not dates.is_monotonic_increasing:
        raise ValueError("date column must be sorted in ascending order")

    demand = pd.to_numeric(df["demand_mw_avg"], errors="coerce")
    if demand.isna().any() or (demand < 0).any():
        raise ValueError("demand_mw_avg must be numeric and non-negative")

    for cf_col in ("solar_cf", "wind_cf"):
        if cf_col in df.columns:
            cf = pd.to_numeric(df[cf_col], errors="coerce")
            if cf.isna().any() or ((cf < -1e-6) | (cf > 1 + 1e-6)).any():
                raise ValueError(f"{cf_col} must be within [0, 1]")

    for gen_col in ("solar_gen_mwh", "wind_gen_mwh"):
        if gen_col in df.columns:
            gen = pd.to_numeric(df[gen_col], errors="coerce")
            if gen.isna().any() or (gen < 0).any():
                raise ValueError(f"{gen_col} must be numeric and non-negative")

    if strict:
        known = REQUIRED_COLUMNS | OPTIONAL_COLUMNS
        known |= {c for c in df.columns if c.startswith("weather_")}
        unknown = set(df.columns) - known
        if unknown:
            raise ValueError(f"Unknown columns in strict mode: {sorted(unknown)}")
