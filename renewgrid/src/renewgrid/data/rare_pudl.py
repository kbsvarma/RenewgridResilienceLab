"""Optional RARE daily generation loader."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)


def load_rare_daily_generation(
    region: str,
    start_date: date,
    end_date: date,
    path: str | Path,
) -> pd.DataFrame | None:
    """Load optional local RARE daily generation for a region.

    Expected parquet columns:
    - ``date``
    - ``region``
    - ``solar_gen`` and ``wind_gen`` (or solar_cf/wind_cf alternatives)
    """
    parquet_path = Path(path)
    if not parquet_path.exists():
        LOGGER.warning("RARE file not found at %s; continuing without RARE merge", parquet_path)
        return None

    frame = pd.read_parquet(parquet_path)
    if frame.empty:
        return None

    required_base = {"date", "region"}
    if not required_base.issubset(set(frame.columns)):
        raise ValueError("RARE parquet must include date and region columns.")

    value_cols = {"solar_gen", "wind_gen"} if {"solar_gen", "wind_gen"}.issubset(
        set(frame.columns)
    ) else {"solar_cf", "wind_cf"} if {"solar_cf", "wind_cf"}.issubset(set(frame.columns)) else None
    if value_cols is None:
        raise ValueError("RARE parquet must include solar/wind generation or capacity-factor columns.")

    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"]).dt.floor("D")
    mask = (
        (data["region"] == region)
        & (data["date"] >= pd.Timestamp(start_date))
        & (data["date"] <= pd.Timestamp(end_date))
    )
    filtered = data.loc[mask, ["date", *sorted(value_cols)]].sort_values("date").reset_index(drop=True)
    return filtered if not filtered.empty else None
