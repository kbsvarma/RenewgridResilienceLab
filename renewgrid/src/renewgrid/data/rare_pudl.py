"""Optional RARE daily generation loader."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)


def normalize_rare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize RARE input to canonical internal columns.

    Preferred canonical output is ``solar_cf`` and ``wind_cf``.
    If only generation is available without capacity, output ``solar_gen_mwh``/``wind_gen_mwh``
    and emit a warning.
    """
    required_base = {"date", "region"}
    if not required_base.issubset(set(frame.columns)):
        raise ValueError("RARE parquet must include date and region columns.")

    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"]).dt.floor("D")

    if {"solar_cf", "wind_cf"}.issubset(set(data.columns)):
        clipped = data.copy()
        for col in ("solar_cf", "wind_cf"):
            raw = pd.to_numeric(clipped[col], errors="coerce")
            if raw.isna().any():
                raise ValueError(f"{col} contains non-numeric values.")
            out_of_bounds = ((raw < 0) | (raw > 1)).sum()
            if out_of_bounds > 0:
                LOGGER.warning("%s had %s out-of-range values; clipping to [0, 1]", col, out_of_bounds)
            clipped[col] = raw.clip(lower=0.0, upper=1.0)
        return clipped[["date", "region", "solar_cf", "wind_cf"]]

    if {"solar_gen", "wind_gen"}.issubset(set(data.columns)):
        if {"solar_capacity", "wind_capacity"}.issubset(set(data.columns)):
            # Assumption: daily generation is in MWh/day and capacity is MW.
            # Daily CF = generation / (capacity * 24h).
            data["solar_cf"] = data["solar_gen"] / (data["solar_capacity"].replace({0: pd.NA}) * 24.0)
            data["wind_cf"] = data["wind_gen"] / (data["wind_capacity"].replace({0: pd.NA}) * 24.0)
            data["solar_cf"] = pd.to_numeric(data["solar_cf"], errors="coerce").clip(lower=0.0, upper=1.0)
            data["wind_cf"] = pd.to_numeric(data["wind_cf"], errors="coerce").clip(lower=0.0, upper=1.0)
            return data[["date", "region", "solar_cf", "wind_cf"]]

        LOGGER.warning("RARE gen provided without capacity; cannot compute capacity factor.")
        data["solar_gen_mwh"] = data["solar_gen"]
        data["wind_gen_mwh"] = data["wind_gen"]
        return data[["date", "region", "solar_gen_mwh", "wind_gen_mwh"]]

    raise ValueError("RARE parquet must include solar/wind CF or generation columns.")


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

    data = normalize_rare_frame(frame)
    mask = (
        (data["region"] == region)
        & (data["date"] >= pd.Timestamp(start_date))
        & (data["date"] <= pd.Timestamp(end_date))
    )
    value_cols = [c for c in data.columns if c not in {"date", "region"}]
    filtered = data.loc[mask, ["date", *value_cols]].sort_values("date").reset_index(drop=True)
    return filtered if not filtered.empty else None
