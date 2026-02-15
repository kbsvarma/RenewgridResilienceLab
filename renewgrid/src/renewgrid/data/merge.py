"""Data merge utilities and hello pipeline orchestration."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from renewgrid.config import REGION_PRESETS
from renewgrid.data.eia import fetch_rto_daily
from renewgrid.data.nasa_power import fetch_daily_solar


def run_hello_pipeline(base_dir: str | Path) -> dict[str, Path]:
    """Run tiny NASA+EIA ingest and save parquet outputs under data/processed."""
    output_dir = Path(base_dir) / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    preset = REGION_PRESETS["ERCOT"]
    start = date(2024, 1, 1)
    end = date(2024, 1, 1)

    nasa = fetch_daily_solar(preset.latitude, preset.longitude, start, end)
    eia = fetch_rto_daily(preset.eia_respondent, start, end)

    nasa_path = output_dir / "nasa_power_daily.parquet"
    eia_path = output_dir / "eia_rto_daily.parquet"
    nasa.to_parquet(nasa_path, index=False)
    eia.to_parquet(eia_path, index=False)

    return {"nasa": nasa_path, "eia": eia_path}


def merge_energy_weather(nasa_frame: pd.DataFrame, eia_frame: pd.DataFrame) -> pd.DataFrame:
    """Outer join weather and demand signals by date."""
    return nasa_frame.merge(eia_frame, on="date", how="outer", suffixes=("_weather", "_grid"))
