"""Data merge utilities and hello pipeline orchestration."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd

from renewgrid.config import REGION_PRESETS, load_environment
from renewgrid.data.eia import fetch_rto_daily, fetch_rto_hourly
from renewgrid.data.nasa_power import fetch_daily_solar, fetch_daily_weather
from renewgrid.data.rare_pudl import load_rare_daily_generation
from renewgrid.util.parquet import require_parquet_engine


def run_hello_pipeline(base_dir: str | Path) -> dict[str, Path]:
    """Run tiny NASA+EIA ingest and save parquet outputs under data/processed."""
    require_parquet_engine()

    output_dir = Path(base_dir) / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    preset = REGION_PRESETS["ERCOT"]
    start = date(2024, 1, 1)
    end = date(2024, 1, 1)

    nasa = fetch_daily_solar(preset.latitude, preset.longitude, start, end)
    eia = fetch_rto_hourly(preset.eia_respondent, start, end)

    nasa_path = output_dir / "nasa_power_daily.parquet"
    eia_path = output_dir / "eia_rto_hourly.parquet"
    nasa.to_parquet(nasa_path, index=False)
    eia.to_parquet(eia_path, index=False)

    return {"nasa": nasa_path, "eia": eia_path}


def merge_energy_weather(nasa_frame: pd.DataFrame, eia_frame: pd.DataFrame) -> pd.DataFrame:
    """Outer join weather and demand signals by date."""
    return nasa_frame.merge(eia_frame, on="date", how="outer", suffixes=("_weather", "_grid"))


def build_daily_dataset(
    region: Literal["CAISO", "ERCOT"],
    start_date: date,
    end_date: date,
    latitude: float | None = None,
    longitude: float | None = None,
    base_dir: str | Path = ".",
    rare_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build and persist a tidy daily dataset for one region.

    Demand is aggregated to daily average MW (``demand_mw_avg``) using UTC-day grouping.
    """
    require_parquet_engine()
    preset = REGION_PRESETS[region]
    lat = latitude if latitude is not None else preset.latitude
    lon = longitude if longitude is not None else preset.longitude

    weather = fetch_daily_weather(lat, lon, start_date, end_date)
    demand = fetch_rto_daily(region, start_date, end_date, method="mean").rename(
        columns={"value": "demand_mw_avg"}
    )

    dataset = demand.merge(
        weather[[c for c in weather.columns if c == "date" or c.startswith("weather_")]],
        on="date",
        how="left",
    ).sort_values("date")
    dataset["region"] = region

    if rare_path is not None:
        rare = load_rare_daily_generation(region, start_date, end_date, rare_path)
        if rare is not None:
            dataset = dataset.merge(rare, on="date", how="left")

    output_dir = Path(base_dir) / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{region}_daily_{start_date.isoformat()}_{end_date.isoformat()}.parquet"
    dataset.to_parquet(out_path, index=False)
    return dataset.reset_index(drop=True)


def main() -> None:
    """Run the hello pipeline from the project root."""
    base_dir = Path(__file__).resolve().parents[3]
    load_environment(base_dir / ".env")
    outputs = run_hello_pipeline(base_dir)
    print(f"Wrote NASA data: {outputs['nasa']}")
    print(f"Wrote EIA data: {outputs['eia']}")


if __name__ == "__main__":
    main()
