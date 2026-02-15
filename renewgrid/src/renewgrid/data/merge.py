"""Data merge utilities and hello pipeline orchestration."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd

from renewgrid.config import REGION_PRESETS, load_environment
from renewgrid.data.eia import fetch_rto_daily, fetch_rto_hourly
from renewgrid.data.nasa_power import fetch_daily_solar, fetch_daily_weather
from renewgrid.data.rare_pudl import load_rare_daily_generation
from renewgrid.util.parquet import require_parquet_engine
from renewgrid.util.schema import validate_daily_frame
from renewgrid.util.units import assert_daily_mw_avg

LOGGER = logging.getLogger(__name__)


def utc_daily_index(values: pd.Series) -> pd.DatetimeIndex:
    """Build a UTC-localized daily DatetimeIndex and assert UTC boundary contract."""
    idx = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce"))
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    assert str(idx.tz) == "UTC", "Daily aggregation must use UTC day boundaries."
    return idx


def _enforce_utc_daily(frame: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Ensure daily timestamps are interpreted on UTC day boundaries."""
    data = frame.copy()
    idx = utc_daily_index(data[date_col])
    data = data.drop(columns=[date_col]).set_index(idx)
    logging.info(
        "Daily UTC aggregation from %s to %s",
        data.index.min().date(),
        data.index.max().date(),
    )
    data = data.reset_index().rename(columns={"index": date_col})
    data[date_col] = pd.to_datetime(data[date_col]).dt.tz_localize(None)
    return data


def run_hello_pipeline(base_dir: str | Path) -> dict[str, Path]:
    """Run tiny NASA+EIA ingest and save daily parquet outputs under data/processed.

    The daily timestep represents average power over the UTC day.
    """
    require_parquet_engine()

    output_dir = Path(base_dir) / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    preset = REGION_PRESETS["ERCOT"]
    start = date(2024, 1, 1)
    end = date(2024, 1, 1)

    nasa = fetch_daily_solar(preset.latitude, preset.longitude, start, end)
    eia_hourly = fetch_rto_hourly(preset.eia_respondent, start, end)
    eia = _enforce_utc_daily(fetch_rto_daily("ERCOT", start, end, method="mean"))

    nasa_path = output_dir / "nasa_power_daily.parquet"
    eia_path = output_dir / "eia_rto_daily.parquet"
    nasa.to_parquet(nasa_path, index=False)
    validate_daily_frame(eia)
    assert_daily_mw_avg(eia)
    eia.to_parquet(eia_path, index=False)

    raw_dir = Path(base_dir) / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    eia_hourly.to_parquet(raw_dir / "eia_rto_hourly_debug.parquet", index=False)

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
    preset = REGION_PRESETS[region]
    lat = latitude if latitude is not None else preset.latitude
    lon = longitude if longitude is not None else preset.longitude

    weather = fetch_daily_weather(lat, lon, start_date, end_date)
    demand = _enforce_utc_daily(fetch_rto_daily(region, start_date, end_date, method="mean"))
    assert_daily_mw_avg(demand)

    dataset = demand.merge(
        weather[[c for c in weather.columns if c == "date" or c.startswith("weather_")]],
        on="date",
        how="left",
    ).sort_values("date")
    dataset["region"] = region

    if rare_path is not None:
        rare = load_rare_daily_generation(region, start_date, end_date, rare_path)
        if rare is not None:
            if {"solar_cf", "wind_cf"}.issubset(set(rare.columns)):
                LOGGER.info("Merging RARE capacity-factor columns for %s", region)
            elif {"solar_gen_mwh", "wind_gen_mwh"}.issubset(set(rare.columns)):
                LOGGER.info("Merging RARE generation-energy columns for %s", region)
            dataset = dataset.merge(rare, on="date", how="left")

    validate_daily_frame(dataset)
    assert_daily_mw_avg(dataset)

    output_dir = Path(base_dir) / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        output_dir / f"{region}_daily_{start_date.isoformat()}_{end_date.isoformat()}.parquet"
    )
    require_parquet_engine()
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
