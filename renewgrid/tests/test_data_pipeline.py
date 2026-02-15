"""Tests for hello data pipeline outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from renewgrid.data import merge
from renewgrid.data.eia import aggregate_hourly_to_daily, fetch_rto_daily, fetch_rto_hourly
from renewgrid.util.parquet import has_parquet_engine


def test_hello_pipeline_writes_expected_schema(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Pipeline writes parquet files with expected minimal schema."""
    if not has_parquet_engine():
        pytest.fail(
            "Parquet engine missing. Run `uv sync --extra dev` OR " '`pip install -e ".[dev]"`.'
        )

    def fake_nasa(*args: object, **kwargs: object) -> pd.DataFrame:
        _ = (args, kwargs)
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01"]),
                "value": [3.2],
                "source": ["nasa_power"],
                "latitude": [31.9],
                "longitude": [-99.9],
            }
        )

    def fake_eia_hourly(*args: object, **kwargs: object) -> pd.DataFrame:
        _ = (args, kwargs)
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01T00:00:00"]),
                "value": [50000.0],
                "source": ["eia"],
                "region": ["ERCO"],
            }
        )

    def fake_eia_daily(*args: object, **kwargs: object) -> pd.DataFrame:
        _ = (args, kwargs)
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01"]),
                "demand_mw_avg": [50000.0],
                "source": ["eia"],
                "region": ["ERCO"],
            }
        )

    monkeypatch.setattr(merge, "fetch_daily_solar", fake_nasa)
    monkeypatch.setattr(merge, "fetch_rto_hourly", fake_eia_hourly)
    monkeypatch.setattr(merge, "fetch_rto_daily", fake_eia_daily)

    paths = merge.run_hello_pipeline(tmp_path)

    nasa = pd.read_parquet(paths["nasa"])
    eia = pd.read_parquet(paths["eia"])

    assert list(nasa.columns) == ["date", "value", "source", "latitude", "longitude"]
    assert list(eia.columns) == ["date", "demand_mw_avg", "source", "region"]
    assert nasa.shape[0] == 1
    assert eia.shape[0] == 1


def test_eia_fetch_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """EIA fetch should fail fast when no API key is configured."""
    monkeypatch.delenv("EIA_KEY", raising=False)
    with pytest.raises(ValueError, match="EIA_KEY is not set"):
        fetch_rto_hourly(
            "ERCO",
            pd.Timestamp("2024-01-01").date(),
            pd.Timestamp("2024-01-01").date(),
        )


def test_aggregate_hourly_to_daily_mean() -> None:
    """Hourly demand should aggregate to daily mean by UTC day."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01T00:00:00Z", "2024-01-01T12:00:00Z", "2024-01-02T00:00:00Z"]
            ),
            "value": [100.0, 140.0, 200.0],
            "source": ["eia", "eia", "eia"],
            "region": ["ERCO", "ERCO", "ERCO"],
        }
    )
    daily = aggregate_hourly_to_daily(frame, method="mean")
    assert daily["demand_mw_avg"].tolist() == [120.0, 200.0]
    assert daily["date"].tolist() == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]


def test_fetch_rto_daily_renames_to_demand_mw_avg(monkeypatch: pytest.MonkeyPatch) -> None:
    """Daily EIA fetch should expose demand using canonical demand_mw_avg name."""

    def fake_hourly(*args: object, **kwargs: object) -> pd.DataFrame:
        _ = (args, kwargs)
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T12:00:00Z"]),
                "value": [100.0, 140.0],
                "source": ["eia", "eia"],
                "region": ["ERCO", "ERCO"],
            }
        )

    monkeypatch.setattr("renewgrid.data.eia.fetch_rto_hourly", fake_hourly)
    daily = fetch_rto_daily(
        "ERCOT",
        pd.Timestamp("2024-01-01").date(),
        pd.Timestamp("2024-01-01").date(),
    )
    assert list(daily.columns) == ["date", "demand_mw_avg", "source", "region"]
    assert daily["demand_mw_avg"].iloc[0] == 120.0


def test_build_daily_dataset_schema(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Daily dataset should contain demand and weather features at minimum."""

    def fake_weather(*args: object, **kwargs: object) -> pd.DataFrame:
        _ = (args, kwargs)
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "weather_t2m": [20.0, 21.0],
                "weather_ws10m": [5.0, 5.5],
                "weather_allsky_sfc_sw_dwn": [4.0, 3.8],
            }
        )

    def fake_daily(*args: object, **kwargs: object) -> pd.DataFrame:
        _ = (args, kwargs)
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "demand_mw_avg": [44000.0, 45000.0],
                "source": ["eia", "eia"],
                "region": ["CISO", "CISO"],
            }
        )

    monkeypatch.setattr(merge, "fetch_daily_weather", fake_weather)
    monkeypatch.setattr(merge, "fetch_rto_daily", fake_daily)
    monkeypatch.setattr(merge, "load_rare_daily_generation", lambda *args, **kwargs: None)

    data = merge.build_daily_dataset(
        region="CAISO",
        start_date=pd.Timestamp("2024-01-01").date(),
        end_date=pd.Timestamp("2024-01-02").date(),
        base_dir=tmp_path,
    )
    expected = {
        "date",
        "demand_mw_avg",
        "weather_t2m",
        "weather_ws10m",
        "weather_allsky_sfc_sw_dwn",
    }
    assert expected.issubset(set(data.columns))
