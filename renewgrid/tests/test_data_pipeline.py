"""Tests for hello data pipeline outputs."""

from __future__ import annotations

from pathlib import Path

import pytest

import pandas as pd

from renewgrid.data.eia import fetch_rto_hourly
from renewgrid.data import merge


def test_hello_pipeline_writes_expected_schema(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Pipeline writes parquet files with expected minimal schema."""

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

    def fake_eia(*args: object, **kwargs: object) -> pd.DataFrame:
        _ = (args, kwargs)
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01T00:00:00"]),
                "value": [50000.0],
                "source": ["eia"],
                "region": ["ERCO"],
            }
        )

    monkeypatch.setattr(merge, "fetch_daily_solar", fake_nasa)
    monkeypatch.setattr(merge, "fetch_rto_hourly", fake_eia)

    paths = merge.run_hello_pipeline(tmp_path)

    nasa = pd.read_parquet(paths["nasa"])
    eia = pd.read_parquet(paths["eia"])

    assert list(nasa.columns) == ["date", "value", "source", "latitude", "longitude"]
    assert list(eia.columns) == ["date", "value", "source", "region"]
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
