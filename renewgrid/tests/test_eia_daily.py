"""Tests for EIA daily aggregation behavior."""

from __future__ import annotations

import pandas as pd
import pytest

from renewgrid.data.eia import aggregate_hourly_to_daily, fetch_rto_daily


def test_aggregate_hourly_to_daily_mean() -> None:
    """Hourly values should aggregate to UTC daily mean demand_mw_avg."""
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


def test_fetch_rto_daily_uses_demand_mw_avg(monkeypatch: pytest.MonkeyPatch) -> None:
    """Daily fetch wrapper should expose demand_mw_avg output schema."""

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
    daily = fetch_rto_daily("ERCOT", pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-01-01").date())
    assert list(daily.columns) == ["date", "demand_mw_avg", "source", "region"]
