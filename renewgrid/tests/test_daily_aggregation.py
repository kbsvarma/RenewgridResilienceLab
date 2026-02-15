"""Tests for daily aggregation and UTC boundary contract."""

from __future__ import annotations

import pandas as pd

from renewgrid.data.eia import aggregate_hourly_to_daily
from renewgrid.data.merge import utc_daily_index


def test_daily_mean_aggregation_across_utc_midnight() -> None:
    """Hourly values crossing UTC midnight should aggregate into correct day means."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01T23:00:00Z",
                    "2026-01-02T00:00:00Z",
                    "2026-01-02T01:00:00Z",
                ]
            ),
            "value": [10.0, 20.0, 40.0],
            "source": ["eia", "eia", "eia"],
            "region": ["ERCO", "ERCO", "ERCO"],
        }
    )
    daily = aggregate_hourly_to_daily(frame, method="mean")
    assert daily["date"].tolist() == [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")]
    assert daily["demand_mw_avg"].tolist() == [10.0, 30.0]


def test_utc_daily_index_is_utc_localized() -> None:
    """UTC helper should localize naive timestamps to UTC and keep UTC contract."""
    dates = pd.Series(pd.to_datetime(["2026-01-01", "2026-01-02"]))
    idx = utc_daily_index(dates)
    assert str(idx.tz) == "UTC"

