"""Tests for deterministic story findings summaries."""

from __future__ import annotations

import pandas as pd

from renewgrid.app.components.story_findings import generate_findings


def test_generate_findings_includes_demand_range_and_strong_negative_corr() -> None:
    """Findings should include demand range and strong negative correlation interpretation."""
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=12, freq="D"),
            "demand_mw_avg": [24000, 24500, 25000, 25500, 26000, 25800, 25200, 24800, 24300, 23800, 23300, 22800],
            "weather_t2m": [5.0, 6.0, 7.0, 8.0, 9.0, 9.5, 10.0, 10.5, 11.0, 12.0, 13.0, 14.0],
        }
    )
    bullets = generate_findings(
        df=frame,
        selected_series=["Demand (MW avg)", "Temperature (T2M)"],
        region="CAISO",
        start_date="2026-01-01",
        end_date="2026-01-12",
    )

    joined = " ".join(bullets)
    assert "Demand averaged" in joined
    assert "ranged from" in joined
    assert "colder days tended to coincide with higher demand" in joined.lower()


def test_generate_findings_reports_weak_correlation() -> None:
    """Findings should label weak correlation when absolute value is below threshold."""
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-02-01", periods=12, freq="D"),
            "demand_mw_avg": [20000, 20100, 20200, 20300, 20400, 20500, 20600, 20700, 20800, 20900, 21000, 21100],
            "weather_t2m": [11.0, 12.0, 11.0, 12.0, 11.0, 12.0, 11.0, 12.0, 11.0, 12.0, 11.0, 12.0],
        }
    )
    bullets = generate_findings(
        df=frame,
        selected_series=["Demand (MW avg)", "Temperature (T2M)"],
        region="ERCOT",
        start_date="2026-02-01",
        end_date="2026-02-12",
    )
    joined = " ".join(bullets).lower()
    assert "correlation is" in joined
    assert "(weak)" in joined


def test_generate_findings_reports_insufficient_overlap_for_correlation() -> None:
    """Correlation message should indicate insufficient data with fewer than 10 aligned rows."""
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-03-01", periods=8, freq="D"),
            "demand_mw_avg": [21000, 21200, 21400, 21600, 21800, 22000, 22200, 22400],
            "weather_t2m": [10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5],
        }
    )
    bullets = generate_findings(
        df=frame,
        selected_series=["Demand (MW avg)", "Temperature (T2M)"],
        region="CAISO",
        start_date="2026-03-01",
        end_date="2026-03-08",
    )
    joined = " ".join(bullets).lower()
    assert "correlation not available" in joined
