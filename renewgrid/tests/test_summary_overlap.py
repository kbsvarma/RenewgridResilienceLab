"""Tests for overlap-aware summary findings."""

from __future__ import annotations

import pandas as pd

from renewgrid.app.components.story_findings import generate_findings


def test_findings_include_overlap_count_for_correlation() -> None:
    """Correlation bullet should include overlap count when enough aligned rows exist."""
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=12, freq="D"),
            "demand_mw_avg": [24000, 23800, 23600, 23500, 23300, 23200, 23100, 23000, 22900, 22800, 22700, 22600],
            "weather_t2m": [10.0, 10.2, 10.3, 10.5, 10.7, 10.8, 11.0, 11.2, 11.4, 11.5, 11.7, 11.9],
        }
    )
    bullets = generate_findings(
        df=frame,
        selected_series=["Demand (MW avg)", "Temperature (T2M)"],
        region="CAISO",
        start_date="2026-01-01",
        end_date="2026-01-12",
    )
    joined = " ".join(bullets).lower()
    assert "overlapping days" in joined

