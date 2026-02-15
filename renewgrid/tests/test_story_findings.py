"""Tests for deterministic story findings summaries."""

from __future__ import annotations

import pandas as pd

from renewgrid.app.components.story_findings import generate_findings


def test_generate_findings_includes_demand_range_and_negative_corr_signal() -> None:
    """Findings should include demand range and negative demand-temperature relationship."""
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=6, freq="D"),
            "demand_mw_avg": [21000, 23000, 25000, 26000, 24000, 22000],
            "weather_t2m": [14.0, 11.0, 8.0, 6.0, 9.0, 12.0],
        }
    )
    bullets = generate_findings(
        df=frame,
        selected_series=["Demand (MW avg)", "Temperature (T2M)"],
        region="CAISO",
        start_date="2026-01-01",
        end_date="2026-01-06",
    )

    joined = " ".join(bullets)
    assert "Demand averaged" in joined
    assert "ranged from" in joined
    assert "Colder days tended to coincide with higher demand." in joined

