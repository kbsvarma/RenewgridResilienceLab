"""Tests for deterministic Phase 2 scenarios."""

from __future__ import annotations

import pandas as pd

from renewgrid.stress.scenarios import apply_scenario, get_scenario


def test_heat_wave_applies_exact_demand_multiplier() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=3, freq="D"),
            "demand_mw_avg": [100.0, 120.0, 140.0],
            "weather_ws10m": [6.0, 7.0, 8.0],
        }
    )
    scenario = get_scenario("heat_wave")
    out = apply_scenario(frame, scenario, {})
    assert out["demand_mw_stressed"].tolist() == [125.0, 150.0, 175.0]


def test_wind_drought_applies_derate_on_deterministic_window_only() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=10, freq="D"),
            "demand_mw_avg": [100.0] * 10,
            "weather_ws10m": [10, 9, 2, 1, 2, 8, 9, 10, 11, 12],
        }
    )
    scenario = get_scenario("wind_drought", wind_drought_days=3)
    out = apply_scenario(frame, scenario, {})

    applied = out["wind_drought_active"].tolist()
    # Lowest 3-day rolling mean is centered on days 3-5 (0-based indices 2..4).
    assert applied[2:5] == [True, True, True]
    assert sum(applied) == 3
    assert all(v == 0.4 for v in out.loc[out["wind_drought_active"], "wind_derate_factor"])
    assert all(v == 1.0 for v in out.loc[~out["wind_drought_active"], "wind_derate_factor"])
