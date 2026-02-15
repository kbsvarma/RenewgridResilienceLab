"""End-to-end tests for stress runner on small synthetic data."""

from __future__ import annotations

import pandas as pd
import pytest

from renewgrid.stress.runner import StressResult, run_stress_test

pytestmark = pytest.mark.phase2


def test_runner_returns_stress_result_with_expected_keys() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=10, freq="D"),
            "demand_mw_avg": [20000, 20500, 21000, 21500, 22000, 21800, 21600, 21400, 21200, 21000],
            "weather_t2m": [8, 9, 10, 11, 12, 11, 10, 9, 8, 7],
            "weather_ws10m": [6, 5, 4, 3, 2, 3, 4, 5, 6, 7],
            "weather_allsky_sfc_sw_dwn": [100, 120, 130, 140, 150, 140, 130, 120, 110, 100],
        }
    )

    result = run_stress_test(
        base_daily_df=frame,
        scenario_id="compound",
        supply_model_name="proxy",
        supply_config={"solar_capacity_mw": 5000.0, "wind_capacity_mw": 7000.0},
        battery_config={
            "energy_mwh": 20000.0,
            "power_mw": 2000.0,
            "roundtrip_efficiency": 0.9,
            "initial_soc_pct": 0.5,
        },
        scenario_config={"wind_drought_days": 3},
    )

    assert isinstance(result, StressResult)
    assert set(["scenario_id", "scenario_name", "metadata", "timeseries", "metrics", "findings"]) <= set(result.__dict__.keys())
    for col in [
        "demand_mw_base",
        "demand_mw_stressed",
        "solar_mw",
        "wind_mw",
        "gen_total_mw",
        "net_load_mw",
        "battery_charge_mw",
        "battery_discharge_mw",
        "soc_mwh",
        "unserved_mw",
        "curtailment_mw",
    ]:
        assert col in result.timeseries.columns
    assert "deficit_days" in result.metrics
    assert "unserved_reduction_pct" in result.metrics
    assert len(result.findings) >= 4
