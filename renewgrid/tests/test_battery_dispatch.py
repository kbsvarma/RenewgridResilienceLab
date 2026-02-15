"""Tests for deterministic battery dispatch simulation."""

from __future__ import annotations

import pandas as pd

from renewgrid.dispatch.battery import simulate_battery_dispatch


def test_battery_soc_bounds_and_zero_battery_no_discharge() -> None:
    frame = pd.DataFrame({"net_load_mw": [50.0, -20.0, 30.0, -10.0]})
    out = simulate_battery_dispatch(
        frame,
        {
            "energy_mwh": 0.0,
            "power_mw": 0.0,
            "roundtrip_efficiency": 0.9,
            "initial_soc_pct": 0.5,
        },
    )
    assert (out["soc_mwh"] == 0.0).all()
    assert (out["battery_discharge_mw"] == 0.0).all()


def test_larger_battery_reduces_unserved_energy() -> None:
    frame = pd.DataFrame({"net_load_mw": [100.0, 100.0, -150.0, 100.0, 100.0]})
    small = simulate_battery_dispatch(
        frame,
        {
            "energy_mwh": 0.0,
            "power_mw": 0.0,
            "roundtrip_efficiency": 0.9,
            "initial_soc_pct": 0.0,
        },
    )
    large = simulate_battery_dispatch(
        frame,
        {
            "energy_mwh": 5000.0,
            "power_mw": 500.0,
            "roundtrip_efficiency": 0.9,
            "initial_soc_pct": 0.5,
        },
    )
    assert float((large["unserved_mw"] * 24).sum()) < float((small["unserved_mw"] * 24).sum())
    assert (large["soc_mwh"] >= 0).all()
    assert (large["soc_mwh"] <= 5000.0).all()
