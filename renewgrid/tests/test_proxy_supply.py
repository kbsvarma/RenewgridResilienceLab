"""Tests for proxy supply model behavior."""

from __future__ import annotations

import pandas as pd
import pytest

from renewgrid.supply.proxy import ProxySupplyModel

pytestmark = pytest.mark.phase2


def test_proxy_supply_non_negative_and_bounded() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=5, freq="D"),
            "weather_allsky_sfc_sw_dwn": [50, 150, 250, 350, 450],
            "weather_ws10m": [2, 5, 10, 15, 30],
            "solar_derate_factor": [1.0] * 5,
            "wind_derate_factor": [1.0] * 5,
        }
    )
    model = ProxySupplyModel()
    out = model.generate(frame, {"solar_capacity_mw": 1000.0, "wind_capacity_mw": 500.0})

    assert (out["solar_cf"] >= 0).all() and (out["solar_cf"] <= 1).all()
    assert (out["wind_cf"] >= 0).all() and (out["wind_cf"] <= 1).all()
    assert (out["solar_mw"] >= 0).all()
    assert (out["wind_mw"] >= 0).all()
    assert (out["gen_total_mw"] >= 0).all()
