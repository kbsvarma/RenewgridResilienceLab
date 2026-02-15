"""Tests for Phase 2 stress metrics."""

from __future__ import annotations

import pandas as pd

from renewgrid.metrics.stress_metrics import compute_stress_metrics


def test_stress_metrics_deficit_streak_and_unserved() -> None:
    frame = pd.DataFrame(
        {
            "demand_mw_stressed": [100, 100, 100, 100],
            "gen_total_mw": [90, 90, 95, 100],
            "unserved_mw": [10, 10, 5, 0],
            "curtailment_mw": [0, 0, 0, 2],
            "battery_discharge_mw": [0, 0, 0, 0],
        }
    )
    metrics = compute_stress_metrics(frame, {"energy_mwh": 1000.0})
    assert metrics["deficit_days"] == 3
    assert metrics["max_deficit_streak_days"] == 3
    assert metrics["total_unserved_mwh"] == (10 + 10 + 5) * 24
