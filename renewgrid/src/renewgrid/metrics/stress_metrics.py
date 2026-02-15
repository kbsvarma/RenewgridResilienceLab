"""Phase 2 stress-test metrics for supply-demand resilience diagnostics."""

from __future__ import annotations

import pandas as pd


def _longest_true_streak(mask: pd.Series) -> int:
    max_streak = 0
    current = 0
    for flag in mask.astype(bool):
        if flag:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def compute_stress_metrics(
    timeseries: pd.DataFrame,
    battery_config: dict[str, float],
) -> dict[str, float | int]:
    """Compute Phase 2 stress metrics from simulated timeseries."""
    unserved_mw = pd.to_numeric(timeseries.get("unserved_mw", 0.0), errors="coerce").fillna(0.0)
    curtailment_mw = pd.to_numeric(timeseries.get("curtailment_mw", 0.0), errors="coerce").fillna(0.0)
    gen_total_mw = pd.to_numeric(timeseries.get("gen_total_mw", 0.0), errors="coerce").fillna(0.0)
    demand_mw = pd.to_numeric(timeseries.get("demand_mw_stressed", 0.0), errors="coerce").fillna(0.0)
    discharge_mw = pd.to_numeric(timeseries.get("battery_discharge_mw", 0.0), errors="coerce").fillna(0.0)

    deficit_mask = unserved_mw > 0
    energy_mwh = max(0.0, float(battery_config.get("energy_mwh", 0.0)))

    total_discharge_mwh = float((discharge_mw * 24.0).sum())
    total_demand = float(demand_mw.sum())

    return {
        "deficit_days": int(deficit_mask.sum()),
        "total_unserved_mwh": float((unserved_mw * 24.0).sum()),
        "peak_unserved_mw": float(unserved_mw.max() if not unserved_mw.empty else 0.0),
        "max_deficit_streak_days": int(_longest_true_streak(deficit_mask)),
        "curtailment_mwh": float((curtailment_mw * 24.0).sum()),
        "renewable_share": float(gen_total_mw.sum() / total_demand) if total_demand > 0 else 0.0,
        "battery_utilization_proxy": float(total_discharge_mwh / energy_mwh) if energy_mwh > 0 else 0.0,
    }
