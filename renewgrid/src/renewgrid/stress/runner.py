"""Phase 2 stress-test orchestration runner."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from renewgrid.app.components.stress_findings import generate_stress_findings
from renewgrid.dispatch.battery import simulate_battery_dispatch
from renewgrid.metrics.stress_metrics import compute_stress_metrics
from renewgrid.supply.proxy import ProxySupplyModel
from renewgrid.stress.scenarios import apply_scenario, get_scenario


@dataclass
class StressResult:
    """Structured output for a single deterministic stress simulation run."""

    scenario_id: str
    scenario_name: str
    metadata: dict
    timeseries: pd.DataFrame
    metrics: dict
    findings: list[str]


def run_stress_test(
    base_daily_df: pd.DataFrame,
    scenario_id: str,
    supply_model_name: str = "proxy",
    supply_config: dict[str, float] | None = None,
    battery_config: dict[str, float] | None = None,
    scenario_config: dict[str, float | int] | None = None,
) -> StressResult:
    """Run deterministic scenario->supply->dispatch->metrics pipeline."""
    if base_daily_df.empty:
        raise ValueError("base_daily_df must be non-empty")
    if "demand_mw_avg" not in base_daily_df.columns:
        raise ValueError("base_daily_df must include demand_mw_avg")

    supply_cfg = dict(supply_config or {})
    batt_cfg = dict(battery_config or {})
    scenario_cfg = dict(scenario_config or {})

    scenario = get_scenario(
        scenario_id=scenario_id,
        wind_drought_days=int(scenario_cfg.get("wind_drought_days", 5)),
    )

    working = base_daily_df.copy()
    working["date"] = pd.to_datetime(working["date"])
    working = working.sort_values("date").reset_index(drop=True)
    working["demand_mw_base"] = pd.to_numeric(working["demand_mw_avg"], errors="coerce")

    stressed = apply_scenario(working, scenario, scenario_cfg)

    if supply_model_name != "proxy":
        raise ValueError(f"Unsupported supply_model_name: {supply_model_name}")

    supply_model = ProxySupplyModel()
    supply = supply_model.generate(stressed, supply_cfg)
    warnings = list(supply.attrs.get("warnings", []))

    timeseries = stressed.copy()
    for col in ["solar_cf", "wind_cf", "solar_mw", "wind_mw", "gen_total_mw"]:
        timeseries[col] = supply[col]

    timeseries["net_load_mw"] = timeseries["demand_mw_stressed"] - timeseries["gen_total_mw"]
    timeseries = simulate_battery_dispatch(timeseries, batt_cfg)

    metrics = compute_stress_metrics(timeseries, batt_cfg)

    baseline_cfg = {
        "energy_mwh": 0.0,
        "power_mw": 0.0,
        "roundtrip_efficiency": batt_cfg.get("roundtrip_efficiency", 0.9),
        "initial_soc_pct": 0.0,
    }
    no_battery = simulate_battery_dispatch(
        timeseries[[c for c in timeseries.columns if c not in {
            "battery_charge_mw",
            "battery_discharge_mw",
            "soc_mwh",
            "unserved_mw",
            "curtailment_mw",
        }]].copy(),
        baseline_cfg,
    )
    baseline_unserved_mwh = float((pd.to_numeric(no_battery["unserved_mw"], errors="coerce").fillna(0.0) * 24.0).sum())
    current_unserved_mwh = float(metrics["total_unserved_mwh"])
    if baseline_unserved_mwh > 0:
        reduction_pct = 100.0 * (baseline_unserved_mwh - current_unserved_mwh) / baseline_unserved_mwh
    else:
        reduction_pct = 0.0
    metrics["unserved_reduction_pct"] = float(reduction_pct)

    scenario_meta = {
        "demand_multiplier": float(
            scenario_cfg.get("demand_multiplier", scenario.params.demand_multiplier)
        ),
        "solar_derate_factor": float(
            scenario_cfg.get("solar_derate_factor", scenario.params.solar_derate_factor)
        ),
        "wind_derate_factor": float(
            scenario_cfg.get("wind_derate_factor", scenario.params.wind_derate_factor)
        ),
        "wind_drought_days": int(
            scenario_cfg.get("wind_drought_days", scenario.params.wind_drought_days)
        ),
    }
    findings = generate_stress_findings(scenario.name, scenario_meta, metrics)

    metadata = {
        "scenario": scenario_meta,
        "supply_model": supply_model_name,
        "supply_config": supply_cfg,
        "battery_config": batt_cfg,
        "warnings": warnings,
        "start_date": str(timeseries["date"].min().date()),
        "end_date": str(timeseries["date"].max().date()),
    }

    return StressResult(
        scenario_id=scenario.scenario_id,
        scenario_name=scenario.name,
        metadata=metadata,
        timeseries=timeseries,
        metrics=metrics,
        findings=findings,
    )
