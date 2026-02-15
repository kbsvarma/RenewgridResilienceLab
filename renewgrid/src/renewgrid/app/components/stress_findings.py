"""Deterministic plain-English findings for Phase 2 stress tests."""

from __future__ import annotations


def generate_stress_findings(
    scenario_name: str,
    scenario_meta: dict[str, float | int | str],
    metrics: dict[str, float | int],
) -> list[str]:
    """Generate deterministic stress-test findings bullets."""
    demand_mult = float(scenario_meta.get("demand_multiplier", 1.0))
    solar_derate = float(scenario_meta.get("solar_derate_factor", 1.0))
    wind_derate = float(scenario_meta.get("wind_derate_factor", 1.0))

    findings = [
        (
            f"Under {scenario_name}, demand multiplier is {demand_mult:.2f}, "
            f"solar factor is {solar_derate:.2f}, and wind factor is {wind_derate:.2f}."
        ),
        (
            f"This produced {int(metrics.get('deficit_days', 0))} deficit days and "
            f"{float(metrics.get('total_unserved_mwh', 0.0)):,.0f} MWh of unserved energy."
        ),
        f"Worst-day deficit was {float(metrics.get('peak_unserved_mw', 0.0)):,.0f} MW.",
        (
            f"Longest deficit streak lasted "
            f"{int(metrics.get('max_deficit_streak_days', 0))} day(s)."
        ),
        (
            f"Curtailment totaled {float(metrics.get('curtailment_mwh', 0.0)):,.0f} MWh and "
            f"renewable share was {float(metrics.get('renewable_share', 0.0)) * 100.0:.1f}%."
        ),
        (
            f"Battery reduced unserved energy by "
            f"{float(metrics.get('unserved_reduction_pct', 0.0)):.1f}% versus no-battery baseline."
        ),
    ]
    return findings
