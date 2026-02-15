"""Phase 2 Stress Test page for deterministic scenario simulation."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from renewgrid.app.components.stress_charts import render_stress_charts
from renewgrid.config import load_environment
from renewgrid.data.merge import build_daily_dataset
from renewgrid.stress.runner import run_stress_test

SCENARIO_LABEL_TO_ID = {
    "Heat Wave": "heat_wave",
    "Wind Drought": "wind_drought",
    "Demand Shock": "demand_shock",
    "Compound": "compound",
}


def _default_window() -> tuple[date, date]:
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=89)
    return start_date, end_date


@st.cache_data(ttl=600)
def _load_base_daily(
    region: str,
    start_date: date,
    end_date: date,
    base_dir: str,
    rare_path: str | None,
) -> pd.DataFrame:
    return build_daily_dataset(
        region=region,
        start_date=start_date,
        end_date=end_date,
        base_dir=base_dir,
        rare_path=rare_path,
    )


def render_stress_test(base_dir: Path) -> None:
    """Render Stress Test controls, outputs, findings, and downloads."""
    st.subheader("Stress Test (Phase 2)")
    st.caption("Deterministic daily stress simulation using public data and proxy supply estimates.")

    run_config = st.session_state.get("run_config")
    default_region = run_config.region if run_config is not None else "CAISO"
    if run_config is not None:
        default_start = run_config.start_date
        default_end = run_config.end_date
    else:
        default_start, default_end = _default_window()

    col_a, col_b = st.columns(2)
    region = col_a.selectbox("Region", ["CAISO", "ERCOT"], index=0 if default_region == "CAISO" else 1)
    scenario_label = col_b.selectbox("Stress scenario", list(SCENARIO_LABEL_TO_ID.keys()))

    col_c, col_d = st.columns(2)
    start_date = col_c.date_input("Start date", value=default_start)
    end_date = col_d.date_input("End date", value=default_end)

    drought_days = 5
    if scenario_label in {"Wind Drought", "Compound"}:
        drought_days = st.slider("Wind drought days", min_value=2, max_value=14, value=5)

    st.markdown("### Supply Configuration")
    defaults = {
        "CAISO": {"solar": 20000.0, "wind": 15000.0},
        "ERCOT": {"solar": 25000.0, "wind": 35000.0},
    }
    col_e, col_f = st.columns(2)
    solar_capacity_mw = col_e.slider(
        "Solar capacity (MW)",
        min_value=0.0,
        max_value=100000.0,
        value=float(defaults[region]["solar"]),
        step=500.0,
    )
    wind_capacity_mw = col_f.slider(
        "Wind capacity (MW)",
        min_value=0.0,
        max_value=100000.0,
        value=float(defaults[region]["wind"]),
        step=500.0,
    )

    st.markdown("### Battery Configuration")
    col_g, col_h = st.columns(2)
    energy_mwh = col_g.slider("Battery energy (MWh)", 0.0, 500000.0, 50000.0, step=1000.0)
    power_mw = col_h.slider("Battery power (MW)", 0.0, 50000.0, 5000.0, step=100.0)
    col_i, col_j = st.columns(2)
    eff_pct = col_i.slider("Roundtrip efficiency (%)", 50, 100, 90)
    soc_pct = col_j.slider("Initial SOC (%)", 0, 100, 50)

    if st.button("Run Stress Test", type="primary"):
        if start_date > end_date:
            st.error("Start date must be on or before end date.")
            return

        rare_path = run_config.rare_path if run_config is not None else None
        base_daily = _load_base_daily(
            region=region,
            start_date=start_date,
            end_date=end_date,
            base_dir=str(base_dir),
            rare_path=rare_path,
        )

        scenario_id = SCENARIO_LABEL_TO_ID[scenario_label]
        result = run_stress_test(
            base_daily_df=base_daily,
            scenario_id=scenario_id,
            supply_model_name="proxy",
            supply_config={
                "solar_capacity_mw": float(solar_capacity_mw),
                "wind_capacity_mw": float(wind_capacity_mw),
            },
            battery_config={
                "energy_mwh": float(energy_mwh),
                "power_mw": float(power_mw),
                "roundtrip_efficiency": float(eff_pct) / 100.0,
                "initial_soc_pct": float(soc_pct) / 100.0,
            },
            scenario_config={"wind_drought_days": int(drought_days)},
        )
        st.session_state["stress_result"] = result

    result = st.session_state.get("stress_result")
    if result is None:
        return

    warnings = result.metadata.get("warnings", [])
    for warning in warnings:
        st.warning(str(warning))

    c1, c2, c3 = st.columns(3)
    c1.metric("Deficit days", int(result.metrics.get("deficit_days", 0)))
    c2.metric("Unserved energy (MWh)", f"{float(result.metrics.get('total_unserved_mwh', 0.0)):,.0f}")
    c3.metric("Peak deficit (MW)", f"{float(result.metrics.get('peak_unserved_mw', 0.0)):,.0f}")

    render_stress_charts(result.timeseries)

    st.markdown("### Stress Findings")
    for bullet in result.findings:
        st.markdown(f"- {bullet}")

    st.download_button(
        "Download stress_timeseries.csv",
        data=result.timeseries.to_csv(index=False).encode("utf-8"),
        file_name="stress_timeseries.csv",
        mime="text/csv",
    )

    metrics_payload = {
        "scenario_id": result.scenario_id,
        "scenario_name": result.scenario_name,
        "metadata": result.metadata,
        "metrics": result.metrics,
        "findings": result.findings,
    }
    st.download_button(
        "Download stress_metrics.json",
        data=json.dumps(metrics_payload, indent=2, default=str).encode("utf-8"),
        file_name="stress_metrics.json",
        mime="application/json",
    )


def main() -> None:
    """Standalone Streamlit entrypoint for multipage stress test rendering."""
    st.set_page_config(page_title="Stress Test - RenewGrid", layout="wide")
    base_dir = Path(__file__).resolve().parents[4]
    load_environment(base_dir / ".env")
    render_stress_test(base_dir)


if __name__ == "__main__":
    main()
