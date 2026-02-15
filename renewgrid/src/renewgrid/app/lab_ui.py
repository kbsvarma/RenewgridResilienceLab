"""Novice-friendly Resilience Lab dashboard for Phase 1 workflows."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from renewgrid.app.components.health import render_health_checklist
from renewgrid.app.components.monitor_map import render_monitor_map as render_monitor_map_component
from renewgrid.app.components.story_chart import render_story_chart
from renewgrid.app.components.transparency import render_transparency_box
from renewgrid.app.pages.compare_runs import render_compare_runs
from renewgrid.app.pages.data_explorer import render_data_explorer
from renewgrid.app.pages.forecast_lab import render_forecast_lab
from renewgrid.app.pages.monitor_map import render_monitor_map
from renewgrid.app.snapshots import save_snapshot
from renewgrid.app.state import RunConfig
from renewgrid.config import load_environment
from renewgrid.data.merge import build_daily_dataset
from renewgrid.features.build_features import build_feature_frame
from renewgrid.forecast.evaluate import rolling_origin_evaluate
from renewgrid.util.schema import validate_daily_frame
from renewgrid.util.units import assert_daily_mw_avg


def _resolve_dates(preset: str, custom_start: date, custom_end: date) -> tuple[date, date]:
    end_date = date.today() - timedelta(days=1)
    presets = {"30D": 30, "90D": 90, "180D": 180, "365D": 365}
    if preset in presets:
        start_date = end_date - timedelta(days=presets[preset] - 1)
        return start_date, end_date
    return custom_start, custom_end


@st.cache_data(ttl=600)
def _load_dataset(
    region: str,
    start_date: date,
    end_date: date,
    use_rare: bool,
    rare_path: str | None,
    base_dir: str,
) -> pd.DataFrame:
    """Build and cache deterministic daily dataset for UI use."""
    return build_daily_dataset(
        region=region,
        start_date=start_date,
        end_date=end_date,
        base_dir=base_dir,
        rare_path=rare_path if use_rare else None,
    )


@st.cache_data(ttl=600)
def _evaluate(
    frame: pd.DataFrame,
    target_col: str,
    feature_cols: tuple[str, ...],
    model_choice: str,
    horizon_days: int,
    max_splits: int,
    backtest_window_days: int,
    refit_every: int,
) -> dict[str, object]:
    """Run cached Phase 1 rolling-origin evaluation."""
    models = ("persistence",) if model_choice == "persistence" else (model_choice, "persistence")
    return rolling_origin_evaluate(
        frame=frame,
        target_col=target_col,
        feature_cols=list(feature_cols),
        horizons=tuple(range(1, horizon_days + 1)),
        min_train_size=max(20, min(60, len(frame) // 3)),
        max_splits=max_splits,
        backtest_window_days=backtest_window_days,
        refit_every=refit_every,
        models=models,
    )


def _render_answer_cards(dataset: pd.DataFrame, summary: pd.DataFrame | None) -> None:
    c1, c2, c3, c4 = st.columns(4)
    demand_series = (
        dataset["demand_mw_avg"].dropna() if "demand_mw_avg" in dataset.columns else pd.Series()
    )
    if not demand_series.empty:
        avg_demand = float(demand_series.mean())
        c1.metric("Avg Demand (MW)", f"{avg_demand:,.1f}")
        c1.caption("Average daily power demand for selected window.")
    else:
        c1.metric("Avg Demand (MW)", "N/A")
        c1.caption("No data available in this window.")

    t_col = "weather_t2m" if "weather_t2m" in dataset.columns else None
    w_col = "weather_ws10m" if "weather_ws10m" in dataset.columns else None
    if t_col:
        temp_series = dataset[t_col].dropna()
        temp_text = (
            f"{temp_series.mean():.1f} / {temp_series.max():.1f}" if not temp_series.empty else "N/A"
        )
    else:
        temp_text = "N/A"
    if w_col:
        wind_series = dataset[w_col].dropna()
        wind_text = (
            f"{wind_series.mean():.1f} / {wind_series.min():.1f}" if not wind_series.empty else "N/A"
        )
    else:
        wind_text = "N/A"

    c2.metric("Temp Avg / Max", temp_text)
    c2.caption(
        "Daily near-surface temperature summary."
        if temp_text != "N/A"
        else "No data available in this window."
    )
    c3.metric("Wind Avg / Min", wind_text)
    c3.caption("Daily wind speed summary." if wind_text != "N/A" else "No data available in this window.")

    if summary is not None and not summary.empty and "skill_vs_persistence" in summary.columns:
        model_rows = summary[summary["model"] != "persistence"]
        skill = float(model_rows["skill_vs_persistence"].mean()) if not model_rows.empty else 0.0
        c4.metric("Forecast Skill vs Persist", f"{skill:.3f}")
    else:
        c4.metric("Forecast Skill vs Persist", "N/A")
    c4.caption("Positive means better than persistence baseline.")

    with st.expander("Details"):
        st.write(
            "Cards summarize demand, weather, and model benchmark signal for quick interpretation."
        )


def _render_guided(base_dir: Path) -> None:
    st.subheader("Guided Run")
    st.caption("Recommended for first-time users: choose region, question, timeframe, then run.")

    region = st.radio(
        "Step 1: Choose region",
        options=["CAISO", "ERCOT"],
        help="CAISO: solar-heavy heat stress testbed. ERCOT: wind-heavy demand-growth testbed.",
        horizontal=True,
    )
    question = st.radio(
        "Step 2: Choose question",
        options=["How did demand & weather behave?", "How good are forecasts?"],
        horizontal=True,
    )
    preset = st.selectbox("Step 3: Choose timeframe", ["30D", "90D", "180D", "365D", "Custom"])
    col_a, col_b = st.columns(2)
    custom_start = col_a.date_input("Custom start", value=date.today() - timedelta(days=180))
    custom_end = col_b.date_input("Custom end", value=date.today() - timedelta(days=1))

    use_rare = st.checkbox("Use optional RARE file", value=False)
    rare_path = st.text_input("RARE path", value="") if use_rare else ""
    model_choice = st.selectbox("Model", ["persistence", "prophet", "xgboost"], index=0)
    horizon_days = st.slider("Horizon days", min_value=1, max_value=3, value=3)
    if model_choice in {"prophet", "xgboost"}:
        st.info(
            "Bounded backtest is enabled for speed: recent-window evaluation with capped splits "
            "and periodic refits."
        )
    with st.expander("Advanced evaluation controls", expanded=False):
        max_splits = st.slider("max_splits", min_value=10, max_value=50, value=20)
        backtest_window_days = st.slider(
            "backtest_window_days",
            min_value=30,
            max_value=365,
            value=90,
        )
        refit_every = st.slider("refit_every", min_value=1, max_value=14, value=7)

    start_date, end_date = _resolve_dates(preset, custom_start, custom_end)
    st.caption(
        f"Data freshness: latest available daily API window ending {end_date.isoformat()} "
        "(UTC day)."
    )
    st.caption(f"Selected analysis question: {question}")

    default_flags = {
        "dataset_loaded": False,
        "schema_valid": False,
        "units_valid": False,
        "eval_ran": False,
        "snapshot_saved": False,
    }
    default_messages = {k: "pending" for k in default_flags}
    if "health_flags" not in st.session_state:
        st.session_state["health_flags"] = default_flags.copy()
    if "health_messages" not in st.session_state:
        st.session_state["health_messages"] = default_messages.copy()

    flags = st.session_state["health_flags"]
    messages = st.session_state["health_messages"]

    if st.button("Run", type="primary"):
        config = RunConfig(
            region=region,
            start_date=start_date,
            end_date=end_date,
            timeframe_preset=preset,
            use_rare=use_rare,
            rare_path=rare_path or None,
            model_choice=model_choice,
            horizon_days=horizon_days,
        )

        dataset = _load_dataset(
            region=config.region,
            start_date=config.start_date,
            end_date=config.end_date,
            use_rare=config.use_rare,
            rare_path=config.rare_path,
            base_dir=str(base_dir),
        )
        flags["dataset_loaded"] = True
        messages["dataset_loaded"] = f"{len(dataset)} daily rows loaded"

        validate_daily_frame(dataset)
        flags["schema_valid"] = True
        messages["schema_valid"] = "Daily schema contract passed"

        assert_daily_mw_avg(dataset)
        flags["units_valid"] = True
        messages["units_valid"] = "Units contract checks passed"

        features = build_feature_frame(dataset, target_col="demand_mw_avg")
        feature_cols = tuple(
            c
            for c in features.columns
            if c not in {"date", "region", "source", "demand_mw_avg"}
            and pd.api.types.is_numeric_dtype(features[c])
        )
        evaluation = _evaluate(
            features,
            "demand_mw_avg",
            feature_cols,
            config.model_choice,
            config.horizon_days,
            max_splits=max_splits,
            backtest_window_days=backtest_window_days,
            refit_every=refit_every,
        )
        flags["eval_ran"] = True
        messages["eval_ran"] = "Rolling-origin evaluation complete"

        st.session_state["run_config"] = config
        st.session_state["dataset"] = dataset
        st.session_state["evaluation"] = evaluation
        st.session_state["region"] = config.region
        st.session_state["health_flags"] = flags
        st.session_state["health_messages"] = messages

    if "run_config" in st.session_state and "dataset" in st.session_state:
        config = st.session_state["run_config"]
        dataset = st.session_state["dataset"]
        evaluation = st.session_state.get("evaluation")

        summary = evaluation.get("summary") if isinstance(evaluation, dict) else None
        if isinstance(summary, pd.DataFrame):
            _render_answer_cards(dataset, summary)
        else:
            _render_answer_cards(dataset, None)

        story_sources = ["EIA Open Data", "NASA POWER"]
        if any(c in dataset.columns for c in ["solar_cf", "wind_cf", "solar_gen_mwh", "wind_gen_mwh"]):
            story_sources.append("RARE (optional)")
        render_story_chart(
            dataset,
            key_prefix="guided_story",
            region=config.region,
            start_date=config.start_date,
            end_date=config.end_date,
            sources=story_sources,
        )
        with st.expander("Map preview", expanded=False):
            render_monitor_map_component(
                base_dir=base_dir,
                dataset_by_region={config.region: dataset},
                key_prefix="guided_preview",
            )

        dataset_info = {
            "rare_loaded": (
                "yes"
                if any(
                    c in dataset.columns
                    for c in ["solar_cf", "wind_cf", "solar_gen_mwh", "wind_gen_mwh"]
                )
                else "no"
            )
        }
        render_transparency_box(config, dataset_info)

        if st.button("Save snapshot"):
            summary_rows = len(dataset)
            summary_dates = {
                "rows": summary_rows,
                "start_date": str(dataset["date"].min()),
                "end_date": str(dataset["date"].max()),
                "region": config.region,
            }
            key_cols = [
                c
                for c in [
                    "date",
                    "demand_mw_avg",
                    "weather_t2m",
                    "weather_ws10m",
                    "weather_allsky_sfc_sw_dwn",
                    "solar_cf",
                    "wind_cf",
                ]
                if c in dataset.columns
            ]
            key_series = dataset[key_cols].copy() if key_cols else None
            run_id = save_snapshot(
                run_config=config,
                dataset_summary=summary_dates,
                eval_results=evaluation if isinstance(evaluation, dict) else {},
                base_dir=base_dir,
                key_series=key_series,
            )
            flags["snapshot_saved"] = True
            messages["snapshot_saved"] = f"Snapshot written: {run_id}"
            st.success(f"Snapshot saved under reports/runs/{run_id}")

        st.markdown("### Downloads")
        st.download_button(
            "Download dataset CSV",
            data=dataset.to_csv(index=False).encode("utf-8"),
            file_name=f"{config.region}_daily_dataset.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download run_config.json",
            data=json.dumps(config.to_dict(), indent=2).encode("utf-8"),
            file_name="run_config.json",
            mime="application/json",
        )
        if isinstance(evaluation, dict):
            payload = {
                "availability": evaluation.get("availability", {}),
                "summary": evaluation.get("summary", pd.DataFrame()).to_dict(orient="records")
                if isinstance(evaluation.get("summary"), pd.DataFrame)
                else [],
            }
            st.download_button(
                "Download evaluation_metrics.json",
                data=json.dumps(payload, indent=2).encode("utf-8"),
                file_name="evaluation_metrics.json",
                mime="application/json",
            )

    st.session_state["health_flags"] = flags
    st.session_state["health_messages"] = messages
    render_health_checklist(flags, messages)


def main() -> None:
    """Streamlit entrypoint for Resilience Lab dashboard."""
    st.set_page_config(page_title="RenewGrid Resilience Lab", layout="wide")
    st.title("RenewGrid Resilience Lab")
    st.caption("Phase 1 only: public-data daily monitoring and forecast evaluation")

    base_dir = Path(__file__).resolve().parents[3]
    load_environment(base_dir / ".env")

    mode = st.toggle("Guided Run (recommended)", value=True)

    with st.sidebar:
        st.markdown("### Trust & Transparency")
        config = st.session_state.get("run_config")
        dataset = st.session_state.get("dataset")
        if config is not None and isinstance(dataset, pd.DataFrame):
            dataset_info = {
                "rare_loaded": (
                    "yes"
                    if any(
                        c in dataset.columns
                        for c in ["solar_cf", "wind_cf", "solar_gen_mwh", "wind_gen_mwh"]
                    )
                    else "no"
                )
            }
            render_transparency_box(config, dataset_info)
        else:
            st.caption("Run Guided flow to populate provenance and unit details.")
        render_health_checklist(
            st.session_state.get(
                "health_flags",
                {
                    "dataset_loaded": False,
                    "schema_valid": False,
                    "units_valid": False,
                    "eval_ran": False,
                    "snapshot_saved": False,
                },
            ),
            st.session_state.get(
                "health_messages",
                {
                    "dataset_loaded": "pending",
                    "schema_valid": "pending",
                    "units_valid": "pending",
                    "eval_ran": "pending",
                    "snapshot_saved": "pending",
                },
            ),
        )

    if mode:
        _render_guided(base_dir)
    else:
        tabs = st.tabs([
            "Guided Run",
            "Monitor Map",
            "Data Explorer",
            "Forecast Lab",
            "Compare Runs",
            "Scenarios (Phase 2 preview)",
        ])
        with tabs[0]:
            _render_guided(base_dir)
        with tabs[1]:
            render_monitor_map(base_dir)
        with tabs[2]:
            render_data_explorer(st.session_state.get("dataset"))
        with tabs[3]:
            render_forecast_lab(st.session_state.get("dataset"))
        with tabs[4]:
            render_compare_runs(str(base_dir))
        with tabs[5]:
            st.warning("Phase 2 scenarios are preview-only in Phase 1 UI.")


if __name__ == "__main__":
    main()
