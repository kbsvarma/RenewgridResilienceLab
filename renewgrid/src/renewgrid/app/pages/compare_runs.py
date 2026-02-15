"""Compare runs tab for snapshot notebook workflow."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from renewgrid.app.snapshots import delete_snapshot, list_snapshots, load_snapshot


def render_compare_runs(base_dir: str) -> None:
    """Render side-by-side run comparison UI."""
    st.subheader("Compare Runs")
    run_ids = list_snapshots(base_dir)
    if len(run_ids) < 1:
        st.info("No snapshots yet. Save one from Guided Run first.")
        return

    run_a = st.selectbox("Run A", options=run_ids, index=0)
    run_b = st.selectbox("Run B", options=run_ids, index=min(1, len(run_ids) - 1))

    snap_a = load_snapshot(base_dir, run_a)
    snap_b = load_snapshot(base_dir, run_b)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"### Run A: {run_a}")
        st.json(snap_a["dataset_summary"])
    with c2:
        st.markdown(f"### Run B: {run_b}")
        st.json(snap_b["dataset_summary"])

    st.markdown("### Metrics Differences")
    a_summary = pd.DataFrame(snap_a["eval_results"].get("summary", []))
    b_summary = pd.DataFrame(snap_b["eval_results"].get("summary", []))
    if not a_summary.empty and not b_summary.empty:
        join_cols = ["model", "horizon"]
        merged = a_summary.merge(b_summary, on=join_cols, suffixes=("_a", "_b"))
        for metric in ["mae", "rmse", "mape", "skill_vs_persistence"]:
            merged[f"{metric}_diff"] = merged[f"{metric}_b"] - merged[f"{metric}_a"]
        st.dataframe(merged, use_container_width=True)
    else:
        st.info("Metrics summary unavailable for one or both snapshots.")

    st.markdown("### Overlay Plot")
    key_series_a = snap_a.get("key_series")
    key_series_b = snap_b.get("key_series")
    if isinstance(key_series_a, pd.DataFrame) and isinstance(key_series_b, pd.DataFrame):
        common = [c for c in key_series_a.columns if c in key_series_b.columns and c != "date"]
        if not common:
            st.info("No common key-series columns available to overlay.")
            return
        variable = st.selectbox(
            "Variable",
            options=common,
        )
        a_plot = key_series_a[["date", variable]].rename(columns={variable: f"A:{variable}"})
        b_plot = key_series_b[["date", variable]].rename(columns={variable: f"B:{variable}"})
        joined = a_plot.merge(b_plot, on="date", how="outer").sort_values("date").set_index("date")
        st.line_chart(joined)
    else:
        st.info("Overlay plot unavailable (key_series.parquet missing).")

    st.markdown("### Manage Snapshots")
    delete_target = st.selectbox("Delete snapshot", options=run_ids, key="delete_snapshot_target")
    if st.button("Delete selected snapshot"):
        if delete_snapshot(base_dir, delete_target):
            st.success(f"Deleted snapshot {delete_target}")
            st.rerun()
        else:
            st.warning(f"Could not delete snapshot {delete_target}")
