"""Story chart component with scale modes for novice readability."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

SERIES_OPTIONS: dict[str, tuple[str, str]] = {
    "Demand (MW avg)": ("demand_mw_avg", "MW"),
    "Temperature (T2M)": ("weather_t2m", "C"),
    "Wind Speed (WS10M)": ("weather_ws10m", "m/s"),
    "Solar Proxy (ALLSKY)": ("weather_allsky_sfc_sw_dwn", "kWh/m^2/day"),
}


def _normalized(series: pd.Series) -> pd.Series:
    span = float(series.max() - series.min())
    if span <= 0:
        return pd.Series([50.0] * len(series), index=series.index)
    return ((series - series.min()) / span) * 100.0


def _dual_axis_chart(frame: pd.DataFrame, left_col: str, right_col: str) -> alt.Chart:
    left = (
        alt.Chart(frame[[left_col, "date"]].dropna())
        .mark_line(color="#1565C0")
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y(f"{left_col}:Q", title=left_col),
            tooltip=["date:T", alt.Tooltip(f"{left_col}:Q", format=".2f")],
        )
    )
    right = (
        alt.Chart(frame[[right_col, "date"]].dropna())
        .mark_line(color="#EF6C00")
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y(f"{right_col}:Q", title=right_col),
            tooltip=["date:T", alt.Tooltip(f"{right_col}:Q", format=".2f")],
        )
    )
    return alt.layer(left, right).resolve_scale(y="independent")


def render_story_chart(dataset: pd.DataFrame, key_prefix: str = "story_chart") -> None:
    """Render story chart with dual-axis, normalized, and single-series modes."""
    available = {
        label: spec
        for label, spec in SERIES_OPTIONS.items()
        if spec[0] in dataset.columns and dataset[spec[0]].notna().any()
    }
    if not available:
        st.info("No story-chart variables available in this window.")
        return

    default_selection = [
        label for label in ("Demand (MW avg)", "Temperature (T2M)") if label in available
    ]
    selected = st.multiselect(
        "Story chart variables",
        options=list(available.keys()),
        default=default_selection or list(available.keys())[:1],
        key=f"{key_prefix}_variables",
    )
    if not selected:
        st.info("Select at least one variable to plot.")
        return

    if len(selected) > 2:
        mode = "Normalized (0-100)"
        st.caption("Multiple series: normalized view enabled for readability.")
    elif len(selected) > 1:
        mode = st.radio(
            "Chart mode",
            options=["Dual axis", "Normalized (0-100)", "Single series"],
            index=0,
            horizontal=True,
            key=f"{key_prefix}_mode_multi",
        )
    else:
        mode = st.radio(
            "Chart mode",
            options=["Single series", "Normalized (0-100)"],
            index=0,
            horizontal=True,
            key=f"{key_prefix}_mode_single",
        )

    if mode == "Single series":
        label = st.selectbox(
            "Series",
            options=selected,
            key=f"{key_prefix}_single_series",
        )
        col, unit = available[label]
        series = dataset[["date", col]].dropna()
        if series.empty:
            st.warning("No data available for selected series in this window.")
            return
        st.caption(f"Units: {unit}")
        st.line_chart(series.set_index("date")[[col]])
        return

    if mode == "Dual axis":
        if len(selected) < 2:
            st.info("Dual axis requires at least two selected series.")
            return
        left_label = "Demand (MW avg)" if "Demand (MW avg)" in selected else selected[0]
        right_candidates = [s for s in selected if s != left_label]
        right_label = right_candidates[0]
        left_col, left_unit = available[left_label]
        right_col, right_unit = available[right_label]
        st.caption(f"Left axis: {left_label} ({left_unit}) | Right axis: {right_label} ({right_unit})")
        chart = _dual_axis_chart(dataset, left_col, right_col)
        st.altair_chart(chart, use_container_width=True)
        return

    long_frames: list[pd.DataFrame] = []
    for label in selected:
        col, unit = available[label]
        series = dataset[["date", col]].dropna()
        if series.empty:
            continue
        norm = _normalized(series[col])
        long_frames.append(
            pd.DataFrame(
                {
                    "date": series["date"],
                    "value": norm,
                    "series": f"{label} ({unit})",
                }
            )
        )
    if not long_frames:
        st.warning("No non-missing data available for normalized comparison.")
        return
    norm_frame = pd.concat(long_frames, ignore_index=True)
    st.caption("Normalized for comparison only.")
    chart = (
        alt.Chart(norm_frame)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title="Normalized scale (0-100)"),
            color=alt.Color("series:N", title="Series"),
            tooltip=["date:T", "series:N", alt.Tooltip("value:Q", format=".1f")],
        )
    )
    st.altair_chart(chart, use_container_width=True)

