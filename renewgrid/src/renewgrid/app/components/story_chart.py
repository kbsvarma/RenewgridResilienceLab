"""Story chart component with scale modes and plain-language summaries."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from renewgrid.app.components.story_findings import generate_findings

SERIES_OPTIONS: dict[str, tuple[str, str]] = {
    "Demand (MW avg)": ("demand_mw_avg", "MW"),
    "Temperature (T2M)": ("weather_t2m", "°C"),
    "Wind Speed (WS10M)": ("weather_ws10m", "m/s"),
    "Solar Proxy (ALLSKY)": ("weather_allsky_sfc_sw_dwn", "kWh/m^2/day"),
}


def _normalized(series: pd.Series) -> pd.Series:
    span = float(series.max() - series.min())
    if span <= 0:
        return pd.Series([50.0] * len(series), index=series.index)
    return ((series - series.min()) / span) * 100.0


def compute_axis_range(series: pd.Series, include_zero: bool) -> tuple[float, float]:
    """Compute a stable y-axis range using controlled padding."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return (0.0, 1.0)
    data_min = float(clean.min())
    data_max = float(clean.max())
    data_range = data_max - data_min
    pad = max(data_range * 0.03, data_range * 0.02)
    if data_range <= 0:
        pad = 1.0
    if include_zero:
        return (0.0, data_max + pad)
    return (data_min - pad, data_max + pad)


def _series_units(selected_series: list[str]) -> str:
    pairs = [f"{label}: {SERIES_OPTIONS[label][1]}" for label in selected_series if label in SERIES_OPTIONS]
    return ", ".join(pairs)


def generate_chart_summary(
    region: str,
    start_date: object,
    end_date: object,
    resolution: str,
    selected_series: list[str],
    sources: list[str],
    chart_mode: str,
) -> str:
    """Generate a short deterministic plain-English chart summary."""
    series_text = ", ".join(selected_series)
    units_text = _series_units(selected_series)
    return (
        f"This chart shows {series_text} for {region} from {start_date} to {end_date}. "
        f"Values are {resolution}. "
        f"Data sources are {', '.join(sources)}, and units are {units_text} (mode: {chart_mode}). "
        "This helps show how weather shifts can influence demand patterns and stress-relevant days."
    )


def _analytical_sentence(
    dataset: pd.DataFrame,
    first_label: str,
    second_label: str,
) -> str:
    if first_label not in SERIES_OPTIONS or second_label not in SERIES_OPTIONS:
        return "Correlation not available (insufficient data)."
    first_col = SERIES_OPTIONS[first_label][0]
    second_col = SERIES_OPTIONS[second_label][0]
    pair = dataset[[first_col, second_col]].dropna()
    if len(pair) < 3:
        return "Correlation not available (insufficient data)."
    corr = float(pair[first_col].corr(pair[second_col]))
    if pd.isna(corr):
        return "Correlation not available (insufficient data)."
    if corr > 0.2:
        direction = "higher"
    elif corr < -0.2:
        direction = "lower"
    else:
        direction = "little change in"
    return (
        f"During this window, changes in {second_label} tended to align with {direction} "
        f"{first_label} (correlation {corr:.2f})."
    )


def _render_plotly_dual_axis(
    dataset: pd.DataFrame,
    left_label: str,
    right_label: str,
    include_zero: bool,
) -> None:
    left_col, left_unit = SERIES_OPTIONS[left_label]
    right_col, right_unit = SERIES_OPTIONS[right_label]
    left_axis_title = f"{left_label} ({left_unit})"
    right_axis_title = f"{right_label} ({right_unit})"
    left_data = dataset[["date", left_col]].dropna()
    right_data = dataset[["date", right_col]].dropna()
    if left_data.empty or right_data.empty:
        st.warning("No data available for the selected dual-axis series in this window.")
        return

    left_range = compute_axis_range(left_data[left_col], include_zero=include_zero)
    right_range = compute_axis_range(right_data[right_col], include_zero=include_zero)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=left_data["date"],
            y=left_data[left_col],
            mode="lines",
            name=left_axis_title,
            line={"color": "#1565C0"},
            cliponaxis=True,
        )
        ,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=right_data["date"],
            y=right_data[right_col],
            mode="lines",
            name=right_axis_title,
            line={"color": "#EF6C00"},
            cliponaxis=True,
        )
        ,
        secondary_y=True,
    )
    fig.update_yaxes(range=list(left_range), secondary_y=False, title_text=left_axis_title)
    fig.update_yaxes(range=list(right_range), secondary_y=True, title_text=right_axis_title)
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.06)",
        gridwidth=1,
        zeroline=False,
        nticks=6,
        secondary_y=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.06)",
        gridwidth=1,
        zeroline=False,
        nticks=6,
        secondary_y=True,
    )
    fig.update_xaxes(showgrid=False, title_text="Date")
    fig.update_traces(line=dict(width=2))
    fig.update_layout(
        yaxis_autorange=False,
        yaxis2_autorange=False,
        height=520,
        margin=dict(l=70, r=70, t=40, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def render_story_chart(
    dataset: pd.DataFrame,
    key_prefix: str = "story_chart",
    region: str | None = None,
    start_date: object | None = None,
    end_date: object | None = None,
    sources: list[str] | None = None,
) -> None:
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

    summary_mode = st.radio(
        "Summary mode",
        options=["Basic", "Analytical"],
        index=0,
        horizontal=True,
        key=f"{key_prefix}_summary_mode",
    )

    if len(selected) > 2:
        mode = "Normalized (0-100)"
        st.caption("Multiple series: normalized view enabled for readability.")
        st.caption("For 3+ series, Normalized view is usually clearer.")
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

    if mode == "Dual axis":
        scale = st.radio(
            "Scale",
            options=["Fit to data (recommended)", "Include zero baseline"],
            index=0,
            horizontal=True,
            key=f"{key_prefix}_scale_mode",
        )
        include_zero = scale == "Include zero baseline"
    else:
        include_zero = False

    chart_summary = generate_chart_summary(
        region=region or "Selected region",
        start_date=start_date or pd.to_datetime(dataset["date"]).min().date(),
        end_date=end_date or pd.to_datetime(dataset["date"]).max().date(),
        resolution="daily averages (UTC day)",
        selected_series=selected,
        sources=sources or ["EIA Open Data", "NASA POWER", "RARE (optional)"],
        chart_mode=mode,
    )
    if summary_mode == "Analytical" and len(selected) >= 2:
        chart_summary = f"{chart_summary} {_analytical_sentence(dataset, selected[0], selected[1])}"
    elif summary_mode == "Analytical":
        chart_summary = f"{chart_summary} Correlation not available (insufficient data)."
    st.info(chart_summary)

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
    elif mode == "Dual axis":
        if len(selected) < 2:
            st.info("Dual axis requires at least two selected series.")
            return
        left_label = "Demand (MW avg)" if "Demand (MW avg)" in selected else selected[0]
        right_candidates = [s for s in selected if s != left_label]
        right_label = right_candidates[0]
        left_axis = f"{left_label} ({available[left_label][1]})"
        right_axis = f"{right_label} ({available[right_label][1]})"
        st.caption(f"Left axis: {left_axis} | Right axis: {right_axis}")
        _render_plotly_dual_axis(dataset, left_label, right_label, include_zero=include_zero)
    else:
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

    findings = generate_findings(
        df=dataset,
        selected_series=selected,
        region=region or "Selected region",
        start_date=start_date or pd.to_datetime(dataset["date"]).min().date(),
        end_date=end_date or pd.to_datetime(dataset["date"]).max().date(),
    )
    st.subheader("What this window suggests")
    for bullet in findings:
        st.markdown(f"- {bullet}")
    st.caption("These bullets summarize patterns in the selected window (descriptive only, not a forecast).")
