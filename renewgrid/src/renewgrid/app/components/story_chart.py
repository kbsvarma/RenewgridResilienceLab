"""Story chart component with clean Plotly rendering for Phase 1."""

from __future__ import annotations

import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from renewgrid.app.components.story_findings import generate_findings

SERIES_OPTIONS: dict[str, tuple[str, str]] = {
    "Demand (MW avg)": ("demand_mw_avg", "MW"),
    "Temperature (T2M)": ("weather_t2m", "°C"),
    "Wind Speed (WS10M)": ("weather_ws10m", "m/s"),
    "Solar Proxy (ALLSKY)": ("weather_allsky_sfc_sw_dwn", "kWh/m^2/day"),
}


def padded_range(series: pd.Series, pad_ratio: float = 0.08) -> tuple[float, float]:
    """Return min/max range padded around data to avoid edge clipping."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return (0.0, 1.0)
    min_v = float(clean.min())
    max_v = float(clean.max())
    span = max_v - min_v
    pad = span * pad_ratio if span > 0 else 1.0
    return (min_v - pad, max_v + pad)


def compute_axis_range(series: pd.Series, include_zero: bool) -> tuple[float, float]:
    """Backward-compatible axis helper with optional zero baseline inclusion."""
    lower, upper = padded_range(series)
    if include_zero:
        return (0.0, upper)
    return (lower, upper)


def _normalized(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    min_v = clean.min()
    max_v = clean.max()
    span = float(max_v - min_v)
    if pd.isna(span) or span <= 0:
        return pd.Series([50.0] * len(series), index=series.index, dtype=float)
    return ((clean - min_v) / span) * 100.0


def _series_units(selected_series: list[str]) -> str:
    return ", ".join(
        f"{label}: {SERIES_OPTIONS[label][1]}" for label in selected_series if label in SERIES_OPTIONS
    )


def generate_chart_summary(
    region: str,
    start_date: object,
    end_date: object,
    selected_series: list[str],
    sources: list[str],
) -> str:
    """Generate a short novice-friendly chart description."""
    labels = ", ".join(selected_series)
    units = _series_units(selected_series)
    return (
        f"What am I looking at? Daily (UTC-day) trends for {region} from {start_date} to {end_date}: "
        f"{labels}. Units: {units}. Sources: {', '.join(sources)}."
    )


def _prepare_daily_series(dataset: pd.DataFrame, col: str) -> pd.DataFrame:
    data = (
        dataset[["date", col]]
        .dropna()
        .assign(date=lambda d: pd.to_datetime(d["date"]))
        .groupby("date", as_index=False)[col]
        .mean()
        .sort_values("date")
    )
    return data


def _x_tick_format(dates: pd.Series) -> str:
    years = pd.to_datetime(dates).dt.year.nunique()
    return "%b %d\n%Y" if years > 1 else "%b %d"


def _base_layout(fig: go.Figure, dates: pd.Series) -> None:
    nticks = max(6, min(10, len(dates) // 7 if len(dates) > 20 else len(dates)))
    fig.update_xaxes(
        showgrid=False,
        showline=False,
        mirror=False,
        automargin=True,
        nticks=nticks,
        tickformat=_x_tick_format(dates),
        title_text="Date",
    )
    fig.update_layout(
        height=470,
        margin=dict(l=70, r=70, t=35, b=65),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_traces(line=dict(width=2))


def _render_plotly_dual_axis(
    dataset: pd.DataFrame,
    left_label: str,
    right_label: str,
    include_zero: bool,
    st_module: object,
) -> None:
    left_col, left_unit = SERIES_OPTIONS[left_label]
    right_col, right_unit = SERIES_OPTIONS[right_label]
    left_axis_title = f"{left_label} ({left_unit})"
    right_axis_title = f"{right_label} ({right_unit})"
    left_data = _prepare_daily_series(dataset, left_col)
    right_data = _prepare_daily_series(dataset, right_col)
    if left_data.empty or right_data.empty:
        st_module.warning("No data available for selected series in this window.")
        return

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=left_data["date"],
            y=left_data[left_col],
            mode="lines",
            name=left_axis_title,
            line={"color": "#1565C0"},
            cliponaxis=True,
        ),
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
        ),
        secondary_y=True,
    )

    left_range = padded_range(left_data[left_col])
    right_range = padded_range(right_data[right_col])
    if include_zero:
        left_range = (min(0.0, left_range[0]), left_range[1])
        right_range = (min(0.0, right_range[0]), right_range[1])

    fig.update_yaxes(
        title_text=left_axis_title,
        range=list(left_range),
        rangemode="tozero" if include_zero else "normal",
        zeroline=False,
        showgrid=True,
        gridcolor="rgba(255,255,255,0.05)",
        gridwidth=1,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text=right_axis_title,
        range=list(right_range),
        rangemode="tozero" if include_zero else "normal",
        zeroline=False,
        showgrid=False,
        secondary_y=True,
    )
    _base_layout(fig, left_data["date"])
    st_module.plotly_chart(fig, use_container_width=True)


def _render_plotly_single(dataset: pd.DataFrame, label: str, st_module: object) -> None:
    col, unit = SERIES_OPTIONS[label]
    data = _prepare_daily_series(dataset, col)
    if data.empty:
        st_module.warning("No data available for selected series in this window.")
        return
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["date"],
            y=data[col],
            mode="lines",
            name=f"{label} ({unit})",
            line={"color": "#1565C0"},
            cliponaxis=True,
        )
    )
    fig.update_yaxes(
        title_text=f"{label} ({unit})",
        range=list(padded_range(data[col])),
        rangemode="normal",
        zeroline=False,
        showgrid=True,
        gridcolor="rgba(255,255,255,0.05)",
        gridwidth=1,
    )
    _base_layout(fig, data["date"])
    st_module.plotly_chart(fig, use_container_width=True)


def _render_plotly_normalized(dataset: pd.DataFrame, labels: list[str], st_module: object) -> None:
    fig = go.Figure()
    dates_ref: pd.Series | None = None
    for label in labels:
        col, unit = SERIES_OPTIONS[label]
        data = _prepare_daily_series(dataset, col)
        if data.empty:
            continue
        dates_ref = data["date"]
        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=_normalized(data[col]),
                mode="lines",
                name=f"{label} ({unit})",
                cliponaxis=True,
            )
        )
    if not fig.data:
        st_module.warning("No non-missing data available for normalized comparison.")
        return
    fig.update_yaxes(
        title_text="Normalized scale (0-100)",
        range=[0, 100],
        rangemode="normal",
        zeroline=False,
        showgrid=True,
        gridcolor="rgba(255,255,255,0.05)",
        gridwidth=1,
    )
    _base_layout(fig, dates_ref if dates_ref is not None else pd.Series(dtype="datetime64[ns]"))
    st_module.caption("Normalized for comparison only.")
    st_module.plotly_chart(fig, use_container_width=True)


def render_story_chart(
    dataset: pd.DataFrame,
    key_prefix: str = "story_chart",
    region: str | None = None,
    start_date: object | None = None,
    end_date: object | None = None,
    sources: list[str] | None = None,
) -> None:
    """Render story chart with dual-axis, normalized, and single-series modes."""
    import streamlit as st

    available = {
        label: spec
        for label, spec in SERIES_OPTIONS.items()
        if spec[0] in dataset.columns and dataset[spec[0]].notna().any()
    }
    if not available:
        st.info("No story-chart variables available in this window.")
        return

    selected = st.multiselect(
        "Story chart variables",
        options=list(available.keys()),
        default=[l for l in ("Demand (MW avg)", "Temperature (T2M)") if l in available]
        or [list(available.keys())[0]],
        key=f"{key_prefix}_variables",
    )
    if not selected:
        st.info("Select at least one variable to plot.")
        return

    st.caption(
        generate_chart_summary(
            region=region or "Selected region",
            start_date=start_date or pd.to_datetime(dataset["date"]).min().date(),
            end_date=end_date or pd.to_datetime(dataset["date"]).max().date(),
            selected_series=selected,
            sources=sources or ["EIA Open Data", "NASA POWER", "RARE (optional)"],
        )
    )

    if len(selected) > 2:
        mode = "Normalized (0-100)"
        st.caption("For 3+ series, normalized mode is usually clearer.")
    elif len(selected) > 1:
        mode = st.radio(
            "Chart mode",
            options=["Dual axis", "Normalized (0-100)", "Single series"],
            index=0,
            horizontal=True,
            key=f"{key_prefix}_mode_multi",
        )
    else:
        mode = "Single series"

    include_zero = False
    if mode == "Dual axis":
        include_zero = (
            st.radio(
                "Scale",
                options=["Fit to data (recommended)", "Include zero baseline"],
                index=0,
                horizontal=True,
                key=f"{key_prefix}_scale_mode",
            )
            == "Include zero baseline"
        )

    if mode == "Single series":
        label = st.selectbox("Series", options=selected, key=f"{key_prefix}_single_series")
        _render_plotly_single(dataset, label, st)
    elif mode == "Dual axis":
        left_label = "Demand (MW avg)" if "Demand (MW avg)" in selected else selected[0]
        right_label = next(s for s in selected if s != left_label)
        st.caption(
            "Left axis: "
            f"{left_label} ({available[left_label][1]}) | Right axis: "
            f"{right_label} ({available[right_label][1]})"
        )
        _render_plotly_dual_axis(
            dataset,
            left_label,
            right_label,
            include_zero=include_zero,
            st_module=st,
        )
    else:
        _render_plotly_normalized(dataset, selected, st)

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
    st.caption("Descriptive summary of the selected window (not a forecast).")
