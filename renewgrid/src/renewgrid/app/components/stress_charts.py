"""Plotly chart renderers for Phase 2 stress-test outputs."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from plotly import graph_objects as go


def _line_fig(df: pd.DataFrame, columns: list[str]) -> go.Figure:
    fig = go.Figure()
    for col in columns:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df["date"], y=df[col], mode="lines", name=col, cliponaxis=True))
    fig.update_layout(
        height=260,
        margin=dict(l=40, r=14, t=8, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
        showlegend=len(fig.data) > 1,
    )
    fig.update_xaxes(showgrid=False, automargin=True, nticks=7, tickformat="%b %d")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", automargin=True)
    return fig


def _bar_fig(df: pd.DataFrame, columns: list[str]) -> go.Figure:
    fig = go.Figure()
    for col in columns:
        if col in df.columns:
            fig.add_trace(go.Bar(x=df["date"], y=df[col], name=col))
    fig.update_layout(
        barmode="group",
        height=260,
        margin=dict(l=40, r=14, t=8, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
        showlegend=len(fig.data) > 1,
    )
    fig.update_xaxes(showgrid=False, automargin=True, nticks=7, tickformat="%b %d")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", automargin=True)
    return fig


def render_stress_charts(timeseries: pd.DataFrame) -> None:
    """Render Phase 2 stress charts."""
    col_a, col_b = st.columns(2, gap="small")
    with col_a:
        st.markdown("#### Demand vs Total Generation")
        st.plotly_chart(
            _line_fig(
                timeseries,
                ["demand_mw_stressed", "gen_total_mw"],
            ),
            use_container_width=True,
        )
    with col_b:
        st.markdown("#### Net Load and Battery Dispatch")
        st.plotly_chart(
            _line_fig(
                timeseries,
                ["net_load_mw", "battery_discharge_mw", "battery_charge_mw"],
            ),
            use_container_width=True,
        )

    col_c, col_d = st.columns(2, gap="small")
    with col_c:
        st.markdown("#### Battery State of Charge")
        st.plotly_chart(
            _line_fig(timeseries, ["soc_mwh"]),
            use_container_width=True,
        )
    with col_d:
        st.markdown("#### Unserved and Curtailment")
        st.plotly_chart(
            _bar_fig(timeseries, ["unserved_mw", "curtailment_mw"]),
            use_container_width=True,
        )
