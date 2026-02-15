"""Trust and transparency panel renderer."""

from __future__ import annotations

import streamlit as st

from renewgrid.app.state import RunConfig


def render_transparency_box(config: RunConfig, dataset_info: dict[str, str]) -> None:
    """Render source, unit, and reproducibility metadata in plain language."""
    with st.expander("Trust & Transparency", expanded=True):
        st.markdown("**Sources**: EIA Open Data, NASA POWER, optional RARE local parquet")
        st.markdown(
            f"**Date range**: {config.start_date.isoformat()} to {config.end_date.isoformat()}"
        )
        st.markdown(f"**Region**: {config.region}")
        st.markdown(f"**RARE loaded**: {dataset_info.get('rare_loaded', 'no')}")
        st.markdown(
            "**Units contract**: `demand_mw_avg` is daily average MW over UTC day. "
            "Derived daily energy is `demand_mwh = demand_mw_avg * 24`."
        )
        st.markdown(
            "**Reproducibility**: deterministic daily aggregation + cached app execution; "
            "no paywalls and no scraping in the core pipeline."
        )
        st.markdown(
            "**Data quality**: NASA POWER sentinels (`-999` and values `<= -900`) are "
            "converted to missing values; summary stats are computed on non-missing rows."
        )
