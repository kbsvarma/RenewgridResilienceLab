"""Data Explorer tab for integrity-first inspection."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from renewgrid.util.schema import validate_daily_frame
from renewgrid.util.units import assert_daily_mw_avg


def render_data_explorer(dataset: pd.DataFrame | None) -> None:
    """Render schema checks, missingness, stats, and interactive table."""
    st.subheader("Data Explorer")
    if dataset is None or dataset.empty:
        st.info("Run Guided flow first to load a dataset.")
        return

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Run validation now"):
            try:
                validate_daily_frame(dataset)
                assert_daily_mw_avg(dataset)
                st.success("Schema + units validation passed.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Validation failed: {exc}")

    with col_b:
        st.caption("Contract: required columns are date + demand_mw_avg")

    st.markdown("### Missingness")
    missing = dataset.isna().mean().mul(100).round(2).rename("missing_pct")
    missing_df = missing.rename_axis("column").reset_index(name="missing_pct")
    st.dataframe(missing_df, use_container_width=True)

    st.markdown("### Summary Stats")
    st.dataframe(dataset.describe(include="all").T, use_container_width=True)

    st.markdown("### Searchable Dataframe")
    search_col = st.selectbox("Filter column", options=list(dataset.columns))
    query = st.text_input("Contains")
    if query:
        filtered = dataset[
            dataset[search_col].astype(str).str.contains(query, case=False, na=False)
        ]
    else:
        filtered = dataset
    st.dataframe(filtered, use_container_width=True)
