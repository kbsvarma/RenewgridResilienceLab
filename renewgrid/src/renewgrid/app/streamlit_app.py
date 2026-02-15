"""Streamlit app entrypoint."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


def main() -> None:
    """Render a tiny dashboard for hello-pipeline outputs."""
    st.title("RenewGrid Resilience Lab")
    st.write("Daily processed outputs")

    base_dir = Path(__file__).resolve().parents[3]
    nasa_path = base_dir / "data" / "processed" / "nasa_power_daily.parquet"
    eia_path = base_dir / "data" / "processed" / "eia_rto_daily.parquet"

    if nasa_path.exists():
        st.subheader("NASA POWER")
        st.dataframe(pd.read_parquet(nasa_path))
    if eia_path.exists():
        st.subheader("EIA Daily")
        st.dataframe(pd.read_parquet(eia_path))

    phase1_files = sorted(
        list((base_dir / "data" / "processed").glob("CAISO_daily_*.parquet"))
        + list((base_dir / "data" / "processed").glob("ERCOT_daily_*.parquet"))
    )
    if phase1_files:
        st.subheader("Phase 1 Daily Datasets")
        selected = st.selectbox(
            "Choose a dataset",
            options=phase1_files,
            format_func=lambda p: p.name,
        )
        st.dataframe(pd.read_parquet(selected))


if __name__ == "__main__":
    main()
