"""Streamlit app entrypoint."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


def main() -> None:
    """Render a tiny dashboard for hello-pipeline outputs."""
    st.title("RenewGrid Resilience Lab")
    st.write("Phase 1 hello pipeline outputs")

    base_dir = Path(__file__).resolve().parents[3]
    nasa_path = base_dir / "data" / "processed" / "nasa_power_daily.parquet"
    eia_path = base_dir / "data" / "processed" / "eia_rto_daily.parquet"

    if nasa_path.exists():
        st.subheader("NASA POWER")
        st.dataframe(pd.read_parquet(nasa_path))
    if eia_path.exists():
        st.subheader("EIA")
        st.dataframe(pd.read_parquet(eia_path))


if __name__ == "__main__":
    main()
