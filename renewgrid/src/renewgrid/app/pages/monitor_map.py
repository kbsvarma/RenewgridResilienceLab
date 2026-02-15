"""Monitor map page for latest available Phase 1 regional context."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

OVERLAY_TO_COLUMN = {
    "Demand level": ("demand_mw_avg", "MW"),
    "Temperature": ("weather_t2m", "C"),
    "Wind speed": ("weather_ws10m", "m/s"),
    "Solar proxy": ("weather_allsky_sfc_sw_dwn", "kWh/m^2/day"),
}


@st.cache_data(ttl=600)
def _load_latest_region_data(base_dir: str) -> dict[str, pd.DataFrame]:
    """Load latest processed CAISO/ERCOT daily datasets with cache TTL."""
    processed = Path(base_dir) / "data" / "processed"
    out: dict[str, pd.DataFrame] = {}
    for region in ("CAISO", "ERCOT"):
        files = sorted(processed.glob(f"{region}_daily_*.parquet"))
        if files:
            out[region] = pd.read_parquet(files[-1]).sort_values("date").reset_index(drop=True)
    return out


def _load_region_geojson(base_dir: str) -> dict[str, object]:
    path = Path(base_dir) / "src" / "renewgrid" / "app" / "assets" / "regions.geojson"
    return json.loads(path.read_text(encoding="utf-8"))


def _status_text(value: float, series: pd.Series) -> str:
    mean = float(series.mean())
    std = float(series.std()) if float(series.std()) > 0 else 1.0
    z = (value - mean) / std
    if z > 1.0:
        return "HIGH compared to this period"
    if z < -1.0:
        return "LOW compared to this period"
    return "NORMAL compared to this period"


def render_monitor_map(base_dir: Path) -> None:
    """Render map with region overlays and plain-English tooltips."""
    st.subheader("Live Monitor Map")
    st.caption("Latest available daily window (cached 10 minutes)")

    overlay = st.selectbox("Overlay mode", list(OVERLAY_TO_COLUMN.keys()))
    col_name, unit = OVERLAY_TO_COLUMN[overlay]
    data = _load_latest_region_data(str(base_dir))
    if not data:
        st.warning("No Phase 1 processed datasets found. Run `make phase1` first.")
        return

    features = _load_region_geojson(str(base_dir))["features"]
    regions_df = []
    for feature in features:
        region = feature["properties"]["region"]
        if region not in data or col_name not in data[region].columns:
            continue
        region_frame = data[region]
        latest = float(region_frame[col_name].iloc[-1])
        status = _status_text(latest, region_frame[col_name])
        regions_df.append(
            {
                "region": region,
                "lat": feature["properties"]["lat"],
                "lon": feature["properties"]["lon"],
                "value": latest,
                "unit": unit,
                "status": status,
                "tooltip": f"{region}: {overlay} is {status} ({latest:,.2f} {unit})",
                "geometry": feature["geometry"],
            }
        )

    if not regions_df:
        st.warning("Selected overlay is unavailable in latest datasets.")
        return

    marker_df = pd.DataFrame(regions_df)

    polygon_layer = pdk.Layer(
        "GeoJsonLayer",
        data={"type": "FeatureCollection", "features": [
            {
                "type": "Feature",
                "properties": {k: v for k, v in row.items() if k != "geometry"},
                "geometry": row["geometry"],
            }
            for row in regions_df
        ]},
        pickable=True,
        stroked=True,
        filled=True,
        get_fill_color="[90, 155, 212, 60]",
        get_line_color="[20, 50, 120, 180]",
        line_width_min_pixels=2,
    )

    marker_layer = pdk.Layer(
        "ScatterplotLayer",
        data=marker_df,
        get_position="[lon, lat]",
        get_radius=90000,
        get_fill_color="[239, 99, 81, 200]",
        pickable=True,
    )

    view = pdk.ViewState(latitude=34.7, longitude=-109.5, zoom=3.7)
    deck = pdk.Deck(
        layers=[polygon_layer, marker_layer],
        initial_view_state=view,
        tooltip={"text": "{tooltip}"},
    )
    st.pydeck_chart(deck, use_container_width=True)

    right_col, _ = st.columns([2, 1])
    with right_col:
        selected_region = st.selectbox("Detail panel region", marker_df["region"].tolist())
        frame = data[selected_region]
        st.markdown(f"### {selected_region} details")
        st.caption("Plain-English interpretation for selected overlay")
        current = float(frame[col_name].iloc[-1])
        status = _status_text(current, frame[col_name])
        st.write(f"{overlay} is **{status}** at {current:,.2f} {unit}.")
        window = st.selectbox("Detail window", [30, 90], index=0)
        st.line_chart(frame.tail(window).set_index("date")[[col_name]])
