"""Monitor map component for Phase 1 region overlays."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import pydeck as pdk
except ImportError:  # pragma: no cover - import-time environment fallback
    pdk = None

OVERLAY_TO_COLUMN: dict[str, tuple[str, str]] = {
    "Demand": ("demand_mw_avg", "MW"),
    "Temperature": ("weather_t2m", "C"),
    "Wind": ("weather_ws10m", "m/s"),
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


def _qualitative_label(value: float, series: pd.Series) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return "Normal"
    p33 = float(non_null.quantile(0.33))
    p67 = float(non_null.quantile(0.67))
    if value < p33:
        return "Low"
    if value > p67:
        return "High"
    return "Normal"


def render_monitor_map(
    base_dir: Path,
    dataset_by_region: dict[str, pd.DataFrame] | None = None,
    key_prefix: str = "monitor_map",
) -> None:
    """Render map with region overlays and plain-English tooltips."""
    st.subheader("Monitor Map")
    st.caption("Latest available daily window (cached 10 minutes)")
    if pdk is None:
        st.error(
            "Monitor map requires pydeck. Install dependencies with "
            "`uv sync --extra dev` or `pip install -e \".[dev]\"`."
        )
        return

    overlay = st.selectbox(
        "Overlay mode",
        list(OVERLAY_TO_COLUMN.keys()),
        key=f"{key_prefix}_overlay",
    )
    col_name, unit = OVERLAY_TO_COLUMN[overlay]

    data = _load_latest_region_data(str(base_dir))
    for region, frame in (dataset_by_region or {}).items():
        if not frame.empty:
            data[region] = frame.sort_values("date").reset_index(drop=True)

    if not data:
        st.warning("No Phase 1 processed datasets found. Run `make phase1` first.")
        return

    features = _load_region_geojson(str(base_dir)).get("features", [])
    regions_df: list[dict[str, object]] = []
    for feature in features:
        props = feature.get("properties", {})
        region = str(props.get("name", props.get("region", ""))).upper()
        if region not in data or col_name not in data[region].columns:
            continue
        region_frame = data[region]
        series = region_frame[col_name].dropna()
        if series.empty:
            continue
        overlay_value = float(series.mean())
        label = _qualitative_label(overlay_value, series)
        regions_df.append(
            {
                "region": region,
                "name": props.get("name", region),
                "lat": float(props.get("lat", 0.0)),
                "lon": float(props.get("lon", 0.0)),
                "value": overlay_value,
                "unit": unit,
                "label": label,
                "tooltip": (
                    f"{region}\n{overlay}: {overlay_value:,.2f} {unit}\n"
                    f"Status: {label} for this selected window"
                ),
                "geometry": feature.get("geometry"),
            }
        )

    if not regions_df:
        st.warning("Selected overlay is unavailable in current datasets.")
        return

    marker_df = pd.DataFrame(regions_df)
    if "region" in st.session_state:
        default_region = st.session_state["region"]
    else:
        default_region = marker_df["region"].iloc[0]
    options = marker_df["region"].tolist()
    default_idx = options.index(default_region) if default_region in options else 0
    selected_region = st.selectbox(
        "Select region",
        options=options,
        index=default_idx,
        key=f"{key_prefix}_region",
    )
    st.session_state["region"] = selected_region

    polygon_layer = pdk.Layer(
        "GeoJsonLayer",
        data={
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {k: v for k, v in row.items() if k != "geometry"},
                    "geometry": row["geometry"],
                }
                for row in regions_df
            ],
        },
        pickable=True,
        stroked=True,
        filled=True,
        get_fill_color="[74, 144, 226, 70]",
        get_line_color="[21, 54, 120, 180]",
        line_width_min_pixels=2,
    )
    marker_layer = pdk.Layer(
        "ScatterplotLayer",
        data=marker_df,
        get_position="[lon, lat]",
        get_radius=85000,
        get_fill_color="[236, 112, 99, 200]",
        pickable=True,
    )
    text_layer = pdk.Layer(
        "TextLayer",
        data=marker_df,
        get_position="[lon, lat]",
        get_text="region",
        get_size=16,
        get_color=[25, 25, 25, 255],
        get_alignment_baseline="'bottom'",
        get_pixel_offset=[0, -14],
    )

    deck = pdk.Deck(
        layers=[polygon_layer, marker_layer, text_layer],
        initial_view_state=pdk.ViewState(latitude=34.7, longitude=-109.5, zoom=3.7),
        tooltip={"text": "{tooltip}"},
    )
    st.pydeck_chart(deck, use_container_width=True)

    detail = data[selected_region]
    st.caption(f"{selected_region} mean {overlay}: {detail[col_name].dropna().mean():,.2f} {unit}")

