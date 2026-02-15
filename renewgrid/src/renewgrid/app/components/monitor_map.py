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
    "Temperature": ("weather_t2m", "°C"),
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


def _status_from_latest(series: pd.Series) -> str:
    clean = series.dropna()
    if clean.empty:
        return "NORMAL"
    latest = float(clean.iloc[-1])
    p33 = float(clean.quantile(0.33))
    p67 = float(clean.quantile(0.67))
    if latest < p33:
        return "LOW"
    if latest > p67:
        return "HIGH"
    return "NORMAL"


def _anomaly_fill_color(anomaly: float, scale: float) -> list[int]:
    """Map demand anomaly to a diverging blue-neutral-red fill color."""
    if scale <= 0:
        return [160, 160, 160, 110]
    ratio = max(-1.0, min(1.0, anomaly / scale))
    intensity = int(70 + 150 * abs(ratio))
    if ratio > 0:
        return [intensity, 80, 80, 130]
    if ratio < 0:
        return [80, 110, intensity, 130]
    return [150, 150, 150, 120]


def render_monitor_map(
    base_dir: Path,
    dataset_by_region: dict[str, pd.DataFrame] | None = None,
    key_prefix: str = "monitor_map",
) -> None:
    """Render map with region overlays and plain-English tooltips."""
    st.subheader("Where is this region?")
    st.caption("Click CAISO or ERCOT to view its demand and weather for the selected window.")
    st.caption("Latest available daily window (cached 10 minutes)")
    if pdk is None:
        st.error(
            "Monitor map requires pydeck. Install dependencies with "
            "`uv sync --extra dev` or `pip install -e \".[dev]\"`."
        )
        return

    overlay = st.selectbox(
        "Overlay",
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
    demand_anomalies: list[float] = []
    for feature in features:
        props = feature.get("properties", {})
        region = str(props.get("name", props.get("region", ""))).upper()
        if region not in data or col_name not in data[region].columns:
            continue
        region_frame = data[region]
        series = region_frame[col_name].dropna()
        if series.empty:
            continue
        demand_series = region_frame["demand_mw_avg"].dropna() if "demand_mw_avg" in region_frame.columns else pd.Series(dtype=float)
        demand_mean = float(demand_series.mean()) if not demand_series.empty else 0.0
        demand_latest = float(demand_series.iloc[-1]) if not demand_series.empty else 0.0
        demand_anomaly = demand_latest - demand_mean
        demand_anomalies.append(demand_anomaly)
        latest_value = float(series.iloc[-1])
        mean_value = float(series.mean())
        label_upper = _status_from_latest(series)
        overlay_value = mean_value
        source_name = (
            "EIA Open Data + NASA POWER"
            if col_name.startswith("weather_")
            else "EIA Open Data"
        )
        regions_df.append(
            {
                "region": region,
                "name": props.get("name", region),
                "lat": float(props.get("lat", 0.0)),
                "lon": float(props.get("lon", 0.0)),
                "value": overlay_value,
                "unit": unit,
                "label": label_upper,
                "mean_value": mean_value,
                "latest_value": latest_value,
                "demand_mean": demand_mean,
                "demand_latest": demand_latest,
                "demand_anomaly": demand_anomaly,
                "source": source_name,
                "tooltip": (
                    f"{region} - {overlay} is {label_upper} vs this window's normal\n"
                    f"Mean: {mean_value:,.2f} {unit} | Latest: {latest_value:,.2f} {unit}\n"
                    f"Demand anomaly: {demand_anomaly:+,.2f} MW\n"
                    f"Source: {source_name}"
                ),
                "geometry": feature.get("geometry"),
            }
        )

    if not regions_df:
        st.warning("Selected overlay is unavailable in current datasets.")
        return

    anomaly_scale = max((abs(v) for v in demand_anomalies), default=1.0)
    for row in regions_df:
        row["fill_color"] = _anomaly_fill_color(float(row["demand_anomaly"]), anomaly_scale)
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
        get_fill_color="fill_color",
        get_line_color="[43, 89, 71, 210]",
        line_width_min_pixels=2,
    )
    marker_layer = pdk.Layer(
        "ScatterplotLayer",
        data=marker_df,
        get_position="[lon, lat]",
        get_radius=85000,
        get_fill_color="[245, 245, 245, 230]",
        get_line_color="[20, 20, 20, 220]",
        stroked=True,
        line_width_min_pixels=1,
        pickable=True,
    )

    deck = pdk.Deck(
        layers=[polygon_layer, marker_layer],
        initial_view_state=pdk.ViewState(latitude=37.0, longitude=-98.0, zoom=3.0),
        map_style="mapbox://styles/mapbox/dark-v11",
        tooltip={"text": "{tooltip}"},
    )
    st.pydeck_chart(deck, use_container_width=True, height=470)

    detail = data[selected_region]
    detail_series = detail[col_name].dropna()
    if not detail_series.empty:
        mean_value = float(detail_series.mean())
        latest_value = float(detail_series.iloc[-1])
        st.caption(
            f"{selected_region} {overlay}: mean {mean_value:,.2f} {unit} | latest {latest_value:,.2f} {unit}"
        )
