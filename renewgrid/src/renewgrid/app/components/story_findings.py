"""Deterministic findings generator for story-chart selected windows."""

from __future__ import annotations

from typing import Any

import pandas as pd

SERIES_TO_COL: dict[str, str] = {
    "Demand (MW avg)": "demand_mw_avg",
    "Temperature (T2M)": "weather_t2m",
    "Wind Speed (WS10M)": "weather_ws10m",
    "Solar Proxy (ALLSKY)": "weather_allsky_sfc_sw_dwn",
}


def _kfmt(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value/1000.0:.1f}k"
    return f"{value:.1f}"


def generate_findings(
    df: pd.DataFrame,
    selected_series: list[str],
    region: str,
    start_date: Any,
    end_date: Any,
) -> list[str]:
    """Generate plain-language findings for the selected story-chart window."""
    findings: list[str] = []

    findings.append(
        f"Window covers {region} from {start_date} to {end_date} at daily (UTC-day) resolution."
    )

    demand_col = SERIES_TO_COL["Demand (MW avg)"]
    if demand_col in df.columns:
        demand = pd.to_numeric(df[demand_col], errors="coerce").dropna()
        if not demand.empty:
            mean_val = float(demand.mean())
            min_val = float(demand.min())
            max_val = float(demand.max())
            swing_pct = ((max_val - min_val) / mean_val * 100.0) if mean_val != 0 else 0.0
            findings.append(
                "Demand averaged "
                f"{_kfmt(mean_val)} MW and ranged from {_kfmt(min_val)} to {_kfmt(max_val)} MW "
                f"(~{swing_pct:.0f}% swing)."
            )

            demand_diff = demand.diff().dropna()
            if not demand_diff.empty:
                jump_date = demand_diff.abs().idxmax()
                jump = float(demand_diff.loc[jump_date])
                date_value = pd.to_datetime(df.loc[jump_date, "date"]).date()
                findings.append(
                    f"Largest 1-day demand move was {jump:+.0f} MW on {date_value}."
                )

    temp_col = SERIES_TO_COL["Temperature (T2M)"]
    if temp_col in df.columns and "Temperature (T2M)" in selected_series:
        temp = pd.to_numeric(df[temp_col], errors="coerce").dropna()
        if not temp.empty:
            cold_idx = temp.idxmin()
            warm_idx = temp.idxmax()
            cold_day = pd.to_datetime(df.loc[cold_idx, "date"]).date()
            warm_day = pd.to_datetime(df.loc[warm_idx, "date"]).date()
            findings.append(
                f"Coldest day was {cold_day} (~{float(temp.loc[cold_idx]):.1f}°C); "
                f"warmest day was {warm_day} (~{float(temp.loc[warm_idx]):.1f}°C)."
            )

    if {"Demand (MW avg)", "Temperature (T2M)"}.issubset(set(selected_series)):
        aligned = df[[demand_col, temp_col]].dropna()
        n_overlap = len(aligned)
        if n_overlap < 5:
            findings.append("Not enough overlapping data to compute stable correlation.")
        else:
            corr = float(aligned[demand_col].corr(aligned[temp_col]))
            if pd.isna(corr):
                findings.append("Not enough overlapping data to compute stable correlation.")
            elif abs(corr) < 0.30:
                findings.append(
                    f"Demand-temperature correlation is {corr:.2f} (weak) based on {n_overlap} "
                    "overlapping days; hourly or lag effects may differ."
                )
            elif abs(corr) < 0.60:
                direction = (
                    "colder days tended to coincide with higher demand"
                    if corr < 0
                    else "warmer days tended to coincide with higher demand"
                )
                findings.append(
                    f"Demand-temperature correlation is {corr:.2f} (moderate) based on "
                    f"{n_overlap} overlapping days: {direction}."
                )
            elif corr <= -0.60:
                findings.append(
                    f"Demand-temperature correlation is {corr:.2f} (strong) based on {n_overlap} "
                    "overlapping days: colder days tended to coincide with higher demand."
                )
            else:
                findings.append(
                    f"Demand-temperature correlation is {corr:.2f} (strong) based on {n_overlap} "
                    "overlapping days: warmer days tended to coincide with higher demand."
                )

    return findings[:6]
