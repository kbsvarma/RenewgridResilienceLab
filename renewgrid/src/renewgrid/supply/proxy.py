"""Proxy supply estimation using NASA POWER weather variables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from renewgrid.supply.base import SupplyModel


class ProxySupplyModel(SupplyModel):
    """Estimate daily solar/wind generation from weather proxies."""

    def generate(self, df: pd.DataFrame, config: dict[str, float]) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        warnings: list[str] = []

        solar_capacity_mw = float(config.get("solar_capacity_mw", 0.0))
        wind_capacity_mw = float(config.get("wind_capacity_mw", 0.0))

        solar_derate = pd.to_numeric(df.get("solar_derate_factor", 1.0), errors="coerce").fillna(1.0)
        wind_derate = pd.to_numeric(df.get("wind_derate_factor", 1.0), errors="coerce").fillna(1.0)

        if "weather_allsky_sfc_sw_dwn" in df.columns:
            irradiance = pd.to_numeric(df["weather_allsky_sfc_sw_dwn"], errors="coerce")
            p05 = float(irradiance.quantile(0.05))
            p95 = float(irradiance.quantile(0.95))
            denom = p95 - p05
            if denom <= 0:
                solar_cf = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
            else:
                solar_cf = ((irradiance - p05) / denom).clip(lower=0.0, upper=1.0)
        else:
            solar_cf = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
            warnings.append("Missing weather_allsky_sfc_sw_dwn; solar proxy set to 0.")

        if "weather_ws10m" in df.columns:
            ws = pd.to_numeric(df["weather_ws10m"], errors="coerce")
            wind_cf = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
            wind_cf = wind_cf.mask((ws >= 3.0) & (ws < 12.0), (ws - 3.0) / 9.0)
            wind_cf = wind_cf.mask((ws >= 12.0) & (ws <= 25.0), 1.0)
            wind_cf = wind_cf.mask(ws > 25.0, 0.3)
            wind_cf = wind_cf.fillna(0.0).clip(lower=0.0, upper=1.0)
        else:
            wind_cf = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
            warnings.append("Missing weather_ws10m; wind proxy set to 0.")

        out["solar_cf"] = solar_cf
        out["wind_cf"] = wind_cf
        out["solar_mw"] = solar_capacity_mw * solar_cf * solar_derate
        out["wind_mw"] = wind_capacity_mw * wind_cf * wind_derate
        out["gen_total_mw"] = out["solar_mw"] + out["wind_mw"]

        out.attrs["warnings"] = warnings
        return out
