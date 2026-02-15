"""Deterministic Phase 2 stress scenario definitions and transforms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScenarioParams:
    """Scenario multipliers/derates for deterministic stress transforms."""

    demand_multiplier: float
    solar_derate_factor: float
    wind_derate_factor: float
    wind_drought_days: int = 0


@dataclass(frozen=True)
class Scenario:
    """Named stress scenario with stable identifier and params."""

    scenario_id: str
    name: str
    params: ScenarioParams


def get_scenario(scenario_id: str, wind_drought_days: int = 5) -> Scenario:
    """Return a deterministic Scenario by id."""
    sid = scenario_id.lower().strip()
    if sid == "heat_wave":
        return Scenario(
            scenario_id="heat_wave",
            name="Heat Wave",
            params=ScenarioParams(
                demand_multiplier=1.25,
                solar_derate_factor=0.85,
                wind_derate_factor=1.00,
                wind_drought_days=0,
            ),
        )
    if sid == "wind_drought":
        return Scenario(
            scenario_id="wind_drought",
            name="Wind Drought",
            params=ScenarioParams(
                demand_multiplier=1.00,
                solar_derate_factor=1.00,
                wind_derate_factor=0.40,
                wind_drought_days=wind_drought_days,
            ),
        )
    if sid == "demand_shock":
        return Scenario(
            scenario_id="demand_shock",
            name="Demand Shock",
            params=ScenarioParams(
                demand_multiplier=1.15,
                solar_derate_factor=1.00,
                wind_derate_factor=1.00,
                wind_drought_days=0,
            ),
        )
    if sid == "compound":
        return Scenario(
            scenario_id="compound",
            name="Compound",
            params=ScenarioParams(
                demand_multiplier=1.40,
                solar_derate_factor=0.85,
                wind_derate_factor=0.40,
                wind_drought_days=wind_drought_days,
            ),
        )
    raise ValueError(f"Unsupported scenario_id: {scenario_id}")


def _wind_drought_window_mask(
    frame: pd.DataFrame,
    days: int,
    wind_col: str = "weather_ws10m",
) -> pd.Series:
    """Select deterministic N-day drought window by lowest rolling wind mean."""
    if days <= 0 or wind_col not in frame.columns or frame.empty:
        return pd.Series([False] * len(frame), index=frame.index)

    ws = pd.to_numeric(frame[wind_col], errors="coerce")
    rolling = ws.rolling(window=days, min_periods=days).mean()
    rolling_vals = rolling.to_numpy(dtype=float)
    valid_positions = np.where(~np.isnan(rolling_vals))[0]
    if len(valid_positions) == 0:
        return pd.Series([False] * len(frame), index=frame.index)

    min_pos = int(valid_positions[np.argmin(rolling_vals[valid_positions])])
    start_idx = max(0, min_pos - days + 1)
    end_idx = min(len(frame), start_idx + days)
    mask = pd.Series([False] * len(frame), index=frame.index)
    mask.iloc[start_idx:end_idx] = True
    return mask


def apply_scenario(
    daily_df: pd.DataFrame,
    scenario: Scenario,
    config: dict[str, float | int] | None = None,
) -> pd.DataFrame:
    """Apply deterministic scenario transform to base daily frame."""
    cfg = config or {}
    out = daily_df.copy()

    if "demand_mw_base" not in out.columns:
        if "demand_mw_avg" not in out.columns:
            raise ValueError("Input dataframe must include demand_mw_avg or demand_mw_base")
        out["demand_mw_base"] = pd.to_numeric(out["demand_mw_avg"], errors="coerce")

    demand_multiplier = float(cfg.get("demand_multiplier", scenario.params.demand_multiplier))
    solar_derate_factor = float(cfg.get("solar_derate_factor", scenario.params.solar_derate_factor))
    wind_derate_factor = float(cfg.get("wind_derate_factor", scenario.params.wind_derate_factor))
    wind_days = int(cfg.get("wind_drought_days", scenario.params.wind_drought_days))

    out["demand_mw_stressed"] = out["demand_mw_base"] * demand_multiplier
    out["solar_derate_factor"] = solar_derate_factor
    out["wind_derate_factor"] = 1.0
    out["wind_drought_active"] = False

    apply_wind_window = scenario.scenario_id in {"wind_drought", "compound"} and wind_days > 0
    if apply_wind_window:
        mask = _wind_drought_window_mask(out, wind_days, wind_col="weather_ws10m")
        out.loc[mask, "wind_derate_factor"] = wind_derate_factor
        out.loc[mask, "wind_drought_active"] = True
    elif scenario.scenario_id in {"heat_wave", "demand_shock"}:
        out["wind_derate_factor"] = scenario.params.wind_derate_factor

    out["scenario_id"] = scenario.scenario_id
    out["scenario_name"] = scenario.name
    return out
