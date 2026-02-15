"""Deterministic greedy battery dispatch simulator for Phase 2."""

from __future__ import annotations

import math

import pandas as pd


def simulate_battery_dispatch(
    frame: pd.DataFrame,
    battery_config: dict[str, float],
) -> pd.DataFrame:
    """Simulate daily battery charging/discharging against net load in MW."""
    if "net_load_mw" not in frame.columns:
        raise ValueError("simulate_battery_dispatch requires net_load_mw column")

    out = frame.copy()
    energy_mwh = max(0.0, float(battery_config.get("energy_mwh", 0.0)))
    power_mw = max(0.0, float(battery_config.get("power_mw", 0.0)))
    roundtrip_eff = float(battery_config.get("roundtrip_efficiency", 0.9))
    roundtrip_eff = min(1.0, max(0.01, roundtrip_eff))
    initial_soc_pct = min(1.0, max(0.0, float(battery_config.get("initial_soc_pct", 0.5))))

    eta_c = math.sqrt(roundtrip_eff)
    eta_d = math.sqrt(roundtrip_eff)
    power_limit_mwh = power_mw * 24.0
    soc = energy_mwh * initial_soc_pct

    charge_mw: list[float] = []
    discharge_mw: list[float] = []
    soc_mwh: list[float] = []
    unserved_mw: list[float] = []
    curtailment_mw: list[float] = []

    for net_load in pd.to_numeric(out["net_load_mw"], errors="coerce").fillna(0.0):
        net_mwh = float(net_load) * 24.0

        if net_mwh > 0:
            discharge_from_soc = min(soc, power_limit_mwh, net_mwh / eta_d if eta_d > 0 else 0.0)
            delivered_mwh = discharge_from_soc * eta_d
            soc -= discharge_from_soc

            remaining_deficit_mwh = max(0.0, net_mwh - delivered_mwh)
            unserved = remaining_deficit_mwh / 24.0
            curtail = 0.0
            charge = 0.0
            discharge = delivered_mwh / 24.0

        elif net_mwh < 0:
            surplus_mwh = -net_mwh
            stored_mwh = min((energy_mwh - soc), power_limit_mwh, surplus_mwh * eta_c)
            absorbed_mwh = stored_mwh / eta_c if eta_c > 0 else 0.0
            soc += stored_mwh

            curtailment_residual_mwh = max(0.0, surplus_mwh - absorbed_mwh)
            unserved = 0.0
            curtail = curtailment_residual_mwh / 24.0
            charge = absorbed_mwh / 24.0
            discharge = 0.0

        else:
            unserved = 0.0
            curtail = 0.0
            charge = 0.0
            discharge = 0.0

        soc = min(max(soc, 0.0), energy_mwh)
        soc_mwh.append(soc)
        charge_mw.append(max(0.0, charge))
        discharge_mw.append(max(0.0, discharge))
        unserved_mw.append(max(0.0, unserved))
        curtailment_mw.append(max(0.0, curtail))

    out["battery_charge_mw"] = charge_mw
    out["battery_discharge_mw"] = discharge_mw
    out["soc_mwh"] = soc_mwh
    out["unserved_mw"] = unserved_mw
    out["curtailment_mw"] = curtailment_mw
    return out
