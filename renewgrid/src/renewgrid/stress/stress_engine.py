"""Stress test engine placeholder for later Phase 2."""

from __future__ import annotations

import pandas as pd


def apply_outage(frame: pd.DataFrame, outage_fraction: float) -> pd.DataFrame:
    """Reduce served load by outage fraction and recompute unserved energy."""
    stressed = frame.copy()
    stressed["served"] = stressed["served"] * (1.0 - outage_fraction)
    stressed["unserved"] = (stressed["demand_mw_avg"] - stressed["served"]).clip(lower=0)
    return stressed
