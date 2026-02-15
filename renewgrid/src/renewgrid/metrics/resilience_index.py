"""Resilience index calculations."""

from __future__ import annotations

import pandas as pd


def energy_served_ratio(served: pd.Series, demand: pd.Series) -> float:
    """Compute total served energy divided by total demand."""
    demand_total = float(demand.sum())
    if demand_total == 0:
        return 0.0
    return float(served.sum() / demand_total)
