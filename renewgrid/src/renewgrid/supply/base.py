"""Supply model interfaces for Phase 2 stress simulations."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class SupplyModel(Protocol):
    """Protocol for deterministic supply estimators."""

    def generate(self, df: pd.DataFrame, config: dict[str, float]) -> pd.DataFrame:
        """Return DataFrame with solar_mw, wind_mw, gen_total_mw (and optional cfs)."""

