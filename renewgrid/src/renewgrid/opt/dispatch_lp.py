"""Phase 3 module (not used in Phase 2). Do not import from Phase 2 UI."""

from __future__ import annotations

import pandas as pd


def greedy_dispatch(demand: pd.Series, available: pd.Series) -> pd.DataFrame:
    """Dispatch available supply against demand and report unserved energy."""
    served = pd.concat([demand, available], axis=1).min(axis=1)
    unserved = (demand - served).clip(lower=0)
    return pd.DataFrame({"served": served, "unserved": unserved})
