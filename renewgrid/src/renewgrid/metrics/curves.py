"""Phase 3 module (not used in Phase 2). Do not import from Phase 2 UI."""

from __future__ import annotations

import pandas as pd


def cumulative_unserved(unserved: pd.Series) -> pd.Series:
    """Return cumulative unserved energy curve."""
    return unserved.cumsum()
