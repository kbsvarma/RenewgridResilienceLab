"""Resilience curve utilities."""

from __future__ import annotations

import pandas as pd


def cumulative_unserved(unserved: pd.Series) -> pd.Series:
    """Return cumulative unserved energy curve."""
    return unserved.cumsum()
