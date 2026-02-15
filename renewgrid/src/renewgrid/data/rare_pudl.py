"""RARE via PUDL connector placeholders."""

from __future__ import annotations

import pandas as pd


def load_rare_pudl_sample() -> pd.DataFrame:
    """Return an empty RARE/PUDL frame placeholder for Phase 1."""
    return pd.DataFrame(columns=["plant_id", "date", "value", "source"])
