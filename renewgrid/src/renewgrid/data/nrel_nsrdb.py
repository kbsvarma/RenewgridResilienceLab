"""NREL NSRDB/Wind Toolkit API placeholders (Phase 1 scaffold)."""

from __future__ import annotations

from datetime import date

import pandas as pd


def fetch_nsrdb_daily(latitude: float, longitude: float, start_date: date, end_date: date) -> pd.DataFrame:
    """Return an empty NSRDB frame placeholder until endpoint wiring is enabled."""
    _ = (latitude, longitude, start_date, end_date)
    return pd.DataFrame(columns=["date", "ghi", "dni", "dhi", "source"])
