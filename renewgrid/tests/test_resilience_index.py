"""Tests for resilience metrics."""

from __future__ import annotations

import pandas as pd

from renewgrid.metrics.curves import cumulative_unserved
from renewgrid.metrics.resilience_index import energy_served_ratio


def test_energy_served_ratio() -> None:
    """Served ratio should be served sum over demand sum."""
    served = pd.Series([80.0, 90.0])
    demand = pd.Series([100.0, 100.0])
    assert energy_served_ratio(served, demand) == 0.85


def test_cumulative_unserved() -> None:
    """Cumulative unserved should be monotonic cumsum."""
    unserved = pd.Series([1.0, 2.0, 3.0])
    assert cumulative_unserved(unserved).tolist() == [1.0, 3.0, 6.0]
