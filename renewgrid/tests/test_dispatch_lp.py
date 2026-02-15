"""Tests for dispatch module."""

from __future__ import annotations

import pandas as pd
import pytest

from renewgrid.opt.dispatch_lp import greedy_dispatch

pytestmark = pytest.mark.phase3


def test_greedy_dispatch_tracks_unserved() -> None:
    """Dispatch output should match min(demand, available) and non-negative unserved."""
    demand = pd.Series([100.0, 80.0])
    available = pd.Series([90.0, 120.0])

    result = greedy_dispatch(demand, available)

    assert result["served"].tolist() == [90.0, 80.0]
    assert result["unserved"].tolist() == [10.0, 0.0]
