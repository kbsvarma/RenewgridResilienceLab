"""Tests for unit conversion and validation helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from renewgrid.util.units import assert_daily_mw_avg, mw_avg_to_mwh, mwh_to_mw_avg


def test_mw_mwh_round_trip() -> None:
    """MW-average to MWh and back should preserve values."""
    series = pd.Series([100.0, 125.5, 200.0])
    restored = mwh_to_mw_avg(mw_avg_to_mwh(series))
    assert restored.equals(series)


def test_assert_daily_mw_avg_catches_negative() -> None:
    """Daily MW-average validator should reject negative demand."""
    frame = pd.DataFrame({"demand_mw_avg": [100.0, -1.0, 90.0]})
    with pytest.raises(ValueError, match="negative"):
        assert_daily_mw_avg(frame)
