"""Tests for daily dataframe schema validation."""

from __future__ import annotations

import pandas as pd
import pytest

from renewgrid.util.schema import validate_daily_frame


def test_validate_daily_frame_catches_duplicate_dates() -> None:
    """Duplicate dates should fail schema validation."""
    frame = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "demand_mw_avg": [100.0, 101.0],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_daily_frame(frame)


def test_validate_daily_frame_catches_negative_demand() -> None:
    """Negative demand should fail schema validation."""
    frame = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "demand_mw_avg": [100.0, -5.0],
        }
    )
    with pytest.raises(ValueError, match="non-negative"):
        validate_daily_frame(frame)


def test_validate_daily_frame_allows_missing_optional_columns() -> None:
    """Frame with required columns only should pass."""
    frame = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "demand_mw_avg": [100.0, 105.0],
        }
    )
    validate_daily_frame(frame)
