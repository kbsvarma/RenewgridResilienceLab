"""Tests for RARE canonical schema normalization."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from renewgrid.data.rare_pudl import normalize_rare_frame


def test_normalize_rare_keeps_capacity_factors() -> None:
    """Existing CF columns should pass through unchanged."""
    frame = pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "region": ["CAISO"],
            "solar_cf": [0.3],
            "wind_cf": [0.4],
        }
    )
    out = normalize_rare_frame(frame)
    assert list(out.columns) == ["date", "region", "solar_cf", "wind_cf"]


def test_normalize_rare_computes_cf_from_gen_and_capacity() -> None:
    """Generation plus capacity should convert to canonical capacity factors."""
    frame = pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "region": ["ERCOT"],
            "solar_gen": [300.0],
            "wind_gen": [400.0],
            "solar_capacity": [1000.0],
            "wind_capacity": [800.0],
        }
    )
    out = normalize_rare_frame(frame)
    assert out["solar_cf"].iloc[0] == 0.3
    assert out["wind_cf"].iloc[0] == 0.5


def test_normalize_rare_warns_and_renames_when_capacity_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Generation without capacity should be retained as _gen_mwh with warning."""
    caplog.set_level(logging.WARNING)
    frame = pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "region": ["ERCOT"],
            "solar_gen": [300.0],
            "wind_gen": [400.0],
        }
    )
    out = normalize_rare_frame(frame)
    assert {"solar_gen_mwh", "wind_gen_mwh"}.issubset(set(out.columns))
    assert "cannot compute capacity factor" in caplog.text
