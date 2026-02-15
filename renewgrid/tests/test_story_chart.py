"""Tests for story chart helpers."""

from __future__ import annotations

import pandas as pd

from renewgrid.app.components.story_chart import compute_axis_range


def test_compute_axis_range_fit_to_data_excludes_zero_baseline() -> None:
    """Fit-to-data range should track the data band and not force zero."""
    series = pd.Series([21000.0, 23000.0, 26000.0])
    lower, upper = compute_axis_range(series, include_zero=False)
    assert lower > 0.0
    assert lower < 21000.0
    assert upper > 26000.0


def test_compute_axis_range_include_zero_baseline_starts_at_zero() -> None:
    """Zero-baseline mode should force a zero lower bound."""
    series = pd.Series([21000.0, 23000.0, 26000.0])
    lower, upper = compute_axis_range(series, include_zero=True)
    assert lower == 0.0
    assert upper > 26000.0

