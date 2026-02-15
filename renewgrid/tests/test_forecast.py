"""Tests for forecast utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from renewgrid.forecast.baseline import persistence_forecast
from renewgrid.forecast.evaluate import mean_absolute_error


def test_persistence_forecast_returns_shifted_series() -> None:
    """Persistence forecast should return prior-value predictions."""
    frame = pd.DataFrame({"value": [10.0, 12.0, 14.0]})
    pred = persistence_forecast(frame)
    assert pred.tolist() == [10.0, 10.0, 12.0]


def test_mean_absolute_error() -> None:
    """MAE should match expected arithmetic mean absolute error."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.5, 2.0])
    assert mean_absolute_error(y_true, y_pred) == 0.5
