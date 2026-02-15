"""Tests for persistence baseline shift behavior."""

from __future__ import annotations

import pandas as pd

from renewgrid.forecast.baseline import persistence_forecast


def test_persistence_forecast_uses_previous_day_value() -> None:
    """Persistence forecast should be y[t-1] with no backfilled leakage."""
    frame = pd.DataFrame({"value": [100.0, 110.0, 120.0, 130.0]})
    y_hat = persistence_forecast(frame, target_col="value")
    assert pd.isna(y_hat.iloc[0])
    assert y_hat.iloc[1] == 100.0
    assert y_hat.iloc[2] == 110.0
    assert y_hat.iloc[3] == 120.0

