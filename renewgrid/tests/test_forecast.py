"""Tests for forecast utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from renewgrid.features.build_features import build_feature_frame
from renewgrid.forecast.baseline import persistence_forecast
from renewgrid.forecast.evaluate import (
    mean_absolute_error,
    rolling_origin_evaluate,
    save_evaluation_report,
)


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


def test_feature_engineering_no_leakage() -> None:
    """Lag and rolling features must be NaN at sequence start (no forward fill leakage)."""
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "demand_mw_avg": np.arange(10, dtype=float),
        }
    )
    features = build_feature_frame(frame, target_col="demand_mw_avg")
    assert pd.isna(features.loc[0, "demand_mw_avg_lag_1"])
    assert pd.isna(features.loc[0, "demand_mw_avg_roll_7"])
    assert pd.isna(features.loc[6, "demand_mw_avg_lag_7"])


def test_evaluate_runs_end_to_end(tmp_path: Path) -> None:
    """Rolling evaluation should produce summary and save report artifacts."""
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=40, freq="D"),
            "demand_mw_avg": np.linspace(100.0, 130.0, 40),
            "dow": [d.dayofweek for d in pd.date_range("2024-01-01", periods=40, freq="D")],
            "month": [1] * 31 + [2] * 9,
            "demand_mw_avg_lag_1": pd.Series(np.linspace(100.0, 130.0, 40)).shift(1),
            "demand_mw_avg_lag_7": pd.Series(np.linspace(100.0, 130.0, 40)).shift(7),
            "demand_mw_avg_roll_7": pd.Series(np.linspace(100.0, 130.0, 40)).shift(1).rolling(7).mean(),
            "demand_mw_avg_roll_14": pd.Series(np.linspace(100.0, 130.0, 40)).shift(1).rolling(14).mean(),
        }
    )
    feature_cols = [
        "dow",
        "month",
        "demand_mw_avg_lag_1",
        "demand_mw_avg_lag_7",
        "demand_mw_avg_roll_7",
        "demand_mw_avg_roll_14",
    ]
    evaluation = rolling_origin_evaluate(
        frame=frame,
        target_col="demand_mw_avg",
        feature_cols=feature_cols,
        horizons=(1, 2, 3),
        min_train_size=20,
        models=("persistence",),
    )
    summary = evaluation["summary"]
    assert isinstance(summary, pd.DataFrame)
    assert "persistence" in set(summary["model"].tolist())

    paths = save_evaluation_report(evaluation, "ERCOT", "demand_mw_avg", tmp_path / "reports")
    assert paths["json"].exists()
    assert paths["markdown"].exists()
