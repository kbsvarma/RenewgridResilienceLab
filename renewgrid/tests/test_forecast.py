"""Tests for forecast utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from renewgrid.features.build_features import build_feature_frame
from renewgrid.forecast.baseline import persistence_forecast
from renewgrid.forecast.evaluate import (
    mean_absolute_error,
    rolling_origin_evaluate,
    save_evaluation_report,
)
from renewgrid.scripts import phase1_run


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


def test_phase1_runs_optional_targets_when_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Phase1 runner should evaluate demand plus solar/wind when present."""
    dataset = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=40, freq="D"),
            "region": ["CAISO"] * 40,
            "demand_mw_avg": np.linspace(100.0, 130.0, 40),
            "weather_t2m": np.linspace(15.0, 20.0, 40),
            "weather_ws10m": np.linspace(3.0, 7.0, 40),
            "weather_allsky_sfc_sw_dwn": np.linspace(2.0, 5.0, 40),
            "solar_gen": np.linspace(10.0, 20.0, 40),
            "wind_gen": np.linspace(8.0, 16.0, 40),
        }
    )

    monkeypatch.setattr(phase1_run, "build_daily_dataset", lambda *args, **kwargs: dataset.copy())

    def fake_eval(*args: object, **kwargs: object) -> dict[str, object]:
        _ = (args, kwargs)
        summary = pd.DataFrame(
            [
                {
                    "model": "persistence",
                    "horizon": 1,
                    "mae": 1.0,
                    "rmse": 1.0,
                    "mape": 1.0,
                    "skill_vs_persistence": 0.0,
                }
            ]
        )
        return {
            "availability": {"persistence": True, "prophet": False, "xgboost": False},
            "predictions": pd.DataFrame(),
            "summary": summary,
        }

    monkeypatch.setattr(phase1_run, "rolling_origin_evaluate", fake_eval)

    def fake_save(evaluation: dict[str, object], region: str, target_col: str, output_dir: Path) -> dict[str, Path]:
        _ = evaluation
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / f"{region.lower()}_{target_col}_report.json"
        md_path = out / f"{region.lower()}_{target_col}_report.md"
        json_path.write_text("{}", encoding="utf-8")
        md_path.write_text("# report", encoding="utf-8")
        return {"json": json_path, "markdown": md_path}

    monkeypatch.setattr(phase1_run, "save_evaluation_report", fake_save)

    outputs = phase1_run.run_phase1(
        base_dir=tmp_path,
        start_date=pd.Timestamp("2024-01-01").date(),
        end_date=pd.Timestamp("2024-02-09").date(),
        rare_path=None,
    )
    for region in ("CAISO", "ERCOT"):
        assert {"demand_mw_avg", "solar_gen", "wind_gen"}.issubset(set(outputs[region].keys()))
