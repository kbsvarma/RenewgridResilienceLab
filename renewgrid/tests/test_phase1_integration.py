"""Network-free integration test for Phase 1 pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from renewgrid.data import merge
from renewgrid.features.build_features import build_feature_frame
from renewgrid.forecast.evaluate import rolling_origin_evaluate, save_evaluation_report
from renewgrid.util.schema import validate_daily_frame


def test_phase1_end_to_end_mocked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Build daily dataset, engineer features, and evaluate persistence without network."""

    def fake_weather(*args: object, **kwargs: object) -> pd.DataFrame:
        _ = (args, kwargs)
        days = pd.date_range("2024-01-01", periods=30, freq="D")
        return pd.DataFrame(
            {
                "date": days,
                "weather_t2m": np.linspace(10.0, 20.0, 30),
                "weather_ws10m": np.linspace(3.0, 6.0, 30),
                "weather_allsky_sfc_sw_dwn": np.linspace(2.0, 5.0, 30),
            }
        )

    def fake_daily(region: str, *args: object, **kwargs: object) -> pd.DataFrame:
        _ = (args, kwargs)
        days = pd.date_range("2024-01-01", periods=30, freq="D")
        return pd.DataFrame(
            {
                "date": days,
                "demand_mw_avg": np.linspace(40000.0, 43000.0, 30),
                "source": ["eia"] * 30,
                "region": ["CISO" if region == "CAISO" else "ERCO"] * 30,
            }
        )

    def fake_rare(*args: object, **kwargs: object) -> pd.DataFrame:
        _ = (args, kwargs)
        days = pd.date_range("2024-01-01", periods=30, freq="D")
        return pd.DataFrame(
            {
                "date": days,
                "solar_cf": np.linspace(0.2, 0.3, 30),
                "wind_cf": np.linspace(0.3, 0.4, 30),
            }
        )

    monkeypatch.setattr(merge, "fetch_daily_weather", fake_weather)
    monkeypatch.setattr(merge, "fetch_rto_daily", fake_daily)
    monkeypatch.setattr(merge, "load_rare_daily_generation", fake_rare)

    for region in ("CAISO", "ERCOT"):
        dataset = merge.build_daily_dataset(
            region=region,
            start_date=pd.Timestamp("2024-01-01").date(),
            end_date=pd.Timestamp("2024-01-30").date(),
            base_dir=tmp_path,
            rare_path="dummy.parquet",
        )
        validate_daily_frame(dataset)
        features = build_feature_frame(dataset, target_col="demand_mw_avg")
        feature_cols = [
            c
            for c in features.columns
            if c not in {"date", "source", "region", "demand_mw_avg", "solar_cf", "wind_cf"}
        ]
        evaluation = rolling_origin_evaluate(
            frame=features,
            target_col="demand_mw_avg",
            feature_cols=feature_cols,
            horizons=(1, 2, 3),
            min_train_size=14,
            models=("persistence",),
        )

        summary = evaluation["summary"]
        preds = evaluation["predictions"]
        assert np.isfinite(summary["skill_vs_persistence"]).all()
        assert preds["y_true"].notna().all()
        assert preds["y_pred"].notna().all()

        report_paths = save_evaluation_report(
            evaluation=evaluation,
            region=region,
            target_col="demand_mw_avg",
            output_dir=tmp_path / "reports" / "phase1",
        )
        assert report_paths["json"].exists()
        assert report_paths["markdown"].exists()
