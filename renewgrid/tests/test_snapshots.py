"""Tests for snapshot save/list/load helpers."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from renewgrid.app.snapshots import delete_snapshot, list_snapshots, load_snapshot, save_snapshot
from renewgrid.app.state import RunConfig


def test_save_snapshot_writes_expected_files(tmp_path: Path) -> None:
    """Snapshot save should emit manifest and JSON artifacts."""
    config = RunConfig(
        region="CAISO",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 30),
        timeframe_preset="30D",
        use_rare=False,
        rare_path=None,
        model_choice="persistence",
        horizon_days=3,
    )
    summary = {"rows": 30, "region": "CAISO"}
    eval_results = {
        "availability": {"persistence": True},
        "summary": pd.DataFrame(
            [
                {
                    "model": "persistence",
                    "horizon": 1,
                    "mae": 1.0,
                    "rmse": 1.2,
                    "mape": 1.5,
                    "skill_vs_persistence": 0.0,
                }
            ]
        ),
        "predictions": pd.DataFrame(
            [
                {
                    "model": "persistence",
                    "horizon": 1,
                    "date": "2024-01-20",
                    "y_true": 10.0,
                    "y_pred": 9.5,
                }
            ]
        ),
    }
    key_series = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=3), "demand_mw_avg": [1.0, 2.0, 3.0]}
    )

    run_id = save_snapshot(config, summary, eval_results, tmp_path, key_series)
    run_dir = tmp_path / "reports" / "runs" / run_id
    assert (run_dir / "run_config.json").exists()
    assert (run_dir / "dataset_summary.json").exists()
    assert (run_dir / "eval_results.json").exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "key_series.parquet").exists()


def test_list_and_load_snapshots(tmp_path: Path) -> None:
    """Saved snapshots should be discoverable and loadable."""
    config = RunConfig(
        region="ERCOT",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 30),
        timeframe_preset="30D",
        use_rare=False,
        rare_path=None,
        model_choice="persistence",
        horizon_days=1,
    )
    run_id = save_snapshot(config, {"rows": 30}, {"availability": {}, "summary": pd.DataFrame(), "predictions": pd.DataFrame()}, tmp_path)

    listed = list_snapshots(tmp_path)
    assert run_id in listed

    loaded = load_snapshot(tmp_path, run_id)
    assert loaded["run_id"] == run_id
    assert loaded["run_config"]["region"] == "ERCOT"


def test_delete_snapshot_removes_run_directory(tmp_path: Path) -> None:
    """Delete should only remove existing run directories."""
    config = RunConfig(
        region="ERCOT",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 30),
        timeframe_preset="30D",
        use_rare=False,
        rare_path=None,
        model_choice="persistence",
        horizon_days=1,
    )
    run_id = save_snapshot(
        config,
        {"rows": 30},
        {"availability": {}, "summary": pd.DataFrame(), "predictions": pd.DataFrame()},
        tmp_path,
    )
    assert delete_snapshot(tmp_path, run_id) is True
    assert run_id not in list_snapshots(tmp_path)
