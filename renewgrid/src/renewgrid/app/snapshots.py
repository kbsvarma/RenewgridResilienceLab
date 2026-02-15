"""Snapshot persistence helpers for run comparison notebook."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from renewgrid.app.state import RunConfig
from renewgrid.utils.parquet import require_parquet_engine


def _serialize_eval(eval_results: dict[str, object]) -> dict[str, object]:
    summary = eval_results.get("summary")
    predictions = eval_results.get("predictions")
    if isinstance(summary, pd.DataFrame):
        summary = summary.copy()
        for col in summary.columns:
            if pd.api.types.is_datetime64_any_dtype(summary[col]):
                summary[col] = summary[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(predictions, pd.DataFrame):
        predictions = predictions.copy()
        for col in predictions.columns:
            if pd.api.types.is_datetime64_any_dtype(predictions[col]):
                predictions[col] = predictions[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "availability": eval_results.get("availability", {}),
        "summary": summary.to_dict(orient="records") if isinstance(summary, pd.DataFrame) else [],
        "predictions": (
            predictions.to_dict(orient="records") if isinstance(predictions, pd.DataFrame) else []
        ),
    }


def save_snapshot(
    run_config: RunConfig,
    dataset_summary: dict[str, object],
    eval_results: dict[str, object],
    base_dir: str | Path,
    key_series: pd.DataFrame | None = None,
) -> str:
    """Save run snapshot artifacts and return run_id."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(base_dir) / "reports" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "run_config.json"
    summary_path = run_dir / "dataset_summary.json"
    eval_path = run_dir / "eval_results.json"

    config_path.write_text(json.dumps(run_config.to_dict(), indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")
    eval_path.write_text(json.dumps(_serialize_eval(eval_results), indent=2), encoding="utf-8")

    files = [config_path.name, summary_path.name, eval_path.name]
    if isinstance(key_series, pd.DataFrame) and not key_series.empty:
        require_parquet_engine()
        series_path = run_dir / "key_series.parquet"
        key_series.to_parquet(series_path, index=False)
        files.append(series_path.name)

    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return run_id


def list_snapshots(base_dir: str | Path) -> list[str]:
    """List run IDs under reports/runs sorted newest-first."""
    root = Path(base_dir) / "reports" / "runs"
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()], reverse=True)


def load_snapshot(base_dir: str | Path, run_id: str) -> dict[str, object]:
    """Load snapshot JSON artifacts for a given run_id."""
    run_dir = Path(base_dir) / "reports" / "runs" / run_id
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    dataset_summary = json.loads((run_dir / "dataset_summary.json").read_text(encoding="utf-8"))
    eval_results = json.loads((run_dir / "eval_results.json").read_text(encoding="utf-8"))

    key_series = None
    series_path = run_dir / "key_series.parquet"
    if series_path.exists():
        key_series = pd.read_parquet(series_path)

    return {
        "run_id": run_id,
        "run_config": config,
        "dataset_summary": dataset_summary,
        "eval_results": eval_results,
        "key_series": key_series,
    }


def delete_snapshot(base_dir: str | Path, run_id: str) -> bool:
    """Delete a snapshot directory under reports/runs and return success."""
    run_dir = Path(base_dir) / "reports" / "runs" / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        return False
    shutil.rmtree(run_dir)
    return True
