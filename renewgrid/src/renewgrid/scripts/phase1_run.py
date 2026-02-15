"""Phase 1 CLI: build daily datasets and evaluate forecasts."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from renewgrid.config import load_environment
from renewgrid.data.merge import build_daily_dataset
from renewgrid.features.build_features import build_feature_frame
from renewgrid.forecast.evaluate import rolling_origin_evaluate, save_evaluation_report


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Phase 1 run."""
    parser = argparse.ArgumentParser(description="Run RenewGrid Phase 1 pipeline")
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=180, help="Fallback lookback window in days")
    parser.add_argument("--rare-path", type=str, default=None, help="Optional RARE parquet path")
    parser.add_argument("--base-dir", type=str, default=".", help="Project base directory")
    return parser.parse_args()


def _resolve_dates(start: str | None, end: str | None, days: int) -> tuple[date, date]:
    if start and end:
        return date.fromisoformat(start), date.fromisoformat(end)
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=days - 1)
    return start_date, end_date


def _targets_for_dataset(dataset: pd.DataFrame) -> list[str]:
    """Return forecast targets available in the assembled dataset."""
    targets = ["demand_mw_avg"]
    if {"solar_gen_mwh", "wind_gen_mwh"}.issubset(set(dataset.columns)):
        targets.extend(["solar_gen_mwh", "wind_gen_mwh"])
    elif {"solar_gen", "wind_gen"}.issubset(set(dataset.columns)):
        targets.extend(["solar_gen", "wind_gen"])
    elif {"solar_cf", "wind_cf"}.issubset(set(dataset.columns)):
        targets.extend(["solar_cf", "wind_cf"])
    return targets


def run_phase1(
    base_dir: str | Path,
    start_date: date,
    end_date: date,
    rare_path: str | None,
) -> dict[str, dict[str, dict[str, Path]]]:
    """Run dataset build and evaluation for CAISO and ERCOT."""
    results: dict[str, dict[str, dict[str, Path]]] = {}
    reports_dir = Path(base_dir) / "reports" / "phase1"
    non_feature_targets = {
        "demand_mw_avg",
        "solar_gen",
        "wind_gen",
        "solar_gen_mwh",
        "wind_gen_mwh",
        "solar_cf",
        "wind_cf",
    }

    for region in ("CAISO", "ERCOT"):
        dataset = build_daily_dataset(
            region=region,
            start_date=start_date,
            end_date=end_date,
            base_dir=base_dir,
            rare_path=rare_path,
        )
        region_reports: dict[str, dict[str, Path]] = {}

        for target_col in _targets_for_dataset(dataset):
            feature_frame = build_feature_frame(dataset, target_col=target_col)
            exclude = {"date", "region", "source", *non_feature_targets}
            feature_cols = [
                col
                for col in feature_frame.columns
                if col not in exclude and pd.api.types.is_numeric_dtype(feature_frame[col])
            ]

            evaluation = rolling_origin_evaluate(
                frame=feature_frame,
                target_col=target_col,
                feature_cols=feature_cols,
                min_train_size=max(30, min(60, max(10, len(feature_frame) // 3))),
            )
            region_reports[target_col] = save_evaluation_report(
                evaluation=evaluation,
                region=region,
                target_col=target_col,
                output_dir=reports_dir,
            )

        results[region] = region_reports

    return results


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    base_dir = Path(args.base_dir)
    load_environment(base_dir / ".env")
    start_date, end_date = _resolve_dates(args.start, args.end, args.days)
    outputs = run_phase1(base_dir, start_date, end_date, args.rare_path)
    for region, targets in outputs.items():
        for target_col, paths in targets.items():
            print(f"{region} {target_col} report JSON: {paths['json']}")
            print(f"{region} {target_col} report Markdown: {paths['markdown']}")


if __name__ == "__main__":
    main()
