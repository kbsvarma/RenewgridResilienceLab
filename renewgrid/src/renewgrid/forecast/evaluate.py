"""Rolling-origin evaluation and report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from renewgrid.forecast.baseline import persistence_predict_from_train
from renewgrid.forecast.prophet_model import predict_with_prophet, prophet_is_available
from renewgrid.forecast.xgb_model import predict_with_xgb, xgboost_is_available


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAE for two equally sized vectors."""
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RMSE for two equally sized vectors."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAPE for non-zero targets only."""
    mask = y_true != 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }


def rolling_origin_evaluate(
    frame: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    horizons: Iterable[int] = (1, 2, 3),
    min_train_size: int = 60,
    models: Iterable[str] = ("persistence", "prophet", "xgboost"),
) -> dict[str, object]:
    """Evaluate persistence, Prophet, and XGBoost on rolling-origin splits."""
    data = frame.sort_values("date").reset_index(drop=True)
    selected_models = set(models)
    available = {
        "persistence": "persistence" in selected_models,
        "prophet": "prophet" in selected_models and prophet_is_available(),
        "xgboost": "xgboost" in selected_models and xgboost_is_available(),
    }

    records: list[dict[str, object]] = []
    horizons_list = list(horizons)
    max_h = max(horizons_list)

    for split_idx in range(min_train_size, len(data) - max_h + 1):
        train = data.iloc[:split_idx].copy()
        for horizon in horizons_list:
            test_idx = split_idx + horizon - 1
            test_row = data.iloc[test_idx]
            truth = float(test_row[target_col])
            eval_date = pd.Timestamp(test_row["date"])

            pred_persist = persistence_predict_from_train(train, target_col, horizon)
            records.append(
                {
                    "model": "persistence",
                    "horizon": horizon,
                    "date": eval_date,
                    "y_true": truth,
                    "y_pred": pred_persist,
                }
            )

            if available["prophet"]:
                pred_prophet = float(
                    predict_with_prophet(train[["date", target_col]], target_col, pd.Series([eval_date])).iloc[0]
                )
                records.append(
                    {
                        "model": "prophet",
                        "horizon": horizon,
                        "date": eval_date,
                        "y_true": truth,
                        "y_pred": pred_prophet,
                    }
                )

            if available["xgboost"]:
                train_xy = train[feature_cols + [target_col]].dropna()
                if not train_xy.empty and pd.notna(test_row[feature_cols]).all():
                    pred_xgb = float(
                        predict_with_xgb(
                            train_xy[feature_cols],
                            train_xy[target_col],
                            pd.DataFrame([test_row[feature_cols]], columns=feature_cols),
                        ).iloc[0]
                    )
                    records.append(
                        {
                            "model": "xgboost",
                            "horizon": horizon,
                            "date": eval_date,
                            "y_true": truth,
                            "y_pred": pred_xgb,
                        }
                    )

    preds = pd.DataFrame.from_records(records)
    summary_rows: list[dict[str, object]] = []
    for (model, horizon), group in preds.groupby(["model", "horizon"]):
        y_true = group["y_true"].to_numpy(dtype=float)
        y_pred = group["y_pred"].to_numpy(dtype=float)
        metrics = _compute_metrics(y_true, y_pred)
        summary_rows.append(
            {
                "model": model,
                "horizon": int(horizon),
                **metrics,
            }
        )

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        persist_mae = summary[summary["model"] == "persistence"].set_index("horizon")["mae"].to_dict()
        summary["skill_vs_persistence"] = summary.apply(
            lambda row: (
                0.0
                if row["model"] == "persistence"
                else (persist_mae.get(int(row["horizon"]), np.nan) - row["mae"])
                / persist_mae.get(int(row["horizon"]), np.nan)
            ),
            axis=1,
        )

    return {
        "availability": available,
        "predictions": preds,
        "summary": summary,
    }


def save_evaluation_report(
    evaluation: dict[str, object],
    region: str,
    target_col: str,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save JSON and Markdown report artifacts under reports/phase1."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = evaluation["summary"]
    if not isinstance(summary, pd.DataFrame):
        raise TypeError("evaluation['summary'] must be a pandas DataFrame")

    json_path = out_dir / f"{region.lower()}_{target_col}_report.json"
    md_path = out_dir / f"{region.lower()}_{target_col}_report.md"

    payload = {
        "region": region,
        "target": target_col,
        "availability": evaluation["availability"],
        "summary": summary.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        f"# Phase 1 Forecast Report: {region} ({target_col})",
        "",
        "## Model Availability",
        f"- persistence: {evaluation['availability']['persistence']}",
        f"- prophet: {evaluation['availability']['prophet']}",
        f"- xgboost: {evaluation['availability']['xgboost']}",
        "",
        "## Metrics",
        "",
        "| model | horizon | mae | rmse | mape | skill_vs_persistence |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"| {row['model']} | {int(row['horizon'])} | {row['mae']:.4f} | "
            f"{row['rmse']:.4f} | {row['mape']:.4f} | {row['skill_vs_persistence']:.4f} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    return {"json": json_path, "markdown": md_path}
