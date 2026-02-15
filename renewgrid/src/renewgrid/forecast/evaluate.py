"""Rolling-origin evaluation and report generation."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from renewgrid.forecast.baseline import persistence_predict_from_train
from renewgrid.forecast.prophet_model import fit_prophet, predict_prophet, prophet_is_available
from renewgrid.forecast.xgb_model import fit_xgb, predict_xgb, xgboost_is_available


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


def _select_split_indices(candidate_splits: list[int], max_splits: int | None) -> list[int]:
    """Select deterministic split indices with optional cap (most recent preferred)."""
    if max_splits is None or len(candidate_splits) <= max_splits:
        return candidate_splits
    return candidate_splits[-max_splits:]


def rolling_origin_evaluate(
    frame: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    horizons: Iterable[int] = (1, 2, 3),
    min_train_size: int = 60,
    max_splits: int | None = None,
    backtest_window_days: int | None = None,
    refit_every: int = 7,
    models: Iterable[str] = ("persistence", "prophet", "xgboost"),
) -> dict[str, object]:
    """Evaluate selected models on bounded rolling-origin splits for fast feedback."""
    data = frame.sort_values("date").reset_index(drop=True)
    if data.empty:
        return {
            "availability": {"persistence": False, "prophet": False, "xgboost": False},
            "predictions": pd.DataFrame(
                columns=["model", "horizon", "date", "y_true", "y_pred", "split_idx"]
            ),
            "summary": pd.DataFrame(
                columns=["model", "horizon", "mae", "rmse", "mape", "skill_vs_persistence"]
            ),
        }
    selected_models = set(models)
    available = {
        "persistence": "persistence" in selected_models,
        "prophet": "prophet" in selected_models and prophet_is_available(),
        "xgboost": "xgboost" in selected_models and xgboost_is_available(),
    }

    if max_splits is not None and max_splits < 1:
        raise ValueError("max_splits must be >= 1")
    if backtest_window_days is not None and backtest_window_days < 1:
        raise ValueError("backtest_window_days must be >= 1")
    if refit_every < 1:
        raise ValueError("refit_every must be >= 1")

    records: list[dict[str, object]] = []
    horizons_list = sorted(set(int(h) for h in horizons))
    if not horizons_list:
        raise ValueError("horizons cannot be empty")
    max_h = max(horizons_list)
    earliest_split = min_train_size
    latest_split = len(data) - max_h
    if latest_split < earliest_split:
        preds = pd.DataFrame(columns=["model", "horizon", "date", "y_true", "y_pred", "split_idx"])
        summary = pd.DataFrame(
            columns=["model", "horizon", "mae", "rmse", "mape", "skill_vs_persistence"]
        )
        return {"availability": available, "predictions": preds, "summary": summary}

    candidates = list(range(earliest_split, latest_split + 1))
    if backtest_window_days is not None:
        latest_date = pd.to_datetime(data["date"]).max()
        window_start = latest_date - pd.Timedelta(days=backtest_window_days - 1)
        window_candidates = [
            idx for idx in candidates if pd.Timestamp(data.loc[idx, "date"]) >= window_start
        ]
        if window_candidates:
            candidates = window_candidates
    split_indices = _select_split_indices(candidates, max_splits=max_splits)

    prophet_model: object | None = None
    xgb_model: object | None = None
    xgb_feature_cols: list[str] | None = None

    for split_counter, split_idx in enumerate(split_indices):
        train = data.iloc[:split_idx].copy()

        if available["prophet"] and (prophet_model is None or split_counter % refit_every == 0):
            prophet_model = fit_prophet(train[["date", target_col]], target_col)

        xgb_train_xy = train[feature_cols + [target_col]].dropna()
        if available["xgboost"] and (
            xgb_model is None or split_counter % refit_every == 0 or xgb_feature_cols != feature_cols
        ):
            if not xgb_train_xy.empty:
                xgb_model = fit_xgb(xgb_train_xy[feature_cols], xgb_train_xy[target_col])
                xgb_feature_cols = list(feature_cols)
            else:
                xgb_model = None

        prophet_future_dates: list[pd.Timestamp] = []
        prophet_meta: list[tuple[int, float, pd.Timestamp]] = []
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
                    "split_idx": split_idx,
                }
            )

            if available["prophet"] and prophet_model is not None:
                prophet_future_dates.append(eval_date)
                prophet_meta.append((horizon, truth, eval_date))

            if available["xgboost"] and xgb_model is not None:
                if pd.notna(test_row[feature_cols]).all():
                    pred_xgb = float(
                        predict_xgb(
                            xgb_model,
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
                            "split_idx": split_idx,
                        }
                    )

        if available["prophet"] and prophet_model is not None and prophet_future_dates:
            prophet_preds = predict_prophet(prophet_model, pd.Series(prophet_future_dates))
            for (horizon, truth, eval_date), pred in zip(prophet_meta, prophet_preds, strict=True):
                records.append(
                    {
                        "model": "prophet",
                        "horizon": horizon,
                        "date": eval_date,
                        "y_true": truth,
                        "y_pred": float(pred),
                        "split_idx": split_idx,
                    }
                )

    preds = pd.DataFrame.from_records(
        records,
        columns=["model", "horizon", "date", "y_true", "y_pred", "split_idx"],
    )
    summary_rows: list[dict[str, object]] = []
    for (model, horizon), group in preds.groupby(["model", "horizon"], dropna=False):
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

    summary = pd.DataFrame(
        summary_rows,
        columns=["model", "horizon", "mae", "rmse", "mape"],
    )
    if not summary.empty:
        persist_mae = (
            summary[summary["model"] == "persistence"].set_index("horizon")["mae"].to_dict()
        )
        summary["skill_vs_persistence"] = summary.apply(
            lambda row: (
                0.0
                if row["model"] == "persistence"
                else (persist_mae.get(int(row["horizon"]), np.nan) - row["mae"])
                / persist_mae.get(int(row["horizon"]), np.nan)
            ),
            axis=1,
        )
    else:
        summary["skill_vs_persistence"] = pd.Series(dtype=float)

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
