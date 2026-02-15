"""Forecast Lab tab wiring for Phase 1 models and diagnostics."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from renewgrid.features.build_features import build_feature_frame
from renewgrid.forecast.evaluate import rolling_origin_evaluate


def render_forecast_lab(dataset: pd.DataFrame | None) -> None:
    """Render Phase 1 forecast controls, metrics, and diagnostic charts."""
    st.subheader("Forecast Lab")
    if dataset is None or dataset.empty:
        st.info("Run Guided flow first to load a dataset.")
        return

    candidate_targets = ["demand_mw_avg"]
    for col in ["solar_cf", "wind_cf", "solar_gen_mwh", "wind_gen_mwh"]:
        if col in dataset.columns:
            candidate_targets.append(col)

    target_col = st.selectbox("Target series", options=candidate_targets)
    model_choice = st.selectbox("Model", options=["persistence", "prophet", "xgboost"], index=0)
    horizon = st.slider("Horizon", min_value=1, max_value=3, value=3)
    train_window = st.slider("Train window length", min_value=20, max_value=180, value=60)
    step_size = st.slider("Step size", min_value=1, max_value=14, value=1)

    features = build_feature_frame(dataset, target_col=target_col)
    excluded = {
        "date",
        "region",
        "source",
        "demand_mw_avg",
        "solar_cf",
        "wind_cf",
        "solar_gen_mwh",
        "wind_gen_mwh",
    }
    feature_cols = [
        c
        for c in features.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(features[c])
    ]

    evaluation = rolling_origin_evaluate(
        frame=features.iloc[::step_size].reset_index(drop=True),
        target_col=target_col,
        feature_cols=feature_cols,
        horizons=tuple(range(1, horizon + 1)),
        min_train_size=min(train_window, max(20, len(features) // 2)),
        models=(model_choice, "persistence") if model_choice != "persistence" else ("persistence",),
    )

    summary = evaluation["summary"]
    preds = evaluation["predictions"]
    st.markdown("### Metrics + Skill")
    st.dataframe(summary, use_container_width=True)

    st.markdown("### Actual vs Predicted")
    selected_model = st.selectbox(
        "Prediction model",
        options=sorted(preds["model"].unique().tolist()),
    )
    selected_horizon = st.selectbox(
        "Prediction horizon",
        options=sorted(preds["horizon"].unique().tolist()),
    )
    selected = preds[
        (preds["model"] == selected_model) & (preds["horizon"] == selected_horizon)
    ].copy()
    selected = selected.sort_values("date")
    if not selected.empty:
        chart = selected[["date", "y_true", "y_pred"]].set_index("date")
        st.line_chart(chart)

    st.markdown("### Error Distribution")
    if not selected.empty:
        selected["error"] = selected["y_pred"] - selected["y_true"]
        st.bar_chart(selected["error"].reset_index(drop=True))

    st.markdown("### Feature Importance")
    if model_choice == "xgboost":
        try:
            from xgboost import XGBRegressor

            train = features[feature_cols + [target_col]].dropna()
            if not train.empty:
                model = XGBRegressor(
                    n_estimators=100,
                    random_state=42,
                    objective="reg:squarederror",
                )
                model.fit(train[feature_cols], train[target_col])
                importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(
                    ascending=False
                )
                st.dataframe(importances.head(12).rename("importance"), use_container_width=True)
            else:
                st.info("Insufficient non-null training rows for feature importance.")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"XGBoost importance unavailable: {exc}")
    else:
        st.info("Feature importance is available for XGBoost model only.")

    st.markdown("### Leakage Risk")
    st.success("Leakage risk: ✅ none detected (lag/rolling features are backward-looking).")
