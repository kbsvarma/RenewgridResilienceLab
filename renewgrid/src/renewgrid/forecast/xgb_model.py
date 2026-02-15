"""XGBoost model wrapper."""

from __future__ import annotations

import pandas as pd


def xgboost_is_available() -> bool:
    """Return True when xgboost package is importable."""
    try:
        from xgboost import XGBRegressor  # noqa: F401
    except ImportError:
        return False
    return True


def fit_xgb(train_features: pd.DataFrame, train_target: pd.Series) -> object:
    """Fit XGBoost regressor on provided training set."""
    if not xgboost_is_available():
        raise RuntimeError("xgboost package is not installed.")

    from xgboost import XGBRegressor

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        objective="reg:squarederror",
    )
    model.fit(train_features, train_target)
    return model


def predict_xgb(model: object, predict_features: pd.DataFrame) -> pd.Series:
    """Predict values with a fitted XGBoost model."""
    preds = model.predict(predict_features)
    return pd.Series(preds, index=predict_features.index, dtype=float)


def predict_with_xgb(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    predict_features: pd.DataFrame,
) -> pd.Series:
    """Fit XGBoost regressor and predict values for predict_features."""
    model = fit_xgb(train_features, train_target)
    return predict_xgb(model, predict_features)
