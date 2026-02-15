"""Prophet model wrapper."""

from __future__ import annotations

import pandas as pd


def prophet_is_available() -> bool:
    """Return True when prophet package is importable."""
    try:
        from prophet import Prophet  # noqa: F401
    except ImportError:
        return False
    return True


def predict_with_prophet(
    train_frame: pd.DataFrame,
    target_col: str,
    future_dates: pd.Series,
) -> pd.Series:
    """Train Prophet on train_frame and predict target values for future_dates."""
    if not prophet_is_available():
        raise RuntimeError("prophet package is not installed.")

    from prophet import Prophet

    model_frame = train_frame[["date", target_col]].rename(columns={"date": "ds", target_col: "y"})
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(model_frame)

    future = pd.DataFrame({"ds": pd.to_datetime(future_dates)})
    forecast = model.predict(future)
    return forecast["yhat"].astype(float).reset_index(drop=True)
