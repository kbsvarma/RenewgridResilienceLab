"""Forecast evaluation metrics."""

from __future__ import annotations

import numpy as np


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAE for two equally sized vectors."""
    return float(np.mean(np.abs(y_true - y_pred)))
