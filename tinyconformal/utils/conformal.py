"""Small, array-oriented building blocks shared by conformal estimators."""

import numpy as np


def validate_calibration_values(values, name: str) -> np.ndarray:
    """Return a non-empty finite one-dimensional calibration array."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    return values


def absolute_residual_scores(y_true, y_pred) -> np.ndarray:
    """Return the symmetric ICP nonconformity score ``|y - y_hat|``."""
    return np.abs(np.asarray(y_true) - np.asarray(y_pred))


def signed_forecast_residuals(y_true, y_pred) -> np.ndarray:
    """Return signed forecast residuals in the ``y_hat - y`` convention."""
    return np.asarray(y_pred) - np.asarray(y_true)


def cqr_scores(y_true, q_low, q_high) -> np.ndarray:
    """Return CQR scores ``max(q_low - y, y - q_high)``."""
    y_true = np.asarray(y_true)
    return np.maximum(np.asarray(q_low) - y_true, y_true - np.asarray(q_high))


def symmetric_bounds(predictions, correction) -> tuple[np.ndarray, np.ndarray]:
    """Expand point predictions symmetrically by a conformal correction."""
    predictions = np.asarray(predictions)
    correction = np.asarray(correction)
    return predictions - correction, predictions + correction


def signed_residual_bounds(
    predictions, low_residual_quantile, high_residual_quantile
) -> tuple[np.ndarray, np.ndarray]:
    """Invert signed ``y_hat - y`` residual quantiles into response bounds."""
    predictions = np.asarray(predictions)
    return (
        predictions - np.asarray(high_residual_quantile),
        predictions - np.asarray(low_residual_quantile),
    )


def cqr_bounds(q_low, q_high, correction) -> tuple[np.ndarray, np.ndarray]:
    """Expand base quantile bounds by a CQR correction."""
    correction = np.asarray(correction)
    return np.asarray(q_low) - correction, np.asarray(q_high) + correction
