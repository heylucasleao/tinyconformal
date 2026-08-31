# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

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


def validate_probabilities(probabilities, name: str = "probabilities") -> np.ndarray:
    """Return a finite two-dimensional probability matrix."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError(f"{name} must contain values in [0, 1].")
    return probabilities


def class_indices(labels, classes) -> np.ndarray:
    """Map arbitrary class labels to their probability-column positions."""
    labels = np.asarray(labels)
    classes = np.asarray(classes)
    mapping = {label: index for index, label in enumerate(classes)}
    unknown = sorted(set(labels) - set(mapping), key=str)
    if unknown:
        raise ValueError(f"Unknown class labels: {unknown}")
    return np.asarray([mapping[label] for label in labels], dtype=int)


def probability_scores(probabilities) -> np.ndarray:
    """Return classification nonconformity scores ``1 - probability``."""
    return 1.0 - validate_probabilities(probabilities)


def true_class_probability_scores(probabilities, labels, classes) -> np.ndarray:
    """Return one probability nonconformity score for each observed label."""
    probabilities = validate_probabilities(probabilities)
    indices = class_indices(labels, classes)
    if probabilities.shape[0] != indices.size:
        raise ValueError("probabilities and labels must have the same number of rows.")
    return probability_scores(probabilities)[np.arange(indices.size), indices]


def threshold_prediction_set(scores, thresholds) -> np.ndarray:
    """Include classes whose nonconformity score does not exceed its threshold."""
    return (np.asarray(scores) <= np.asarray(thresholds)).astype(int)


def conformal_p_values(calibration_scores, test_scores) -> np.ndarray:
    """Compute smoothed conformal p-values for a matrix of test scores."""
    calibration_scores = validate_calibration_values(
        calibration_scores, "calibration_scores"
    )
    test_scores = np.asarray(test_scores, dtype=float)
    if not np.all(np.isfinite(test_scores)):
        raise ValueError("test_scores must contain only finite values.")
    counts = np.sum(
        calibration_scores.reshape((-1,) + (1,) * test_scores.ndim) >= test_scores,
        axis=0,
    )
    return (counts + 1.0) / (calibration_scores.size + 1.0)


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
