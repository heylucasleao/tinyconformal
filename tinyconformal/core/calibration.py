# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

"""Cross-validated inputs for conformal calibration."""

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict

from tinyconformal.core.conformal import (
    absolute_residual_scores,
    cqr_scores,
)


@dataclass(frozen=True)
class CrossFittedCPSCalibration:
    """Out-of-fold location errors, scales, and standardized CPS residuals."""

    residuals: np.ndarray
    scales: np.ndarray
    standardized_residuals: np.ndarray


class CrossValidationCalibration:
    """Generate out-of-fold calibration scores from labeled historical data.

    The returned arrays can calibrate an already-fitted conformal estimator
    without reserving a separate calibration split. Predictions are always
    out-of-fold, so no observation is scored by a model fitted on that same
    observation.
    """

    @staticmethod
    def _predictions(
        learner: BaseEstimator,
        X,
        y,
        *,
        cv=5,
        n_jobs: int | None = None,
        method: str = "predict",
    ) -> np.ndarray:
        predictions = np.asarray(
            cross_val_predict(learner, X, y, cv=cv, n_jobs=n_jobs, method=method),
            dtype=float,
        )
        if predictions.size == 0 or not np.all(np.isfinite(predictions)):
            raise ValueError(
                "Cross-validation predictions must be non-empty and finite."
            )
        return predictions

    @classmethod
    def icp_scores(
        cls, learner: BaseEstimator, X, y, *, cv=5, n_jobs: int | None = None
    ) -> np.ndarray:
        """Return OOF absolute-residual scores for ICP."""
        predictions = cls._predictions(learner, X, y, cv=cv, n_jobs=n_jobs)
        if predictions.ndim != 1:
            raise ValueError(
                "ICP cross-validation predictions must be one-dimensional."
            )
        return absolute_residual_scores(y, predictions)

    @classmethod
    def cqr_scores(
        cls, learner: BaseEstimator, X, y, *, cv=5, n_jobs: int | None = None
    ) -> np.ndarray:
        """Return OOF CQR scores from lower and upper quantile predictions."""
        predictions = cls._predictions(learner, X, y, cv=cv, n_jobs=n_jobs)
        if predictions.ndim != 2 or predictions.shape[1] < 2:
            raise ValueError(
                "CQR cross-validation predictions must have lower and upper "
                "quantile columns."
            )
        return cqr_scores(y, predictions[:, 0], predictions[:, -1])

    @classmethod
    def cps_scores(
        cls,
        learner: BaseEstimator,
        dispersion_learner: BaseEstimator,
        X,
        y,
        *,
        cv=5,
        n_jobs: int | None = None,
        min_scale: float = 1e-6,
    ) -> CrossFittedCPSCalibration:
        """Return cross-fitted residuals and locally standardized CPS scores."""
        if not isinstance(min_scale, (int, float)) or min_scale <= 0:
            raise ValueError("min_scale must be strictly positive.")
        predictions = cls._predictions(learner, X, y, cv=cv, n_jobs=n_jobs)
        if predictions.ndim != 1:
            raise ValueError(
                "CPS cross-validation predictions must be one-dimensional."
            )
        residuals = np.asarray(y, dtype=float) - predictions
        scale_targets = np.maximum(np.abs(residuals), float(min_scale))
        scales = cls._predictions(
            dispersion_learner, X, scale_targets, cv=cv, n_jobs=n_jobs
        )
        if scales.ndim != 1:
            raise ValueError("CPS scale predictions must be one-dimensional.")
        if np.any(scales <= 0.0):
            raise ValueError("CPS scale predictions must be strictly positive.")
        return CrossFittedCPSCalibration(
            residuals=residuals,
            scales=scales,
            standardized_residuals=residuals / scales,
        )

    @classmethod
    def classification_probabilities(
        cls, learner: BaseEstimator, X, y, *, cv=5, n_jobs: int | None = None
    ) -> np.ndarray:
        """Return OOF class probabilities for conformal classification."""
        probabilities = cls._predictions(
            learner, X, y, cv=cv, n_jobs=n_jobs, method="predict_proba"
        )
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise ValueError(
                "Classification cross-validation must produce two probability columns."
            )
        return probabilities
