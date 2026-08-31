# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

"""Cross-validated inputs for conformal calibration."""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict

from tinyconformal.core.conformal import (
    absolute_residual_scores,
    cqr_scores,
)


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
    def cps_residuals(
        cls, learner: BaseEstimator, X, y, *, cv=5, n_jobs: int | None = None
    ) -> np.ndarray:
        """Return OOF signed residuals ``y - y_hat`` for a regression CPS."""
        predictions = cls._predictions(learner, X, y, cv=cv, n_jobs=n_jobs)
        if predictions.ndim != 1:
            raise ValueError(
                "CPS cross-validation predictions must be one-dimensional."
            )
        return np.asarray(y, dtype=float) - predictions

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
