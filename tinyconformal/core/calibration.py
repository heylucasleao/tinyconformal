# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

"""Cross-validated inputs for conformal calibration."""

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict

from tinyconformal.core import conformal as core_conformal


@dataclass(frozen=True)
class CrossFittedCPSCalibration:
    """Out-of-fold inputs for a conditionally normalized CPS.

    Attributes
    ----------
    residuals : ndarray of shape (n_samples,)
        Signed location residuals ``y - y_hat``.
    scales : ndarray of shape (n_samples,)
        Strictly positive out-of-fold conditional scale predictions.
    standardized_residuals : ndarray of shape (n_samples,)
        Signed scores ``residuals / scales``.
    """

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
        """Return OOF absolute-residual scores for ICP.

        Parameters
        ----------
        learner : estimator
            Unfitted point estimator.
        X : array-like
            Features used for cross-validation.
        y : array-like of shape (n_samples,)
            Observed targets.
        cv : int or cross-validation splitter, default=5
            Cross-validation strategy.
        n_jobs : int or None, default=None
            Parallel jobs passed to ``cross_val_predict``.

        Returns
        -------
        ndarray of shape (n_samples,)
            Out-of-fold scores ``|y - y_hat|``.
        """
        predictions = cls._predictions(learner, X, y, cv=cv, n_jobs=n_jobs)
        if predictions.ndim != 1:
            raise ValueError(
                "ICP cross-validation predictions must be one-dimensional."
            )
        return core_conformal.absolute_residual_scores(y, predictions)

    @classmethod
    def cqr_scores(
        cls, learner: BaseEstimator, X, y, *, cv=5, n_jobs: int | None = None
    ) -> np.ndarray:
        """Return OOF CQR scores from lower and upper quantile predictions.

        ``learner.predict`` must return at least two columns; the first and last
        are interpreted as the lower and upper quantiles. ``cv`` and ``n_jobs``
        follow :meth:`icp_scores`.

        Returns
        -------
        ndarray of shape (n_samples,)
            Scores ``max(q_low - y, y - q_high)``.
        """
        predictions = cls._predictions(learner, X, y, cv=cv, n_jobs=n_jobs)
        if predictions.ndim != 2 or predictions.shape[1] < 2:
            raise ValueError(
                "CQR cross-validation predictions must have lower and upper "
                "quantile columns."
            )
        return core_conformal.cqr_scores(y, predictions[:, 0], predictions[:, -1])

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
        """Return cross-fitted residuals and locally standardized CPS scores.

        The location estimator first produces OOF predictions. Their absolute
        errors become targets for a second OOF fit of ``dispersion_learner``.
        Thus neither component of a standardized score is predicted by a model
        fitted on that score's observation.

        Parameters
        ----------
        learner : estimator
            Unfitted location estimator.
        dispersion_learner : estimator
            Unfitted estimator of positive conditional absolute-error scale.
        X : array-like
            Features used by both estimators.
        y : array-like of shape (n_samples,)
            Observed targets.
        cv : int or cross-validation splitter, default=5
            Cross-fitting strategy used in both stages.
        n_jobs : int or None, default=None
            Parallel jobs passed to ``cross_val_predict``.
        min_scale : float, default=1e-6
            Positive floor applied to dispersion training targets.

        Returns
        -------
        CrossFittedCPSCalibration
            Raw residuals, OOF scales, and standardized residuals.
        """
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
        """Return OOF probabilities from a binary classifier.

        Parameters are equivalent to :meth:`icp_scores`; ``learner`` must
        implement ``predict_proba`` and produce exactly two columns.

        Returns
        -------
        ndarray of shape (n_samples, 2)
            Out-of-fold class probabilities.
        """
        probabilities = cls._predictions(
            learner, X, y, cv=cv, n_jobs=n_jobs, method="predict_proba"
        )
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise ValueError(
                "Classification cross-validation must produce two probability columns."
            )
        return probabilities
