# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted

from tinyconformal.core.calibration import CrossValidationCalibration

from .base import (
    DiscretePredictiveDistribution,
    EmpiricalResidualDistribution,
    PredictiveDistribution,
)


def _as_1d_finite(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _as_positive_scales(values, name: str = "scales") -> np.ndarray:
    scales = _as_1d_finite(values, name)
    if np.any(scales <= 0.0):
        raise ValueError(f"{name} must contain only strictly positive values.")
    return scales


class _ResidualPredictiveDistribution(EmpiricalResidualDistribution):
    """Empirical signed-residual distributions shifted by point predictions."""

    def __init__(
        self,
        locations: np.ndarray,
        residuals: np.ndarray,
        scales: np.ndarray | None = None,
    ):
        self.locations = _as_1d_finite(locations, "locations")
        self.residuals = np.sort(_as_1d_finite(residuals, "residuals"))
        self.scales = (
            np.ones_like(self.locations)
            if scales is None
            else _as_positive_scales(scales)
        )
        if self.scales.shape != self.locations.shape:
            raise ValueError("scales and locations must have the same shape.")

    def __len__(self) -> int:
        return self.locations.size

    @property
    def n_calibration(self) -> int:
        return self.residuals.size

    def _row_residuals(self) -> np.ndarray:
        return self.scales[:, None] * self.residuals[None, :]


class ContinuousConformalDistribution(_ResidualPredictiveDistribution):
    """Batch of continuous cross-fitted conformal predictive distributions."""


class DiscreteConformalDistribution(
    _ResidualPredictiveDistribution, DiscretePredictiveDistribution
):
    """Batch of cross-fitted conformal distributions for integer counts.

    Parameters
    ----------
    locations : ndarray of shape (n_predictions,)
        Point predictions defining the location of each distribution.
    residuals : ndarray of shape (n_calibration,)
        Signed standardized out-of-fold calibration residuals.
    scales : ndarray of shape (n_predictions,), optional
        Positive conditional scale for each prediction. Defaults to one.
    minimum : int or None, default=0
        Lower boundary of the integer support. ``None`` allows all integers.
    """

    def __init__(
        self,
        locations: np.ndarray,
        residuals: np.ndarray,
        scales: np.ndarray | None = None,
        minimum: int | None = 0,
    ):
        super().__init__(locations, residuals, scales=scales)
        if minimum is not None and not isinstance(minimum, (int, np.integer)):
            raise TypeError("minimum must be an integer or None.")
        self.minimum = None if minimum is None else int(minimum)

    def ppf(self, quantiles):
        result = np.ceil(super().ppf(quantiles))
        if self.minimum is not None:
            result = np.maximum(result, self.minimum)
        return result.astype(int)

    def cdf(self, values):
        values = np.floor(np.asarray(values, dtype=float))
        result = super().cdf(values)
        if self.minimum is None:
            return result
        below = values < self.minimum
        if np.ndim(values) == 0:
            return np.zeros_like(result) if bool(below) else result
        if result.ndim == 1:
            return np.where(np.ravel(below), 0.0, result)
        return np.where(np.broadcast_to(below, np.shape(result)), 0.0, result)


class CrossConformalPredictiveSystem(BaseEstimator):
    """Build locally scaled predictive distributions from cross-fitted scores.

    The system stores signed standardized calibration residuals
    ``(y - y_hat) / sigma_hat``. At prediction time their empirical distribution
    is shifted by the point prediction and multiplied by the conditional scale.
    Quantiles use the finite-sample ``ceil((n + 1) q)`` conformal rank.

    Parameters
    ----------
    learner : estimator
        An unfitted location estimator implementing ``fit`` and ``predict``.
    dispersion_learner : estimator
        An unfitted estimator whose prediction is a positive conditional scale.
    cv : int or cross-validation splitter, default=5
        Cross-fitting strategy used for both location and scale predictions.
    n_jobs : int or None, default=None
        Parallel jobs passed to cross-validated prediction.
    discrete : bool, default=False
        If true, produce an ordered integer distribution with ``pmf`` support.
    minimum : int or None, default=0
        Lower support bound for discrete outcomes. Ignored for continuous outcomes.

    Attributes
    ----------
    learner_ : estimator
        Location estimator refitted on all observations.
    dispersion_learner_ : estimator
        Dispersion estimator refitted on all absolute OOF location errors.
    residuals_ : ndarray of shape (n_samples,)
        Signed out-of-fold location residuals ``y - y_hat``.
    scales_ : ndarray of shape (n_samples,)
        Out-of-fold conditional scale predictions.
    standardized_residuals_ : ndarray of shape (n_samples,)
        Cross-fitted residuals ``residuals_ / scales_``.
    n_calibration_ : int
        Number of cross-fitted calibration scores.
    """

    def __init__(
        self,
        learner: BaseEstimator,
        dispersion_learner: BaseEstimator,
        cv=5,
        n_jobs: int | None = None,
        discrete: bool = False,
        minimum: int | None = 0,
    ):
        self.learner = learner
        self.dispersion_learner = dispersion_learner
        self.cv = cv
        self.n_jobs = n_jobs
        self.discrete = discrete
        self.minimum = minimum

    def fit(self, X, y):
        """Cross-fit standardized scores, then refit both models on all data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features used by both estimators.
        y : array-like of shape (n_samples,)
            Observed targets.

        Returns
        -------
        self
            Fitted predictive system.
        """
        y = _as_1d_finite(y, "y")
        if self.discrete and np.any(y != np.floor(y)):
            raise ValueError("Discrete CPS targets must be integer-valued.")
        if self.discrete and self.minimum is not None and np.any(y < self.minimum):
            raise ValueError(f"Discrete CPS targets must be >= {self.minimum}.")

        calibration = CrossValidationCalibration.cps_scores(
            self.learner,
            self.dispersion_learner,
            X,
            y,
            cv=self.cv,
            n_jobs=self.n_jobs,
        )
        self.residuals_ = calibration.residuals
        self.scales_ = calibration.scales
        self.standardized_residuals_ = calibration.standardized_residuals
        self.n_calibration_ = self.standardized_residuals_.size

        self.learner_ = clone(self.learner).fit(X, y)
        scale_targets = np.maximum(np.abs(self.residuals_), 1e-6)
        self.dispersion_learner_ = clone(self.dispersion_learner).fit(
            X, scale_targets
        )
        return self

    def predict_distribution(self, X) -> PredictiveDistribution:
        """Return one calibrated predictive distribution per input row.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features used by the refitted location and dispersion estimators.

        Returns
        -------
        PredictiveDistribution
            Row-aligned continuous or discrete predictive distributions.
        """
        check_is_fitted(
            self, attributes=["standardized_residuals_", "n_calibration_"]
        )
        locations = _as_1d_finite(self.learner_.predict(X), "learner predictions")
        scales = _as_positive_scales(
            self.dispersion_learner_.predict(X), "dispersion learner predictions"
        )
        if scales.shape != locations.shape:
            raise ValueError("Scale and location predictions must have the same shape.")
        if self.discrete:
            return DiscreteConformalDistribution(
                locations,
                self.standardized_residuals_,
                scales=scales,
                minimum=self.minimum,
            )
        return ContinuousConformalDistribution(
            locations, self.standardized_residuals_, scales=scales
        )

class ContinuousCrossConformalPredictiveSystem(CrossConformalPredictiveSystem):
    """Cross-fitted predictive system for continuous targets.

    Parameters
    ----------
    learner : estimator
        Unfitted location estimator implementing ``fit`` and ``predict``.
    dispersion_learner : estimator
        Unfitted estimator producing strictly positive conditional scales.
    cv : int or cross-validation splitter, default=5
        Cross-fitting strategy for location and dispersion predictions.
    n_jobs : int or None, default=None
        Parallel jobs passed to scikit-learn cross-validation.
    """

    def __init__(self, learner, dispersion_learner, cv=5, n_jobs=None):
        super().__init__(
            learner=learner,
            dispersion_learner=dispersion_learner,
            cv=cv,
            n_jobs=n_jobs,
            discrete=False,
            minimum=None,
        )


class DiscreteCrossConformalPredictiveSystem(CrossConformalPredictiveSystem):
    """Cross-fitted predictive system for ordered integer outcomes.

    Parameters
    ----------
    learner : estimator
        Unfitted location estimator implementing ``fit`` and ``predict``.
    dispersion_learner : estimator
        Unfitted estimator producing strictly positive conditional scales.
    cv : int or cross-validation splitter, default=5
        Cross-fitting strategy for location and dispersion predictions.
    n_jobs : int or None, default=None
        Parallel jobs passed to scikit-learn cross-validation.
    minimum : int or None, default=0
        Lower boundary of the integer support. Use ``0`` for counts, ``1`` for
        strictly positive outcomes, another integer for a known lower bound,
        or ``None`` when negative integers are valid.
    """

    def __init__(
        self,
        learner: BaseEstimator,
        dispersion_learner: BaseEstimator,
        cv=5,
        n_jobs=None,
        minimum: int | None = 0,
    ):
        super().__init__(
            learner=learner,
            dispersion_learner=dispersion_learner,
            cv=cv,
            n_jobs=n_jobs,
            discrete=True,
            minimum=minimum,
        )
