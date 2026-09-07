# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

"""Cross-fitted conformal predictive-system estimator."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted

from tinyconformal.core.calibration import CrossValidationCalibration

from ..base import PredictiveDistribution
from .distribution import (
    ContinuousConformalDistribution,
    DiscreteConformalDistribution,
    _as_1d_finite,
    _as_positive_scales,
)


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
    calibration_ : CrossFittedCPSCalibration
        Encapsulated out-of-fold residuals, scales, and standardized residuals.
    n_calibration_ : int
        Number of cross-fitted calibration scores.
    """

    def __init__(
        self,
        learner: BaseEstimator,
        dispersion_learner: BaseEstimator,
        discrete: bool = False,
        minimum: int | None = 0,
    ):
        """Configure the location model, scale model, and cross-fitting policy."""
        self.learner = learner
        self.dispersion_learner = dispersion_learner
        self.discrete = discrete
        self.minimum = minimum

    def fit(self, X, y, cv=5, n_jobs: int | None = None):
        """Cross-fit CPS scores and refit both learners on all observations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features used by both estimators.
        y : array-like of shape (n_samples,)
            Observed targets.
        cv : int or cross-validation splitter, default=5
            Cross-fitting strategy used for location and scale predictions.
        n_jobs : int or None, default=None
            Parallel jobs passed to cross-validated prediction.

        Returns
        -------
        self
            Fitted predictive system containing out-of-fold standardized
            residuals and final location and dispersion learners.

        Raises
        ------
        ValueError
            If targets are non-finite, violate the configured discrete support,
            or cross-fitted predictions have invalid shapes or values.

        Notes
        -----
        Location predictions are generated out of fold. Absolute location
        residuals become scale targets, and scale predictions are also generated
        out of fold before standardization. Finally, both learner templates are
        cloned and fitted on all observations. ``cv`` and ``n_jobs`` describe
        this fit and are stored as ``cv_`` and ``n_jobs_``.
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
            cv=cv,
            n_jobs=n_jobs,
        )
        self.cv_ = cv
        self.n_jobs_ = n_jobs
        self.calibration_ = calibration
        self.n_calibration_ = calibration.standardized_residuals.size

        self.learner_ = clone(self.learner).fit(X, y)
        scale_targets = np.maximum(np.abs(calibration.residuals), 1e-6)
        self.dispersion_learner_ = clone(self.dispersion_learner).fit(X, scale_targets)
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
            Batch containing one distribution per row of ``X``. Its ``cdf`` and
            ``ppf`` methods accept scalar, common-grid, or row-wise inputs and
            return NumPy arrays whose first dimension follows the order of
            ``X``. The result also exposes ``interval`` and
            ``evaluate``. Discrete systems additionally expose ``pmf`` and
            return integer quantiles.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the predictive system has not been fitted.
        ValueError
            If either estimator returns non-finite predictions, the dispersion
            estimator returns a non-positive scale, or the location and scale
            predictions have different shapes.

        Notes
        -----
        The returned batch is positionally aligned with ``X``. Reordering one
        without the other invalidates that correspondence.
        """
        check_is_fitted(self, attributes=["calibration_", "n_calibration_"])
        locations = _as_1d_finite(self.learner_.predict(X), "learner predictions")
        scales = _as_positive_scales(
            self.dispersion_learner_.predict(X), "dispersion learner predictions"
        )
        if scales.shape != locations.shape:
            raise ValueError("Scale and location predictions must have the same shape.")
        if self.discrete:
            return DiscreteConformalDistribution(
                locations,
                self.calibration_.standardized_residuals,
                scales=scales,
                minimum=self.minimum,
            )
        return ContinuousConformalDistribution(
            locations, self.calibration_.standardized_residuals, scales=scales
        )
