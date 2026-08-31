# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from .base import DiscretePredictiveDistribution, PredictiveDistribution


def _as_1d_finite(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


class _ResidualPredictiveDistribution(PredictiveDistribution):
    """Empirical signed-residual distributions shifted by point predictions."""

    def __init__(self, locations: np.ndarray, residuals: np.ndarray):
        self.locations = _as_1d_finite(locations, "locations")
        self.residuals = np.sort(_as_1d_finite(residuals, "residuals"))

    def __len__(self) -> int:
        return self.locations.size

    def _rowwise_or_grid(self, values, name: str) -> tuple[np.ndarray, bool]:
        array = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")
        if array.ndim == 0:
            return np.full((len(self), 1), float(array)), True
        if array.ndim == 1:
            if array.size == len(self):
                return array[:, None], True
            return np.broadcast_to(array[None, :], (len(self), array.size)), False
        if array.ndim == 2 and array.shape[0] == len(self):
            return array, False
        raise ValueError(
            f"{name} must be a scalar, a one-dimensional grid, a row-wise vector "
            f"of length {len(self)}, or a matrix with {len(self)} rows."
        )

    def cdf(self, values):
        values, squeeze = self._rowwise_or_grid(values, "values")
        scores = values - self.locations[:, None]
        ranks = np.searchsorted(self.residuals, scores, side="right")
        # The n + 1 denominator matches the conformal rank used by ppf. The
        # otherwise unattainable final rank is assigned to the largest observed
        # residual, yielding a finite, proper, conservative distribution.
        result = ranks.astype(float) / (self.residuals.size + 1)
        result[ranks == self.residuals.size] = 1.0
        return result[:, 0] if squeeze else result

    def ppf(self, quantiles):
        quantiles, squeeze = self._rowwise_or_grid(quantiles, "quantiles")
        if np.any((quantiles < 0.0) | (quantiles > 1.0)):
            raise ValueError("quantiles must lie in [0, 1].")

        # Generalized inverse of the conformal residual CDF. The ceil((n+1)q)
        # rank is the finite-sample conformal correction; the otherwise
        # unattainable final rank is conservatively assigned to the maximum
        # calibration residual.
        ranks = np.ceil((self.residuals.size + 1) * quantiles).astype(int)
        ranks = np.clip(ranks, 1, self.residuals.size) - 1
        result = self.locations[:, None] + self.residuals[ranks]
        return result[:, 0] if squeeze else result


class ContinuousConformalDistribution(_ResidualPredictiveDistribution):
    """Batch of continuous split conformal predictive distributions."""


class DiscreteConformalDistribution(
    _ResidualPredictiveDistribution, DiscretePredictiveDistribution
):
    """Batch of split conformal predictive distributions for integer counts."""

    def __init__(
        self,
        locations: np.ndarray,
        residuals: np.ndarray,
        minimum: int | None = 0,
    ):
        super().__init__(locations, residuals)
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
        if values.ndim == 1 and values.size == len(self):
            return np.where(below, 0.0, result)
        return np.where(np.broadcast_to(below, np.shape(result)), 0.0, result)


class SplitConformalPredictiveSystem(BaseEstimator):
    """Conformalize a fitted point regressor into predictive distributions.

    The system stores signed out-of-sample calibration residuals ``y - y_hat``.
    At prediction time their empirical distribution is shifted by each point
    prediction. Quantiles use the finite-sample ``ceil((n + 1) q)`` conformal rank.

    Parameters
    ----------
    learner : estimator
        A fitted estimator implementing ``predict(X)``.
    discrete : bool, default=False
        If true, produce an ordered integer distribution with ``pmf`` support.
    minimum : int or None, default=0
        Lower support bound for discrete outcomes. Ignored for continuous outcomes.
    """

    def __init__(
        self,
        learner: BaseEstimator,
        discrete: bool = False,
        minimum: int | None = 0,
    ):
        self.learner = learner
        self.discrete = discrete
        self.minimum = minimum

    def fit(self, X, y):
        """Calibrate the predictive system on labeled out-of-sample observations."""
        check_is_fitted(self.learner)
        predictions = _as_1d_finite(self.learner.predict(X), "learner predictions")
        return self.fit_from_predictions(y=y, predictions=predictions)

    def fit_from_predictions(self, y, predictions):
        """Calibrate from precomputed out-of-sample point predictions.

        This entry point is useful for forecasting frameworks whose prediction API
        is horizon-based rather than ``predict(X)``, including MLForecast wrappers.
        """
        y = _as_1d_finite(y, "y")
        predictions = _as_1d_finite(predictions, "predictions")
        if predictions.shape != y.shape:
            raise ValueError("predictions and y must have the same shape.")
        if self.discrete and np.any(y != np.floor(y)):
            raise ValueError("Discrete CPS targets must be integer-valued.")
        if self.discrete and self.minimum is not None and np.any(y < self.minimum):
            raise ValueError(f"Discrete CPS targets must be >= {self.minimum}.")
        self.residuals_ = y - predictions
        self.n_calibration_ = y.size
        return self

    def predict_distribution(self, X) -> PredictiveDistribution:
        """Return one calibrated predictive distribution per input row."""
        check_is_fitted(self, attributes=["residuals_", "n_calibration_"])
        locations = _as_1d_finite(self.learner.predict(X), "learner predictions")
        return self.predict_distribution_from_predictions(locations)

    def predict_distribution_from_predictions(
        self, predictions
    ) -> PredictiveDistribution:
        """Build distributions around precomputed point predictions."""
        check_is_fitted(self, attributes=["residuals_", "n_calibration_"])
        locations = _as_1d_finite(predictions, "predictions")
        if self.discrete:
            return DiscreteConformalDistribution(
                locations, self.residuals_, minimum=self.minimum
            )
        return ContinuousConformalDistribution(locations, self.residuals_)

    def predict_interval(self, X, coverage: float = 0.95) -> np.ndarray:
        """Convenience shortcut for ``predict_distribution(X).interval(...)``."""
        return self.predict_distribution(X).interval(coverage)

    def predict(self, X):
        """Return the conformal predictive median."""
        return self.predict_distribution(X).ppf(0.5)


class ContinuousConformalPredictiveSystem(SplitConformalPredictiveSystem):
    """Explicit continuous-target alias for :class:`SplitConformalPredictiveSystem`."""

    def __init__(self, learner: BaseEstimator):
        super().__init__(learner=learner, discrete=False, minimum=None)


class DiscreteConformalPredictiveSystem(SplitConformalPredictiveSystem):
    """Split conformal predictive system for ordered integer outcomes."""

    def __init__(self, learner: BaseEstimator, minimum: int | None = 0):
        super().__init__(learner=learner, discrete=True, minimum=minimum)
