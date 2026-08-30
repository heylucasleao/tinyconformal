# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from .base import PredictiveDistribution


def validate_quantile_levels(quantiles) -> np.ndarray:
    """Validate a strictly increasing interior probability grid."""
    levels = np.asarray(quantiles, dtype=float)
    if levels.ndim != 1 or levels.size < 2:
        raise ValueError("quantiles must contain at least two probability levels.")
    if not np.all(np.isfinite(levels)) or np.any(
        (levels <= 0.0) | (levels >= 1.0)
    ):
        raise ValueError("quantiles must contain finite values strictly inside (0, 1).")
    if np.any(np.diff(levels) <= 0.0):
        raise ValueError("quantiles must be strictly increasing and unique.")
    return levels


def validate_quantile_predictions(
    predictions, quantiles: np.ndarray, name: str = "quantile predictions"
) -> np.ndarray:
    """Validate and monotonize a batch of predicted quantile functions."""
    values = np.asarray(predictions, dtype=float)
    if values.ndim != 2 or values.shape[1] != quantiles.size:
        raise ValueError(
            f"{name} must have shape (n_samples, {quantiles.size})."
        )
    if values.shape[0] == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be non-empty and contain only finite values.")
    # Rearrangement by cumulative maximum is deterministic and preserves already
    # valid quantile functions while removing crossings.
    return np.maximum.accumulate(values, axis=1)


class QuantileGridDistribution(PredictiveDistribution):
    """Conditional distributions represented by monotone quantile grids."""

    def __init__(self, quantile_predictions, quantiles):
        self.quantiles = validate_quantile_levels(quantiles)
        self.quantile_predictions = validate_quantile_predictions(
            quantile_predictions, self.quantiles
        )

    def __len__(self) -> int:
        return self.quantile_predictions.shape[0]

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

    @staticmethod
    def _unique_quantile_knots(values: np.ndarray, levels: np.ndarray):
        """Keep the highest probability at each flat quantile value for CDF inversion."""
        unique_values, first, counts = np.unique(
            values, return_index=True, return_counts=True
        )
        last = first + counts - 1
        return unique_values, levels[last]

    def cdf(self, values):
        values, squeeze = self._rowwise_or_grid(values, "values")
        result = np.empty_like(values, dtype=float)
        for row in range(len(self)):
            knots, probabilities = self._unique_quantile_knots(
                self.quantile_predictions[row], self.quantiles
            )
            result[row] = np.interp(
                values[row], knots, probabilities, left=0.0, right=1.0
            )
        return result[:, 0] if squeeze else result

    def ppf(self, quantiles):
        quantiles, squeeze = self._rowwise_or_grid(quantiles, "quantiles")
        if np.any((quantiles < 0.0) | (quantiles > 1.0)):
            raise ValueError("quantiles must lie in [0, 1].")
        result = np.empty_like(quantiles, dtype=float)
        for row in range(len(self)):
            result[row] = np.interp(
                quantiles[row],
                self.quantiles,
                self.quantile_predictions[row],
                left=self.quantile_predictions[row, 0],
                right=self.quantile_predictions[row, -1],
            )
        return result[:, 0] if squeeze else result


class DistributionalConformalDistribution(PredictiveDistribution):
    """PIT-calibrated conditional distributions.

    ``calibration_pits`` may be shared across rows with shape ``(n_calibration,)``
    or supplied row-wise with shape ``(n_predictions, n_calibration)``. The latter
    enables horizon-specific calibration for time-series forecasts.
    """

    def __init__(self, base_distribution: PredictiveDistribution, calibration_pits):
        self.base_distribution = base_distribution
        pits = np.asarray(calibration_pits, dtype=float)
        if pits.ndim == 1:
            pits = np.broadcast_to(pits, (len(base_distribution), pits.size))
        if pits.ndim != 2 or pits.shape[0] != len(base_distribution) or pits.shape[1] == 0:
            raise ValueError(
                "calibration_pits must be a non-empty vector or a matrix with one row "
                "per predictive distribution."
            )
        if not np.all(np.isfinite(pits)) or np.any((pits < 0.0) | (pits > 1.0)):
            raise ValueError("calibration_pits must contain finite values in [0, 1].")
        self.calibration_pits = np.sort(pits, axis=1)

    def __len__(self) -> int:
        return len(self.base_distribution)

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
        base_probabilities = np.asarray(self.base_distribution.cdf(values))
        squeeze = base_probabilities.ndim == 1
        if squeeze:
            base_probabilities = base_probabilities[:, None]
        ranks = np.sum(
            self.calibration_pits[:, :, None] <= base_probabilities[:, None, :],
            axis=1,
        )
        n = self.calibration_pits.shape[1]
        result = ranks.astype(float) / (n + 1)
        result[ranks == n] = 1.0
        return result[:, 0] if squeeze else result

    def ppf(self, quantiles):
        quantiles, squeeze = self._rowwise_or_grid(quantiles, "quantiles")
        if np.any((quantiles < 0.0) | (quantiles > 1.0)):
            raise ValueError("quantiles must lie in [0, 1].")
        n = self.calibration_pits.shape[1]
        ranks = np.ceil((n + 1) * quantiles).astype(int)
        ranks = np.clip(ranks, 1, n) - 1
        base_levels = np.take_along_axis(self.calibration_pits, ranks, axis=1)
        result = np.asarray(self.base_distribution.ppf(base_levels))
        return result[:, 0] if squeeze else result


class DistributionalConformalPredictiveSystem(BaseEstimator):
    """PIT-based DCP over a fitted conditional quantile learner.

    The learner must implement ``predict(X, quantiles=levels)``. For frameworks
    with a different prediction API, use ``fit_from_predictions`` and
    ``predict_distribution_from_predictions``.
    """

    def __init__(self, learner: BaseEstimator | None = None, quantiles=None):
        self.learner = learner
        self.quantiles = quantiles

    def _levels(self) -> np.ndarray:
        if self.quantiles is None:
            raise ValueError(
                "quantiles must be provided when using a quantile-grid learner."
            )
        return validate_quantile_levels(self.quantiles)

    def _predict_quantiles(self, X) -> np.ndarray:
        if self.learner is None:
            raise ValueError(
                "learner is required for predict(X); use the precomputed or base-"
                "distribution methods when no learner is configured."
            )
        try:
            predictions = self.learner.predict(X, quantiles=list(self._levels()))
        except TypeError as exc:
            raise TypeError(
                "The learner must implement predict(X, quantiles=levels), or use "
                "the precomputed-prediction methods."
            ) from exc
        return np.asarray(predictions)

    def fit(self, X, y):
        if self.learner is None:
            raise ValueError("learner is required for fit(X, y).")
        check_is_fitted(self.learner)
        return self.fit_from_predictions(y, self._predict_quantiles(X))

    def fit_from_predictions(self, y, quantile_predictions):
        levels = self._levels()
        base = QuantileGridDistribution(quantile_predictions, levels)
        return self.fit_from_distribution(y, base)

    def fit_from_distribution(self, y, base_distribution: PredictiveDistribution):
        """Calibrate directly from any batch distribution implementing CDF and PPF."""
        y = np.asarray(y, dtype=float)
        if y.ndim != 1 or y.size != len(base_distribution):
            raise ValueError("y must be one-dimensional and match the distribution rows.")
        if not np.all(np.isfinite(y)):
            raise ValueError("y must contain only finite values.")
        self.calibration_pits_ = np.asarray(base_distribution.cdf(y), dtype=float)
        if self.calibration_pits_.shape != y.shape:
            raise ValueError("base_distribution.cdf(y) must return one PIT per row.")
        self.n_calibration_ = y.size
        return self

    def predict_distribution(self, X) -> DistributionalConformalDistribution:
        check_is_fitted(self, attributes=["calibration_pits_", "n_calibration_"])
        return self.predict_distribution_from_predictions(self._predict_quantiles(X))

    def predict_distribution_from_predictions(
        self, quantile_predictions
    ) -> DistributionalConformalDistribution:
        check_is_fitted(self, attributes=["calibration_pits_", "n_calibration_"])
        base = QuantileGridDistribution(quantile_predictions, self._levels())
        return self.predict_distribution_from_base(base)

    def predict_distribution_from_base(
        self, base_distribution: PredictiveDistribution
    ) -> DistributionalConformalDistribution:
        """Recalibrate any native conditional distribution through fitted PIT ranks."""
        check_is_fitted(self, attributes=["calibration_pits_", "n_calibration_"])
        return DistributionalConformalDistribution(
            base_distribution, self.calibration_pits_
        )

    def predict_interval(self, X, coverage: float = 0.95) -> np.ndarray:
        return self.predict_distribution(X).interval(coverage)

    def predict(self, X):
        return self.predict_distribution(X).ppf(0.5)
