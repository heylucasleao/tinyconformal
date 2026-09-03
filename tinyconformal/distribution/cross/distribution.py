# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

"""Cross-fitted empirical predictive distributions for tabular regression."""

from __future__ import annotations

import numpy as np

from ..base import DiscretePredictiveDistribution, EmpiricalResidualDistribution


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
        """Transform standardized residuals into possible future errors."""
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
        """Evaluate integer predictive quantiles.

        Parameters
        ----------
        quantiles : float or array-like
            Probabilities in ``[0, 1]``. A scalar is applied to every prediction
            row, a one-dimensional array defines a common grid, and a matrix
            with ``len(self)`` rows is evaluated row-wise.

        Returns
        -------
        ndarray of int
            Integer predictive quantiles, truncated at ``minimum`` when a lower
            support boundary is configured. Scalar and single-column row-wise
            inputs have shape ``(n_predictions,)``; a grid of ``m`` quantiles
            has shape ``(n_predictions, m)``.

        Raises
        ------
        ValueError
            If a quantile is non-finite or outside ``[0, 1]``, or the input has
            an unsupported shape.
        """
        result = np.ceil(super().ppf(quantiles))
        if self.minimum is not None:
            result = np.maximum(result, self.minimum)
        return result.astype(int)

    def cdf(self, values):
        """Evaluate the discrete predictive cumulative distribution functions.

        Parameters
        ----------
        values : float or array-like
            Values at which to evaluate the CDF. Values are floored to the
            nearest integer support point. A scalar is applied to every
            prediction row, a one-dimensional array defines a common grid, and
            a matrix with ``len(self)`` rows is evaluated row-wise.

        Returns
        -------
        ndarray
            Cumulative probabilities in ``[0, 1]``. Scalar and single-column
            row-wise inputs have shape ``(n_predictions,)``; a grid of ``m``
            values has shape ``(n_predictions, m)``. Values below ``minimum``
            receive probability zero when a lower boundary is configured.

        Raises
        ------
        ValueError
            If a value is non-finite or the input has an unsupported shape.
        """
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

    def pmf(self, values) -> np.ndarray:
        """Evaluate probability masses at integer support values.

        Parameters
        ----------
        values : int or array-like of int
            Support values at which to evaluate the PMF. A scalar is applied to
            every prediction row, a one-dimensional array defines a common
            grid, and a matrix with ``len(self)`` rows is evaluated row-wise.

        Returns
        -------
        ndarray
            Probability masses computed as ``CDF(k) - CDF(k - 1)``. Scalar and
            single-column row-wise inputs have shape ``(n_predictions,)``; a
            grid of ``m`` values has shape ``(n_predictions, m)``.

        Raises
        ------
        ValueError
            If any support value is non-finite or non-integer, or the input has
            an unsupported shape.
        """
        return super().pmf(values)
