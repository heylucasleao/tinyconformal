# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

"""Cross-fitted empirical predictive distributions for tabular regression."""

from __future__ import annotations

import numpy as np

from ..base import (
    DiscretePredictiveDistribution,
    EmpiricalResidualDistribution,
    _IntegerSupportMixin,
)


def _as_1d_finite(values, name: str) -> np.ndarray:
    """Return a non-empty finite one-dimensional float array."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _as_positive_scales(values, name: str = "scales") -> np.ndarray:
    """Return validated finite and strictly positive scales."""
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
        """Store aligned locations, standardized residuals, and row scales."""
        self.locations = _as_1d_finite(locations, "locations")
        self.residuals = np.sort(_as_1d_finite(residuals, "residuals"))
        self.scales = (
            np.ones_like(self.locations)
            if scales is None
            else _as_positive_scales(scales)
        )
        if self.scales.shape != self.locations.shape:
            raise ValueError("scales and locations must have the same shape.")

    @property
    def n_calibration(self) -> int:
        """Return the number of cross-fitted residuals."""
        return self.residuals.size

    def _row_residuals(self) -> np.ndarray:
        """Transform standardized residuals into possible future errors."""
        return self.scales[:, None] * self.residuals[None, :]


class ContinuousConformalDistribution(_ResidualPredictiveDistribution):
    """Batch of continuous cross-fitted conformal predictive distributions."""


class DiscreteConformalDistribution(
    _IntegerSupportMixin,
    _ResidualPredictiveDistribution,
    DiscretePredictiveDistribution,
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
        """Initialize an integer-support empirical predictive distribution."""
        super().__init__(locations, residuals, scales=scales, minimum=minimum)
