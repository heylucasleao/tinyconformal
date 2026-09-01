# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from tinyconformal.distribution.base import (
    DiscretePredictiveDistribution,
    EmpiricalResidualDistribution,
)


def _validate_matrix(values, name: str) -> np.ndarray:
    """Convert values to a finite, non-empty two-dimensional array."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError(f"{name} must be a non-empty two-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


class HorizonConformalDistribution(EmpiricalResidualDistribution):
    """Batch of residual-based predictive distributions calibrated by horizon.

    Each prediction is represented as a point forecast plus the empirical
    distribution of signed calibration residuals for its forecast step.  The
    object implements the common :class:`PredictiveDistribution` interface, so
    callers can evaluate CDFs, request arbitrary quantiles, construct central
    intervals without refitting the forecasting model.

    Parameters
    ----------
    locations : ndarray of shape (n_predictions,)
        Point forecasts in sorted Nixtla panel order.
    residuals : ndarray or mapping
        Signed residuals ``y - y_hat`` from sequential backtesting. A matrix
        applies the same calibration distribution to every prediction. A
        mapping associates each series identifier with its own matrix of shape
        ``(n_calibration_trajectories, horizon)``.
    horizon_steps : ndarray of shape (n_predictions,)
        Zero-based horizon index associated with every point forecast.
    series_ids : ndarray of shape (n_predictions,), optional
        Series identifier for every prediction. Required when ``residuals`` is
        a mapping.
    scales : ndarray of shape (n_predictions,), optional
        Positive conditional scale for each prediction. Defaults to one.
    weights : ndarray of shape (n_calibration_trajectories,), optional
        Non-negative calibration-window weights. They are normalized internally;
        equal conformal ranks are used when omitted.

    Attributes
    ----------
    locations : ndarray of shape (n_predictions,)
        Point forecasts defining the location of each predictive distribution.
    residuals : ndarray or dict
        Sorted signed calibration residuals, either pooled or keyed by series.
    horizon_steps : ndarray of shape (n_predictions,)
        Horizon step used to select the residual distribution for each row.
    scales : ndarray of shape (n_predictions,)
        Conditional scales applied to the standardized residuals.
    weights : ndarray or None
        Normalized calibration-window weights.

    Notes
    -----
    If ``r = y - y_hat`` denotes a signed calibration residual, the predictive
    distribution for row ``i`` is the empirical distribution of
    ``locations[i] + r[:, horizon_steps[i]]``.  The CDF uses conformal ranks with
    denominator ``n_calibration_trajectories + 1`` and the PPF uses the
    corresponding finite-sample ceiling rule. When ``weights`` are supplied,
    CDFs and quantiles use their normalized weighted empirical counterparts.

    Rows are positional.  Their order must remain identical to the associated
    Nixtla forecast DataFrame returned by ``predict_distribution``.
    """

    def __init__(
        self,
        locations,
        residuals,
        horizon_steps,
        series_ids=None,
        scales=None,
        weights=None,
    ):
        self.locations = self._validate_locations(locations)
        self.horizon_steps = self._validate_horizon_steps(horizon_steps)
        self.residuals, self.series_ids = self._prepare_residuals(residuals, series_ids)
        self.scales = self._validate_scales(scales)
        self._n_calibration, calibrated_horizon = self._residual_shape()
        self.weights = self._validate_weights(weights)
        self._validate_calibrated_horizon(calibrated_horizon)

    def _validate_scales(self, scales) -> np.ndarray:
        """Return positive row-aligned scales, defaulting to one."""
        if scales is None:
            return np.ones_like(self.locations)
        scales = np.asarray(scales, dtype=float)
        if scales.shape != self.locations.shape or not np.all(np.isfinite(scales)):
            raise ValueError("scales must be finite and match locations.")
        if np.any(scales <= 0.0):
            raise ValueError("scales must be strictly positive.")
        return scales

    def _validate_weights(self, weights):
        """Validate and normalize calibration-window weights."""
        if weights is None:
            return None
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (self._n_calibration,):
            raise ValueError("weights must match the number of calibration windows.")
        if (
            not np.all(np.isfinite(weights))
            or np.any(weights < 0)
            or weights.sum() <= 0
        ):
            raise ValueError(
                "weights must be finite, non-negative, and have positive mass."
            )
        return weights / weights.sum()

    @staticmethod
    def _validate_locations(locations) -> np.ndarray:
        """Return locations as a finite one-dimensional array."""
        locations = np.asarray(locations, dtype=float)
        if locations.ndim != 1 or not np.all(np.isfinite(locations)):
            raise ValueError("locations must be a finite one-dimensional array.")
        return locations

    def _validate_horizon_steps(self, horizon_steps) -> np.ndarray:
        """Validate that every location has a corresponding horizon step."""
        horizon_steps = np.asarray(horizon_steps, dtype=int)
        if horizon_steps.shape != self.locations.shape:
            raise ValueError("horizon_steps and locations must have the same shape.")
        return horizon_steps

    def _prepare_residuals(self, residuals, series_ids):
        """Normalize pooled or per-series calibration residuals."""
        if isinstance(residuals, Mapping):
            return self._prepare_series_residuals(residuals, series_ids)
        residuals = _validate_matrix(residuals, "residuals")
        return residuals, None

    def _prepare_series_residuals(self, residuals, series_ids):
        """Validate residual matrices and align them with prediction series."""
        if series_ids is None:
            raise ValueError("series_ids is required when residuals is a mapping.")
        series_ids = np.asarray(series_ids)
        if series_ids.shape != self.locations.shape:
            raise ValueError("series_ids and locations must have the same shape.")
        missing_ids = sorted(set(series_ids) - set(residuals), key=str)
        if missing_ids:
            raise ValueError(
                f"No CPS calibration residuals are available for series: {missing_ids}"
            )
        prepared = {
            series_id: _validate_matrix(values, f"residuals[{series_id!r}]")
            for series_id, values in residuals.items()
        }
        return prepared, series_ids

    def _residual_shape(self) -> tuple[int, int]:
        """Return the common calibration-window and horizon dimensions."""
        if not isinstance(self.residuals, Mapping):
            return self.residuals.shape
        shapes = {values.shape for values in self.residuals.values()}
        if len(shapes) != 1:
            raise ValueError("All series residual matrices must have the same shape.")
        return next(iter(shapes))

    def _validate_calibrated_horizon(self, calibrated_horizon: int) -> None:
        """Reject prediction rows outside the calibrated horizon."""
        if np.any(
            (self.horizon_steps < 0) | (self.horizon_steps >= calibrated_horizon)
        ):
            raise ValueError("horizon_steps contains an uncalibrated horizon index.")

    def __len__(self) -> int:
        return self.locations.size

    @property
    def n_calibration(self) -> int:
        return self._n_calibration

    def _row_residuals(self) -> np.ndarray:
        """Select and scale the calibration residuals for every forecast row."""
        if self.series_ids is not None:
            residuals = np.vstack(
                [
                    self.residuals[series_id][:, horizon_step]
                    for series_id, horizon_step in zip(
                        self.series_ids, self.horizon_steps
                    )
                ]
            )
        else:
            residuals = self.residuals[:, self.horizon_steps].T
        residuals = self.scales[:, None] * residuals
        return residuals if self.weights is not None else np.sort(residuals, axis=1)

    def cdf(self, values):
        if self.weights is None:
            return super().cdf(values)
        values, squeeze = self._rowwise_or_grid(values, "values")
        scores = values - self.locations[:, None]
        residuals = self._row_residuals()
        result = np.sum(
            self.weights[None, :, None] * (residuals[:, :, None] <= scores[:, None, :]),
            axis=1,
        )
        return result[:, 0] if squeeze else result

    def _sorted_weighted_residuals(self) -> tuple[np.ndarray, np.ndarray]:
        """Sort each residual row and align its calibration weights."""
        residuals = self._row_residuals()
        order = np.argsort(residuals, axis=1)
        sorted_residuals = np.take_along_axis(residuals, order, axis=1)
        row_weights = np.broadcast_to(self.weights, residuals.shape)
        sorted_weights = np.take_along_axis(row_weights, order, axis=1)
        return sorted_residuals, sorted_weights

    @staticmethod
    def _weighted_quantile_indices(sorted_weights, quantiles) -> np.ndarray:
        """Locate the first weighted empirical rank reaching each quantile."""
        cumulative = np.cumsum(sorted_weights, axis=1)
        return np.argmax(cumulative[:, :, None] >= quantiles[:, None, :], axis=1)

    def ppf(self, quantiles):
        """Evaluate predictive quantiles, using window weights when configured."""
        if self.weights is None:
            return super().ppf(quantiles)
        quantiles, squeeze = self._rowwise_or_grid(quantiles, "quantiles")
        if np.any((quantiles < 0.0) | (quantiles > 1.0)):
            raise ValueError("quantiles must lie in [0, 1].")
        sorted_residuals, sorted_weights = self._sorted_weighted_residuals()
        indices = self._weighted_quantile_indices(sorted_weights, quantiles)
        selected = np.take_along_axis(sorted_residuals, indices, axis=1)
        result = self.locations[:, None] + selected
        return result[:, 0] if squeeze else result


class DiscreteHorizonConformalDistribution(
    HorizonConformalDistribution, DiscretePredictiveDistribution
):
    """Horizon-wise conformal predictive distributions on integer support.

    This specialization converts CPS quantiles to integers with ``ceil`` and
    optionally truncates the support below at ``minimum``.  It also inherits
    :meth:`~tinyconformal.distribution.base.DiscretePredictiveDistribution.pmf`,
    which evaluates probability masses as ``F(k) - F(k - 1)``.

    Parameters
    ----------
    locations : ndarray of shape (n_predictions,)
        Point forecasts in sorted Nixtla panel order.
    residuals : ndarray of shape (n_calibration_trajectories, horizon)
        Signed residuals ``y - y_hat`` obtained by sequential backtesting.
    horizon_steps : ndarray of shape (n_predictions,)
        Zero-based forecast step associated with each prediction row.
    minimum : int or None, default=0
        Lower boundary of the integer support.  If ``None``, no lower boundary
        is imposed.
    series_ids : ndarray of shape (n_predictions,), optional
        Series identifier for every prediction. Required when ``residuals`` is
        a mapping.
    scales : ndarray of shape (n_predictions,), optional
        Positive conditional scale for each prediction. Defaults to one.
    weights : ndarray of shape (n_calibration_trajectories,), optional
        Non-negative calibration-window weights, normalized internally.

    Notes
    -----
    The class is intended for genuinely discrete targets such as demand counts.
    Rounding a continuous target merely to expose a PMF changes the statistical
    object and is not equivalent to a continuous CPS.
    """

    def __init__(
        self,
        locations,
        residuals,
        horizon_steps,
        minimum: int | None = 0,
        series_ids=None,
        scales=None,
        weights=None,
    ):
        super().__init__(
            locations,
            residuals,
            horizon_steps,
            series_ids=series_ids,
            scales=scales,
            weights=weights,
        )
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
        if values.ndim == 0:
            return np.zeros_like(result) if bool(below) else result
        if result.ndim == 1:
            return np.where(np.ravel(below), 0.0, result)
        return np.where(np.broadcast_to(below, np.shape(result)), 0.0, result)
