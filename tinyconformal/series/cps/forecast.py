"""Panel-aligned facades over CPS predictive distributions."""

# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from tinyconformal.distribution.base import PredictiveDistribution

__all__ = ["DiscretePanelConformalForecast", "PanelConformalForecast"]


class PanelConformalForecast:
    """Panel-aligned facade over a batch of CPS predictive distributions.

    Instances are returned by
    :meth:`TSCPS.predict_distribution
    <tinyconformal.series.cps.base.TSCPS.predict_distribution>`; users normally
    do not construct this class directly. Every distributional method preserves
    the rows and point-forecast columns returned by :meth:`to_frame`.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        distribution: PredictiveDistribution,
        model: str,
        id_col: str,
        time_col: str,
    ):
        """Store an isolated forecast frame and its row-aligned distribution."""
        self._frame = frame.copy()
        self._distribution = distribution
        self.model = model
        self.id_col = id_col
        self.time_col = time_col

    def __len__(self) -> int:
        """Return the number of forecast rows."""
        return len(self._distribution)

    @property
    def distribution(self) -> PredictiveDistribution:
        """Return the row-aligned predictive distribution."""
        return self._distribution

    def to_frame(self) -> pd.DataFrame:
        """Return the point-forecast panel without distributional columns.

        Returns
        -------
        pandas.DataFrame
            A copy containing the series identifier, timestamp, point forecast,
            and any future exogenous columns retained during prediction.
            Changing it does not mutate this forecast.
        """
        return self._frame.copy()

    @staticmethod
    def _label(value) -> str:
        """Format a stable numeric component for a result column name."""
        return np.format_float_positional(float(value), precision=12, trim="-")

    def _apply(self, method: str, inputs, labeler, row_label: str) -> pd.DataFrame:
        """Evaluate a distribution method and append its output to the panel."""
        inputs_array = np.asarray(inputs)
        values = np.asarray(getattr(self._distribution, method)(inputs))
        result = self.to_frame()
        if values.ndim == 1:
            column = labeler(inputs_array) if inputs_array.ndim == 0 else row_label
            result[column] = values
            return result
        labels = np.ravel(inputs_array)
        common_grid = inputs_array.ndim == 1 and labels.size == values.shape[1]
        for index in range(values.shape[1]):
            column = labeler(labels[index]) if common_grid else f"{row_label}-{index}"
            result[column] = values[:, index]
        return result

    def cdf(self, values: ArrayLike) -> pd.DataFrame:
        """Evaluate the cumulative distribution function on the panel.

        Parameters
        ----------
        values : float or array-like of float
            Target values at which to evaluate each predictive CDF. A scalar
            is applied to every forecast row. A one-dimensional array defines
            a common evaluation grid. A two-dimensional array with
            ``len(self)`` rows is evaluated row-wise.

        Returns
        -------
        pandas.DataFrame
            The point-forecast panel plus the evaluated probabilities. Scalar
            and common-grid columns use mathematical names such as
            ``P(Y<=5)``. Row-wise input uses ``P(Y<=value)`` (with a numeric
            suffix when it contains multiple columns).

        Raises
        ------
        ValueError
            If a value is non-finite or the input shape is unsupported.

        Notes
        -----
        CDF values lie in ``[0, 1]`` and remain positionally aligned with
        :meth:`to_frame`.
        """
        return self._apply(
            "cdf", values, lambda value: f"P(Y<={self._label(value)})", "P(Y<=value)"
        )

    def sf(self, values: ArrayLike) -> pd.DataFrame:
        """Evaluate exceedance probabilities on the forecast panel.

        Parameters
        ----------
        values : float or array-like of float
            Thresholds at which to evaluate ``P(Y > value)``. A scalar is
            applied to every forecast row. A one-dimensional array defines a
            common threshold grid. A two-dimensional array with ``len(self)``
            rows is evaluated row-wise.

        Returns
        -------
        pandas.DataFrame
            The point-forecast panel plus exceedance probabilities. Scalar and
            common-grid columns use mathematical names such as ``P(Y>5)``.
            Row-wise input uses ``P(Y>value)`` (with a numeric suffix when it
            contains multiple columns).

        Raises
        ------
        ValueError
            If a value is non-finite or the input shape is unsupported.

        Notes
        -----
        The survival function is the complementary CDF, ``1 - F(value)``.
        """
        return self._apply(
            "sf", values, lambda value: f"P(Y>{self._label(value)})", "P(Y>value)"
        )

    def ppf(self, quantiles: ArrayLike) -> pd.DataFrame:
        """Evaluate predictive quantiles on the forecast panel.

        Parameters
        ----------
        quantiles : float or array-like of float
            Probabilities in ``[0, 1]``. A scalar is applied to every forecast
            row. A one-dimensional array defines a common quantile grid. A
            two-dimensional array with ``len(self)`` rows supplies row-wise
            quantile levels.

        Returns
        -------
        pandas.DataFrame
            The point-forecast panel plus requested quantiles. Scalar and
            common-grid columns use mathematical names such as ``Q(0.9)``.
            Row-wise input uses ``Q(p)`` (with a numeric suffix when it contains
            multiple columns). Discrete forecasts return integer quantiles.

        Raises
        ------
        ValueError
            If a quantile is non-finite, outside ``[0, 1]``, or the input shape
            is unsupported.

        Examples
        --------
        ``forecast.ppf([0.1, 0.5, 0.9])`` returns the 10th percentile, median,
        and 90th percentile for every forecast row.
        """
        return self._apply(
            "ppf", quantiles, lambda value: f"Q({self._label(value)})", "Q(p)"
        )

    def interval(self, coverage: float = 0.95) -> pd.DataFrame:
        """Return an equal-tailed central predictive interval.

        Parameters
        ----------
        coverage : float, default=0.95
            Central probability covered by the interval. It must be strictly
            between 0 and 1. For example, ``0.9`` requests a 90% interval with
            5% probability in each tail.

        Returns
        -------
        pandas.DataFrame
            The point-forecast panel plus lower and upper quantiles named by
            their probability levels, for example ``Q(0.05)`` and ``Q(0.95)``
            for 90% coverage.

        Raises
        ------
        ValueError
            If ``coverage`` is not strictly between 0 and 1.
        TypeError
            If ``coverage`` is not numeric.
        """
        bounds = np.asarray(self._distribution.interval(coverage))
        alpha = 1.0 - float(coverage)
        result = self.to_frame()
        result[f"Q({self._label(alpha / 2.0)})"] = bounds[:, 0]
        result[f"Q({self._label(1.0 - alpha / 2.0)})"] = bounds[:, 1]
        return result

    def evaluate(self, y, coverages=(0.5, 0.8, 0.9, 0.95)) -> pd.DataFrame:
        """Evaluate central-interval calibration against observed outcomes.

        Parameters
        ----------
        y : array-like of float
            One finite observed target per forecast row, in the same positional
            order as :meth:`to_frame`.
        coverages : iterable of float, default=(0.5, 0.8, 0.9, 0.95)
            Central interval coverages to evaluate. Every value must lie
            strictly between 0 and 1.

        Returns
        -------
        pandas.DataFrame
            One row per requested coverage containing empirical coverage,
            average interval width, and mean Winkler interval score.

        Raises
        ------
        ValueError
            If ``y`` does not contain exactly one finite value per forecast row,
            or if a requested coverage is invalid.

        Notes
        -----
        Observations are positionally aligned rather than joined by identifier
        and timestamp. Sort or merge targets to match :meth:`to_frame` before
        calling this method.
        """
        return self.distribution.evaluate(y, coverages=coverages)


class DiscretePanelConformalForecast(PanelConformalForecast):
    """Panel forecast for integer targets, additionally exposing a PMF."""

    def pmf(self, values: ArrayLike) -> pd.DataFrame:
        """Evaluate probability masses on the discrete forecast panel.

        Parameters
        ----------
        values : int or array-like of int
            Integer support values at which to evaluate each predictive PMF. A
            scalar is applied to every forecast row. A one-dimensional array
            defines a common support grid. A two-dimensional array with
            ``len(self)`` rows is evaluated row-wise.

        Returns
        -------
        pandas.DataFrame
            The point-forecast panel plus mathematically named masses, such as
            ``P(Y=5)``. Use :meth:`sf` separately to obtain exceedance
            probabilities such as ``P(Y>5)``.

        Raises
        ------
        ValueError
            If a value is non-finite or non-integer, the input is empty, or its
            shape is unsupported.

        Notes
        -----
        This method exists only for discrete CPS forecasts. Each probability
        mass is evaluated on the integer support of the underlying distribution.
        """
        inputs = np.asarray(values)
        if inputs.size == 0:
            raise ValueError("pmf values must not be empty.")
        return self._apply(
            "pmf", values, lambda value: f"P(Y={self._label(value)})", "P(Y=value)"
        )
