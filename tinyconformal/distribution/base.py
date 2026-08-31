# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class PredictiveDistribution(ABC):
    """A batch of one-dimensional predictive distributions.

    Implementations contain one predictive distribution for each input row. Scalar
    arguments are applied to every row; one-dimensional arguments define a common
    evaluation grid. Two-dimensional arguments with ``len(self)`` rows are
    interpreted row-wise.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of predictive distributions in the batch."""

    @abstractmethod
    def cdf(self, values):
        """Evaluate each predictive cumulative distribution function."""

    @abstractmethod
    def ppf(self, quantiles):
        """Evaluate the generalized inverse CDF for each distribution."""

    def interval(self, coverage: float = 0.95) -> np.ndarray:
        """Return equal-tailed intervals with the requested central coverage."""
        if not isinstance(coverage, (int, float, np.integer, np.floating)):
            raise TypeError("coverage must be numeric.")
        coverage = float(coverage)
        if not 0.0 < coverage < 1.0:
            raise ValueError("coverage must be strictly between 0 and 1.")
        alpha = 1.0 - coverage
        quantiles = np.broadcast_to(
            np.array([alpha / 2.0, 1.0 - alpha / 2.0]), (len(self), 2)
        )
        return np.asarray(self.ppf(quantiles))

    def sample(self, n_samples: int = 1, random_state=None) -> np.ndarray:
        """Draw samples by inverse-transform sampling.

        Returns an array shaped ``(n_distributions, n_samples)``.
        """
        if not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer.")
        rng = np.random.default_rng(random_state)
        uniforms = rng.random((len(self), int(n_samples)))
        return np.asarray(self.ppf(uniforms))

    def evaluate(self, y, coverages=(0.5, 0.8, 0.9, 0.95)) -> pd.DataFrame:
        """Evaluate central interval coverage, width, and Winkler score."""
        y = np.asarray(y, dtype=float)
        if y.shape != (len(self),) or not np.all(np.isfinite(y)):
            raise ValueError("y must contain one finite value per distribution row.")
        records = []
        for coverage in coverages:
            bounds = self.interval(coverage)
            alpha = 1.0 - float(coverage)
            lower, upper = bounds[:, 0], bounds[:, 1]
            covered = (y >= lower) & (y <= upper)
            width = upper - lower
            winkler = (
                width
                + (2.0 / alpha) * (lower - y) * (y < lower)
                + (2.0 / alpha) * (y - upper) * (y > upper)
            )
            records.append(
                {
                    "coverage": float(coverage),
                    "empirical_coverage": float(np.mean(covered)),
                    "mean_width": float(np.mean(width)),
                    "winkler_score": float(np.mean(winkler)),
                }
            )
        return pd.DataFrame(records)


class DiscretePredictiveDistribution(PredictiveDistribution):
    """Predictive distribution with an ordered integer support."""

    def pmf(self, values) -> np.ndarray:
        """Evaluate probability masses at integer support values.

        Parameters
        ----------
        values : int or array-like of int
            Support values at which to evaluate the PMF. A scalar is applied to
            every distribution. A one-dimensional array defines a common grid.
            A two-dimensional array with ``len(self)`` rows is evaluated
            row-wise.

        Returns
        -------
        ndarray
            Probability masses computed as ``CDF(k) - CDF(k - 1)``. Scalar and
            single-column row-wise inputs return shape ``(n_distributions,)``;
            a grid of ``m`` values returns shape ``(n_distributions, m)``.

        Raises
        ------
        ValueError
            If any support value is non-finite or non-integer, or the input has
            an unsupported shape.
        """
        values = np.asarray(values)
        if not np.all(np.isfinite(values)) or np.any(values != np.floor(values)):
            raise ValueError("pmf values must be finite integers.")
        return np.asarray(self.cdf(values)) - np.asarray(self.cdf(values - 1))


class EmpiricalResidualDistribution(PredictiveDistribution):
    """Common implementation for predictive distributions shifted by residuals.

    Subclasses only need to expose one sorted residual vector per prediction row
    through :meth:`_row_residuals`. This accommodates both a single split-
    conformal calibration sample and series/horizon-specific samples.
    """

    def _rowwise_or_grid(self, values, name: str) -> tuple[np.ndarray, bool]:
        array = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")
        if array.ndim == 0:
            return np.full((len(self), 1), float(array)), True
        if array.ndim == 1:
            return np.broadcast_to(array[None, :], (len(self), array.size)), False
        if array.ndim == 2 and array.shape[0] == len(self):
            return array, array.shape[1] == 1
        raise ValueError(
            f"{name} must be a scalar, a one-dimensional grid, or a matrix with "
            f"{len(self)} rows."
        )

    @abstractmethod
    def _row_residuals(self) -> np.ndarray:
        """Return sorted residuals shaped ``(n_rows, n_calibration)``."""

    @property
    @abstractmethod
    def n_calibration(self) -> int:
        """Return the number of residuals available for every row."""

    def cdf(self, values):
        """Evaluate each predictive cumulative distribution function.

        Parameters
        ----------
        values : float or array-like
            Target values at which to evaluate the CDF. A scalar is applied to
            every distribution. A one-dimensional array defines a common grid.
            A two-dimensional array with ``len(self)`` rows is evaluated
            row-wise.

        Returns
        -------
        ndarray
            Cumulative probabilities in ``[0, 1]``. Scalar and single-column
            row-wise inputs return shape ``(n_distributions,)``; a grid of ``m``
            values returns shape ``(n_distributions, m)``.

        Raises
        ------
        ValueError
            If any value is non-finite or the input is not a scalar, a
            one-dimensional grid, or a matrix with ``len(self)`` rows.
        """
        values, squeeze = self._rowwise_or_grid(values, "values")
        scores = values - self.locations[:, None]
        residuals = self._row_residuals()
        ranks = np.sum(residuals[:, :, None] <= scores[:, None, :], axis=1)
        result = ranks.astype(float) / (self.n_calibration + 1)
        result[ranks == self.n_calibration] = 1.0
        return result[:, 0] if squeeze else result

    def ppf(self, quantiles):
        """Evaluate the generalized inverse CDF for each distribution.

        Parameters
        ----------
        quantiles : float or array-like
            Probabilities in ``[0, 1]``. A scalar is applied to every
            distribution. A one-dimensional array defines a common quantile
            grid. A two-dimensional array with ``len(self)`` rows is evaluated
            row-wise.

        Returns
        -------
        ndarray
            Predictive quantiles. Scalar and single-column row-wise inputs
            return shape ``(n_distributions,)``; a grid of ``m`` quantiles
            returns shape ``(n_distributions, m)``.

        Raises
        ------
        ValueError
            If any quantile is non-finite or outside ``[0, 1]``, or the input is
            not a scalar, a one-dimensional grid, or a matrix with ``len(self)``
            rows.

        Notes
        -----
        Quantiles use the finite-sample conformal rank
        ``ceil((n_calibration + 1) * q)``.
        """
        quantiles, squeeze = self._rowwise_or_grid(quantiles, "quantiles")
        if np.any((quantiles < 0.0) | (quantiles > 1.0)):
            raise ValueError("quantiles must lie in [0, 1].")
        ranks = np.ceil((self.n_calibration + 1) * quantiles).astype(int)
        ranks = np.clip(ranks, 1, self.n_calibration) - 1
        selected = np.take_along_axis(self._row_residuals(), ranks, axis=1)
        result = self.locations[:, None] + selected
        return result[:, 0] if squeeze else result
