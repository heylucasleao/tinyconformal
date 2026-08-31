# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class PredictiveDistribution(ABC):
    """A batch of one-dimensional predictive distributions.

    Implementations contain one predictive distribution for each input row. Scalar
    arguments are applied to every row; one-dimensional arguments with ``len(self)``
    are interpreted row-wise.
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
        """Evaluate probability masses as ``F(k) - F(k - 1)``."""
        values = np.asarray(values)
        if not np.all(np.isfinite(values)) or np.any(values != np.floor(values)):
            raise ValueError("pmf values must be finite integers.")
        return np.asarray(self.cdf(values)) - np.asarray(self.cdf(values - 1))
