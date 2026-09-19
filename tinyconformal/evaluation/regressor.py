# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

"""Evaluation of already-produced regression prediction intervals."""

from __future__ import annotations

from numbers import Real

import numpy as np
import pandas as pd

from tinyconformal.core import conformal as core_conformal


class RegressorEvaluator:
    """Evaluate conformal regression intervals independently of their model."""

    @staticmethod
    def evaluate(y_true, intervals, coverage: float) -> pd.DataFrame:
        """Evaluate coverage, width, and mean Winkler interval score.

        Parameters
        ----------
        y_true : array-like of shape (n_observations,)
            Observed target values.
        intervals : array-like of shape (n_observations, 2)
            Lower and upper interval bounds, in that order.
        coverage : float
            Nominal interval coverage, strictly between zero and one.

        Returns
        -------
        pandas.DataFrame
            Single-row evaluation summary. The ``n_obs`` column has integer
            dtype.
        """
        if not isinstance(coverage, Real):
            raise TypeError("coverage must be numeric.")
        coverage = float(coverage)
        if not np.isfinite(coverage) or not 0.0 < coverage < 1.0:
            raise ValueError("coverage must be finite and strictly between 0 and 1.")

        try:
            observed = np.asarray(y_true, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("y_true must contain numeric values.") from exc
        if observed.ndim != 1 or observed.size == 0:
            raise ValueError("y_true must be a non-empty one-dimensional array.")
        if not np.all(np.isfinite(observed)):
            raise ValueError("y_true must contain only finite values.")

        try:
            bounds = np.asarray(intervals, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("intervals must contain numeric values.") from exc
        if bounds.shape != (len(observed), 2):
            raise ValueError("intervals must have shape (len(y_true), 2).")
        if not np.all(np.isfinite(bounds)):
            raise ValueError("intervals must contain only finite values.")
        if np.any(bounds[:, 0] > bounds[:, 1]):
            raise ValueError("lower interval bounds must not exceed upper bounds.")

        alpha = 1.0 - coverage
        result = pd.DataFrame(
            [
                {
                    "coverage": coverage,
                    **core_conformal.interval_metrics(
                        observed, bounds[:, 0], bounds[:, 1], alpha
                    ),
                    "n_obs": len(observed),
                }
            ]
        )
        result["n_obs"] = result["n_obs"].astype("int64")
        return result
