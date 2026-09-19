"""Evaluation of tabular conformal predictive systems."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss

from .regressor import RegressorEvaluator


class CPSEvaluator:
    """Evaluate predictive distributions returned by a tabular CPS."""

    @staticmethod
    def _observed(y_true) -> np.ndarray:
        try:
            observed = np.asarray(y_true, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("y_true must contain numeric values.") from exc
        if observed.ndim != 1 or observed.size == 0:
            raise ValueError("y_true must be a non-empty one-dimensional array.")
        if not np.all(np.isfinite(observed)):
            raise ValueError("y_true must contain only finite values.")
        return observed

    @staticmethod
    def _validate_alignment(y_true: np.ndarray, distribution) -> None:
        if len(distribution) != len(y_true):
            raise ValueError(
                "y_true and the predictive distribution must have equal length."
            )

    @staticmethod
    def _crps(distribution, y_true: np.ndarray) -> np.ndarray:
        """Approximate row-wise CRPS from the predictive quantile function."""
        nodes, weights = leggauss(100)
        probabilities = 0.5 * (nodes + 1.0)
        weights = 0.5 * weights
        quantiles = np.asarray(distribution.ppf(probabilities), dtype=float)
        expected_shape = (len(y_true), len(probabilities))
        if quantiles.shape != expected_shape or not np.all(np.isfinite(quantiles)):
            raise ValueError(
                "distribution.ppf must return finite values with shape "
                f"{expected_shape}."
            )
        errors = y_true[:, None] - quantiles
        quantile_loss = np.where(
            errors >= 0.0,
            probabilities * errors,
            (probabilities - 1.0) * errors,
        )
        return 2.0 * np.sum(quantile_loss * weights, axis=1)

    @staticmethod
    def _ncrps(crps: float, scale: float) -> float:
        """Normalize mean CRPS by a finite strictly positive target scale."""
        scale = float(scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("scale must be finite and strictly positive.")
        return float(crps) / scale

    @classmethod
    def evaluate_interval(
        cls,
        y_true,
        distribution,
        coverages=(0.5, 0.8, 0.9, 0.95),
    ) -> pd.DataFrame:
        """Evaluate equal-tailed intervals derived from a tabular CPS."""
        observed = cls._observed(y_true)
        cls._validate_alignment(observed, distribution)
        records = []
        for coverage in coverages:
            bounds = distribution.interval(coverage)
            metrics = RegressorEvaluator.evaluate(observed, bounds, coverage).iloc[0]
            records.append(metrics.to_dict())
        return pd.DataFrame.from_records(records)

    @classmethod
    def evaluate_distribution(
        cls,
        y_true,
        distribution,
        scale: float | None = None,
    ) -> pd.DataFrame:
        """Evaluate a complete tabular distribution with CRPS and nCRPS."""
        observed = cls._observed(y_true)
        cls._validate_alignment(observed, distribution)
        mean_crps = float(np.mean(cls._crps(distribution, observed)))
        record = {"crps": mean_crps, "n_obs": len(observed)}
        if scale is not None:
            record.update(
                scale=float(scale),
                ncrps=cls._ncrps(mean_crps, scale),
            )
        return pd.DataFrame([record])
