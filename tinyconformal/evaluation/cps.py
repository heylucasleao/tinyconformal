# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

"""Evaluation of tabular conformal predictive systems."""

from __future__ import annotations

import numpy as np
import pandas as pd
from tinyshift.forecasting import crps_distribution, ncrps

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

    @classmethod
    def evaluate_interval(
        cls,
        y_true,
        distribution,
        coverages=(0.5, 0.8, 0.9, 0.95),
    ) -> pd.DataFrame:
        """Evaluate equal-tailed intervals derived from a tabular CPS.

        Parameters
        ----------
        y_true : array-like of shape (n_observations,)
            Observed target values.
        distribution : PredictiveDistribution
            Row-aligned batch of predictive distributions.
        coverages : iterable of float, default=(0.5, 0.8, 0.9, 0.95)
            Central coverage levels to evaluate.

        Returns
        -------
        pandas.DataFrame
            One evaluation row per coverage. The ``n_obs`` column has integer
            dtype.

        Columns
        -------
        **coverage** : ``float``
            Requested central interval coverage.
        **coverage_rate** : ``float``
            Fraction of targets inside the equal-tailed intervals.
        **interval_width_mean** : ``float``
            Mean width of the equal-tailed intervals.
        **mwis** : ``float``
            Mean Winkler interval score. Lower values indicate sharper
            forecasts, conditional on adequate coverage.
        **n_obs** : ``int``
            Number of evaluated predictive distributions.
        """
        observed = cls._observed(y_true)
        cls._validate_alignment(observed, distribution)
        records = []
        for coverage in coverages:
            bounds = distribution.interval(coverage)
            metrics = RegressorEvaluator.evaluate(observed, bounds, coverage).iloc[0]
            records.append(metrics.to_dict())
        result = pd.DataFrame.from_records(records)
        result["n_obs"] = result["n_obs"].astype("int64")
        return result

    @classmethod
    def evaluate_distribution(
        cls,
        y_true,
        distribution,
        scale: float | None = None,
    ) -> pd.DataFrame:
        """Evaluate a complete tabular distribution with CRPS and nCRPS.

        Parameters
        ----------
        y_true : array-like of shape (n_observations,)
            Observed target values.
        distribution : PredictiveDistribution
            Row-aligned batch of predictive distributions.
        scale : float, optional
            Positive scale used to normalize the mean CRPS. When omitted,
            ``scale`` and ``ncrps`` are not returned.

        Returns
        -------
        pandas.DataFrame
            Single-row distributional evaluation summary. The ``n_obs``
            column has integer dtype.

        Columns
        -------
        **crps** : ``float``
            Mean continuous ranked probability score. Lower values are better.
        **scale** : ``float``
            Caller-supplied normalization scale. Present only when ``scale``
            is supplied.
        **ncrps** : ``float``
            Normalized CRPS, computed as ``crps / scale``. Present only when
            ``scale`` is supplied.
        **n_obs** : ``int``
            Number of evaluated predictive distributions.
        """
        observed = cls._observed(y_true)
        cls._validate_alignment(observed, distribution)
        mean_crps = float(np.mean(crps_distribution(observed, distribution)))
        record = {"crps": mean_crps, "n_obs": len(observed)}
        if scale is not None:
            scale = float(scale)
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError("scale must be finite and strictly positive.")
            record.update(
                scale=scale,
                ncrps=float(ncrps(mean_crps, scale)),
            )
        result = pd.DataFrame([record])
        result["n_obs"] = result["n_obs"].astype("int64")
        return result
