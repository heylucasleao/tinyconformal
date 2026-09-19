"""Evaluation of complete predictive distributions and their intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss

from .regressor import RegressorEvaluator


class DistributionEvaluator:
    """Evaluate already-produced predictive distributions."""

    @staticmethod
    def _unwrap(prediction):
        """Return the distribution contained in a panel forecast, if present."""
        return getattr(prediction, "distribution", prediction)

    @staticmethod
    def _observed(
        y_true,
        prediction,
        target_col: str,
        id_col: str | None,
        time_col: str | None,
    ) -> np.ndarray:
        if not isinstance(y_true, pd.DataFrame):
            try:
                observed = np.asarray(y_true, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError("y_true must contain numeric values.") from exc
            if observed.ndim != 1 or observed.size == 0:
                raise ValueError("y_true must be a non-empty one-dimensional array.")
            if not np.all(np.isfinite(observed)):
                raise ValueError("y_true must contain only finite values.")
            return observed

        if not hasattr(prediction, "to_frame"):
            raise TypeError(
                "DataFrame y_true requires a panel forecast exposing to_frame()."
            )
        frame = prediction.to_frame()
        id_col = id_col or getattr(prediction, "id_col", "unique_id")
        time_col = time_col or getattr(prediction, "time_col", "ds")
        keys = [id_col, time_col]
        required_true = [*keys, target_col]
        for value, required, name in (
            (y_true, required_true, "y_true"),
            (frame, keys, "forecast"),
        ):
            missing = [column for column in required if column not in value.columns]
            if missing:
                raise KeyError(f"Columns not found in {name}: {missing}")
            if value.duplicated(keys).any():
                raise ValueError(f"{name} contains duplicate identifier/time rows.")
        aligned = frame[keys].merge(
            y_true[required_true], on=keys, how="left", validate="one_to_one"
        )
        if aligned[target_col].isna().any():
            raise ValueError("y_true must contain a target for every forecast row.")
        observed = aligned[target_col].to_numpy(dtype=float)
        if not np.all(np.isfinite(observed)):
            raise ValueError("y_true must contain only finite values.")
        return observed

    @classmethod
    def evaluate_interval(
        cls,
        y_true,
        distribution=None,
        forecast=None,
        coverages=(0.5, 0.8, 0.9, 0.95),
        target_col: str = "y",
        id_col: str | None = None,
        time_col: str | None = None,
    ) -> pd.DataFrame:
        """Evaluate equal-tailed intervals derived from a distribution."""
        prediction = forecast if forecast is not None else distribution
        if prediction is None:
            raise TypeError("Pass either distribution or forecast.")
        observed = cls._observed(y_true, prediction, target_col, id_col, time_col)
        predictive_distribution = cls._unwrap(prediction)
        if len(predictive_distribution) != len(observed):
            raise ValueError(
                "y_true and the predictive distribution must have equal length."
            )
        records = []
        for coverage in coverages:
            bounds = predictive_distribution.interval(coverage)
            metrics = RegressorEvaluator.evaluate(observed, bounds, coverage).iloc[0]
            records.append(metrics.to_dict())
        return pd.DataFrame.from_records(records)

    @classmethod
    def evaluate_distribution(
        cls,
        y_true,
        distribution=None,
        forecast=None,
        scale: float | None = None,
        target_col: str = "y",
        id_col: str | None = None,
        time_col: str | None = None,
    ) -> pd.DataFrame:
        """Evaluate the complete distribution using CRPS and optional nCRPS."""
        prediction = forecast if forecast is not None else distribution
        if prediction is None:
            raise TypeError("Pass either distribution or forecast.")
        observed = cls._observed(y_true, prediction, target_col, id_col, time_col)
        predictive_distribution = cls._unwrap(prediction)
        if len(predictive_distribution) != len(observed):
            raise ValueError(
                "y_true and the predictive distribution must have equal length."
            )

        nodes, weights = leggauss(100)
        probabilities = 0.5 * (nodes + 1.0)
        weights = 0.5 * weights
        quantiles = np.asarray(predictive_distribution.ppf(probabilities), dtype=float)
        expected_shape = (len(observed), len(probabilities))
        if quantiles.shape != expected_shape or not np.all(np.isfinite(quantiles)):
            raise ValueError(
                "distribution.ppf must return finite values with shape "
                f"{expected_shape}."
            )
        errors = observed[:, None] - quantiles
        quantile_loss = np.where(
            errors >= 0.0,
            probabilities * errors,
            (probabilities - 1.0) * errors,
        )
        row_crps = 2.0 * np.sum(quantile_loss * weights, axis=1)
        record = {"crps": float(np.mean(row_crps)), "n_obs": len(observed)}
        if scale is not None:
            scale = float(scale)
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError("scale must be finite and strictly positive.")
            record.update(scale=scale, ncrps=record["crps"] / scale)
        return pd.DataFrame([record])
