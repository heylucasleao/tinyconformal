# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from tinyconformal.distribution.base import (
    DiscretePredictiveDistribution,
    PredictiveDistribution,
)
from tinyconformal.utils.imports import requires_extra

from .mscp import ConformalDistributionTimeSeriesRegressor


def _validate_matrix(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError(f"{name} must be a non-empty two-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


class HorizonConformalDistribution(PredictiveDistribution):
    """Predictive distributions calibrated separately by forecast horizon.

    Parameters
    ----------
    locations : ndarray of shape (n_predictions,)
        Point forecasts in sorted Nixtla panel order.
    residuals : ndarray of shape (n_calibration_trajectories, horizon)
        Signed residuals ``y - y_hat`` from sequential backtesting.
    horizon_steps : ndarray of shape (n_predictions,)
        Zero-based horizon index associated with every point forecast.
    """

    def __init__(self, locations, residuals, horizon_steps):
        self.locations = np.asarray(locations, dtype=float)
        if self.locations.ndim != 1 or not np.all(np.isfinite(self.locations)):
            raise ValueError("locations must be a finite one-dimensional array.")
        self.residuals = np.sort(_validate_matrix(residuals, "residuals"), axis=0)
        self.horizon_steps = np.asarray(horizon_steps, dtype=int)
        if self.horizon_steps.shape != self.locations.shape:
            raise ValueError("horizon_steps and locations must have the same shape.")
        if np.any(
            (self.horizon_steps < 0)
            | (self.horizon_steps >= self.residuals.shape[1])
        ):
            raise ValueError("horizon_steps contains an uncalibrated horizon index.")

    def __len__(self) -> int:
        return self.locations.size

    def _rowwise_or_grid(self, values, name: str) -> tuple[np.ndarray, bool]:
        array = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")
        if array.ndim == 0:
            return np.full((len(self), 1), float(array)), True
        if array.ndim == 1:
            if array.size == len(self):
                return array[:, None], True
            return np.broadcast_to(array[None, :], (len(self), array.size)), False
        if array.ndim == 2 and array.shape[0] == len(self):
            return array, False
        raise ValueError(
            f"{name} must be a scalar, a one-dimensional grid, a row-wise vector "
            f"of length {len(self)}, or a matrix with {len(self)} rows."
        )

    def _row_residuals(self) -> np.ndarray:
        return self.residuals[:, self.horizon_steps].T

    def cdf(self, values):
        values, squeeze = self._rowwise_or_grid(values, "values")
        scores = values - self.locations[:, None]
        row_residuals = self._row_residuals()
        ranks = np.sum(
            row_residuals[:, :, None] <= scores[:, None, :], axis=1
        )
        n = self.residuals.shape[0]
        result = ranks.astype(float) / (n + 1)
        result[ranks == n] = 1.0
        return result[:, 0] if squeeze else result

    def ppf(self, quantiles):
        quantiles, squeeze = self._rowwise_or_grid(quantiles, "quantiles")
        if np.any((quantiles < 0.0) | (quantiles > 1.0)):
            raise ValueError("quantiles must lie in [0, 1].")
        n = self.residuals.shape[0]
        ranks = np.ceil((n + 1) * quantiles).astype(int)
        ranks = np.clip(ranks, 1, n) - 1
        selected = np.take_along_axis(self._row_residuals(), ranks, axis=1)
        result = self.locations[:, None] + selected
        return result[:, 0] if squeeze else result


class DiscreteHorizonConformalDistribution(
    HorizonConformalDistribution, DiscretePredictiveDistribution
):
    """Horizon-wise conformal distribution on an ordered integer support."""

    def __init__(self, locations, residuals, horizon_steps, minimum: int | None = 0):
        super().__init__(locations, residuals, horizon_steps)
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
        if values.ndim == 1 and values.size == len(self):
            return np.where(below, 0.0, result)
        return np.where(np.broadcast_to(below, np.shape(result)), 0.0, result)


class ConformalPredictiveSystemTimeSeriesRegressor(
    ConformalDistributionTimeSeriesRegressor
):
    """Horizon-wise CPS for Nixtla-compatible forecasting estimators.

    Calibration uses the same sequential rolling-origin backtesting engine as
    MSCP and TSCQR. Signed residual distributions are retained per model and
    horizon instead of being reduced to one interval during prediction.

    ``predict_distribution`` returns a dictionary because Nixtla estimators may
    expose more than one model column. Each value is aligned row-for-row with the
    DataFrame returned alongside it.
    """

    def __init__(
        self,
        learner: BaseEstimator,
        horizon: int,
        n_windows: int = 10,
        alpha: float = 0.05,
        discrete: bool = False,
        minimum: int | None = 0,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ):
        super().__init__(
            learner=learner,
            horizon=horizon,
            n_windows=n_windows,
            alpha=alpha,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
        )
        self.discrete = discrete
        self.minimum = minimum

    def _validate_fit_configuration(self) -> None:
        super()._validate_fit_configuration()
        if not isinstance(self.discrete, (bool, np.bool_)):
            raise TypeError("discrete must be a boolean.")
        if self.discrete and self.minimum is not None and not isinstance(
            self.minimum, (int, np.integer)
        ):
            raise TypeError("minimum must be an integer or None.")

    def fit(self, df, step_size=None, static_features=None, n_jobs=-1):
        if self.discrete:
            self._validate_columns(df)
            target = np.asarray(df[self.target_col], dtype=float)
            if not np.all(np.isfinite(target)) or np.any(target != np.floor(target)):
                raise ValueError("Discrete time-series CPS targets must be finite integers.")
            if self.minimum is not None and np.any(target < self.minimum):
                raise ValueError(
                    f"Discrete time-series CPS targets must be >= {self.minimum}."
                )
        return super().fit(
            df,
            step_size=step_size,
            static_features=static_features,
            n_jobs=n_jobs,
        )

    def _prediction_frame(
        self, h: int | None, X_df: pd.DataFrame | None
    ) -> tuple[pd.DataFrame, list[str], int]:
        h = self._get_horizon(h)
        self._check_is_fitted()
        X_df = self._validate_prediction_features(X_df, h)
        pred_df = (
            self._invoke(self.learner.predict, h=h, X_df=X_df)
            .sort_values([self.id_col, self.time_col])
            .reset_index(drop=True)
        )
        model_cols = self._infer_model_cols(pred_df)
        n_series = self._validate_prediction_panel(pred_df, h)
        return pred_df, model_cols, n_series

    def _build_distributions(
        self, pred_df: pd.DataFrame, model_cols: list[str], h: int, n_series: int
    ) -> dict[str, PredictiveDistribution]:
        horizon_steps = np.tile(np.arange(h), n_series)
        distributions = {}
        for model in model_cols:
            if model not in self.ncscores_:
                raise ValueError(
                    f"Model column '{model}' was not present during calibration. "
                    f"Calibrated model columns: {list(self.ncscores_)}"
                )
            # MSCP stores y_hat - y; predictive distributions use y - y_hat.
            residuals = -self.ncscores_[model][:, :h]
            locations = pred_df[model].to_numpy(dtype=float)
            if self.discrete:
                distributions[model] = DiscreteHorizonConformalDistribution(
                    locations,
                    residuals,
                    horizon_steps,
                    minimum=self.minimum,
                )
            else:
                distributions[model] = HorizonConformalDistribution(
                    locations, residuals, horizon_steps
                )
        return distributions

    @requires_extra("series")
    def predict_distribution(
        self,
        h: int | None = None,
        X_df: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, dict[str, PredictiveDistribution]]:
        """Return the Nixtla forecast frame and aligned CPS distributions."""
        h = self._get_horizon(h)
        pred_df, model_cols, n_series = self._prediction_frame(h, X_df)
        return pred_df, self._build_distributions(
            pred_df, model_cols, h=h, n_series=n_series
        )

    @requires_extra("series")
    def predict_quantiles(
        self,
        quantiles,
        h: int | None = None,
        X_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Return arbitrary CPS quantiles in a Nixtla long-format DataFrame."""
        quantiles = np.asarray(quantiles, dtype=float)
        if quantiles.ndim == 0:
            quantiles = quantiles.reshape(1)
        if quantiles.ndim != 1 or quantiles.size == 0:
            raise ValueError("quantiles must be a non-empty scalar or 1D sequence.")
        if not np.all(np.isfinite(quantiles)) or np.any(
            (quantiles < 0.0) | (quantiles > 1.0)
        ):
            raise ValueError("quantiles must contain finite values in [0, 1].")

        pred_df, distributions = self.predict_distribution(h=h, X_df=X_df)
        for model, distribution in distributions.items():
            quantile_matrix = np.broadcast_to(
                quantiles, (len(distribution), quantiles.size)
            )
            values = distribution.ppf(quantile_matrix)
            for index, quantile in enumerate(quantiles):
                label = np.format_float_positional(
                    quantile * 100.0, precision=12, trim="-"
                )
                pred_df[f"{model}-q-{label}"] = values[:, index]
        return pred_df

    @requires_extra("series")
    def predict_interval(
        self,
        h: int | None = None,
        X_df: pd.DataFrame | None = None,
        alpha: float | None = None,
    ) -> pd.DataFrame:
        """Return equal-tailed CPS intervals in the MSCP column convention."""
        alpha = self._get_alpha(alpha)
        pred_df, distributions = self.predict_distribution(h=h, X_df=X_df)
        level = self._coverage_label(alpha)
        for model, distribution in distributions.items():
            bounds = distribution.interval(1.0 - alpha)
            pred_df[f"{model}-lo-{level}"] = bounds[:, 0]
            pred_df[f"{model}-hi-{level}"] = bounds[:, 1]
        return pred_df


class ContinuousConformalPredictiveSystemTimeSeriesRegressor(
    ConformalPredictiveSystemTimeSeriesRegressor
):
    """Continuous-target horizon-wise CPS for Nixtla estimators."""

    def __init__(
        self,
        learner: BaseEstimator,
        horizon: int,
        n_windows: int = 10,
        alpha: float = 0.05,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ):
        super().__init__(
            learner=learner,
            horizon=horizon,
            n_windows=n_windows,
            alpha=alpha,
            discrete=False,
            minimum=None,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
        )


class DiscreteConformalPredictiveSystemTimeSeriesRegressor(
    ConformalPredictiveSystemTimeSeriesRegressor
):
    """Integer-target horizon-wise CPS for Nixtla estimators."""

    def __init__(
        self,
        learner: BaseEstimator,
        horizon: int,
        n_windows: int = 10,
        alpha: float = 0.05,
        minimum: int | None = 0,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ):
        super().__init__(
            learner=learner,
            horizon=horizon,
            n_windows=n_windows,
            alpha=alpha,
            discrete=True,
            minimum=minimum,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
        )
