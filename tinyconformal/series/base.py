# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

import copy
from abc import abstractmethod

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin

from tinyconformal.core.quantiles import (
    temporal_decay_weights,
    weighted_quantile,
)
from tinyconformal.utils.imports import requires_extra
from tinyconformal.utils.inspection import accepts_parameter, call_with_supported_kwargs


class BaseConformalTimeSeriesRegressor(RegressorMixin, BaseEstimator):
    """
    BaseConformalTimeSeriesRegressor

    Multi-Step Conformal Distribution Regressor for Time Series.

    Applies conformal prediction over multi-step horizons for Nixtla-style
    estimators (MLForecast or StatsForecast) using sequential backtesting
    to build empirical nonconformity scores (signed residuals).

    Notes:
    -----
    The conformal distribution approach captures prediction interval bounds by working
    directly with empirical signed residuals defined as:
        residual = y_hat - y_true

    By computing low and high empirical quantiles (q_low, q_high) of these residuals across
    calibration windows, the prediction bounds are derived by inverting the nonconformity score:\n
        lower_bound = y_hat - q_high\n
        upper_bound = y_hat - q_low

    This directly adjusts the point forecast for asymmetric bias and variance per horizon step.
    """

    def __init__(
        self,
        learner: BaseEstimator,
    ):
        """Initialize the conformal wrapper with an unfitted Nixtla learner."""
        self.learner = learner

        self.exog_cols_ = []
        self.static_features_ = []
        self.ncscores_ = None
        self.n = 0
        self.calibration_weights_ = None
        self._quantile_warning_registry = set()

    def _prepare_and_validate_steps(
        self, df: pd.DataFrame, step_size: int
    ) -> tuple[np.ndarray, int]:
        """Validate calibration length and return the shared temporal grid."""
        time_steps = pd.unique(df[self.time_col])
        total_steps = len(time_steps)
        required_steps = self.horizon + (self.n_windows - 1) * step_size
        if total_steps <= required_steps:
            raise ValueError(
                f"Time series has {total_steps} unique time steps, but "
                f"n_windows={self.n_windows}, horizon={self.horizon}, step_size={step_size} "
                f"requires at least {required_steps + 1} steps."
            )
        return time_steps, total_steps

    def _split_train_val_window(
        self,
        df: pd.DataFrame,
        time_steps: np.ndarray,
        total_steps: int,
        w: int,
        step_size: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split one rolling-origin window while preserving panel order."""
        val_end_idx = total_steps - w * step_size
        val_start_idx = val_end_idx - self.horizon
        train_cutoff = time_steps[val_start_idx - 1]
        val_times = time_steps[val_start_idx:val_end_idx]
        train_df = df[df[self.time_col] <= train_cutoff].reset_index(drop=True)
        val_df = df[df[self.time_col].isin(val_times)].reset_index(drop=True)
        return train_df, val_df

    def _fit_predict_window(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        static_features: list | None = None,
    ) -> pd.DataFrame:
        """Fit an isolated forecaster and predict one validation window."""
        learner = copy.deepcopy(self.learner)
        self._fit_forecaster(learner, train_df, static_features=static_features)
        predict_cols = [self.id_col, self.time_col, *self.exog_cols_]
        X_val = val_df[predict_cols] if self.exog_cols_ else None
        return call_with_supported_kwargs(learner.predict, h=self.horizon, X_df=X_val)

    @abstractmethod
    def _compute_window_residuals(
        self,
        df: pd.DataFrame,
        *args,
        **kwargs,
    ) -> None:
        """Calculates nonconformity scores for the predictions and updates the residuals dictionary."""

    def _compute_qhat(
        self,
        ncscore: np.ndarray,
        q_level: float,
        axis: int | None = None,
        weights: np.ndarray | None = None,
    ):
        """Compute an unweighted or temporally weighted calibration quantile."""
        if not self.nexcp:
            return np.quantile(ncscore, q_level, method="higher", axis=axis)
        if weights is None:
            weights = temporal_decay_weights(ncscore.shape[0], self.decay)
        return weighted_quantile(ncscore, q_level, weights, axis=axis)

    def _validate_nexcp(self) -> None:
        """Validate shared NexCP-style temporal weighting parameters."""
        if not isinstance(self.nexcp, (bool, np.bool_)):
            raise TypeError("nexcp must be a boolean.")
        if not isinstance(self.weighted_refit, (bool, np.bool_)):
            raise TypeError("weighted_refit must be a boolean.")
        if self.nexcp:
            temporal_decay_weights(1, self.decay)

    def _fit_forecaster(self, learner, df, static_features=None) -> None:
        """Fit a Nixtla learner, optionally applying NexCP recency weights."""
        fit_df = df
        weight_col = None
        if self.nexcp and self.weighted_refit:
            if not accepts_parameter(learner.fit, "weight_col"):
                raise TypeError(
                    f"{type(learner).__name__}.fit must accept weight_col when "
                    "nexcp=True and weighted_refit=True."
                )
            weight_col = "_tinyconformal_weight"
            if weight_col in df.columns:
                raise ValueError(f"Training data already contains '{weight_col}'.")
            times = pd.unique(df[self.time_col])
            weights = temporal_decay_weights(len(times), self.decay)
            time_weights = dict(zip(times, weights))
            fit_df = df.copy()
            fit_df[weight_col] = fit_df[self.time_col].map(time_weights)
        call_with_supported_kwargs(
            learner.fit,
            df=fit_df,
            id_col=self.id_col,
            time_col=self.time_col,
            target_col=self.target_col,
            static_features=static_features,
            weight_col=weight_col,
        )

    def _infer_model_cols(self, df: pd.DataFrame) -> list[str]:
        """
        Dynamically infers model prediction columns from the output DataFrame.
        """
        excluded = {self.id_col, self.time_col, *self.exog_cols_}
        model_cols = [c for c in df.columns if c not in excluded]

        if not model_cols:
            raise ValueError(
                "Could not infer any prediction model column from the returned DataFrame."
            )
        return model_cols

    def _validate_fit_configuration(self) -> None:
        """Validate subclass-specific calibration configuration before fitting."""
        self._validate_nexcp()

    def _get_horizon(self, h: int | None = None) -> int:
        """Helper to retrieve and validate the forecast horizon."""
        h = h if h is not None else self.horizon
        if not isinstance(h, (int, np.integer)) or isinstance(h, bool) or h <= 0:
            raise ValueError("Forecast horizon h must be a positive integer.")
        if h > self.horizon:
            raise ValueError(
                f"Requested forecast horizon h={h} exceeds fitted calibration horizon ({self.horizon})."
            )
        return int(h)

    def _check_is_fitted(self) -> None:
        """Raise a clear error when prediction is attempted before calibration."""
        if not self.ncscores_ or self.n <= 0:
            raise RuntimeError(
                "This conformal regressor must be fitted before prediction."
            )

    def _generate_forecast(
        self, h: int | None, X_df: pd.DataFrame | None
    ) -> tuple[pd.DataFrame, int, pd.DataFrame | None, int]:
        """Predict a forecast panel in the order returned by the learner."""
        h = self._get_horizon(h)
        self._check_is_fitted()
        if X_df is not None:
            X_df = X_df[[self.id_col, self.time_col, *self.exog_cols_]]
        pred_df = call_with_supported_kwargs(self.learner.predict, h=h, X_df=X_df)
        n_series = len(pred_df) // h
        return pred_df, h, X_df, n_series

    def _merge_predictions_with_targets(
        self, pred_df: pd.DataFrame, target_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Attach targets while preserving exactly one match per prediction row."""
        keys = [self.id_col, self.time_col]
        required = [*keys, self.target_col]
        missing = [column for column in required if column not in target_df.columns]
        if missing:
            raise ValueError(
                f"The evaluation DataFrame is missing required columns: {missing}"
            )
        if target_df.duplicated(keys).any():
            raise ValueError(
                "The evaluation DataFrame must contain at most one target per "
                "identifier and time."
            )

        merged = pred_df.merge(
            target_df[required], on=keys, how="left", validate="one_to_one"
        )
        if merged[self.target_col].isna().any():
            raise ValueError(
                "The evaluation DataFrame must contain a target for every "
                "prediction row."
            )
        return merged

    def _require_calibrated_model(self, model: str):
        """Return calibration scores for a forecast model or raise clearly."""
        if model not in self.ncscores_:
            raise ValueError(
                f"Model column '{model}' was not present during calibration. "
                f"Calibrated model columns: {list(self.ncscores_)}"
            )
        return self.ncscores_[model]

    def _sequential_backtesting(
        self,
        df: pd.DataFrame,
        n_series: int,
        step_size: int | None = None,
        static_features: list | None = None,
        n_jobs: int = -1,
    ) -> dict:
        """Executes sequential backtesting across n_windows to extract CQR nonconformity scores."""
        step_size = self.horizon if step_size is None else step_size
        if (
            not isinstance(step_size, (int, np.integer))
            or isinstance(step_size, bool)
            or step_size <= 0
        ):
            raise ValueError("step_size must be a positive integer.")

        time_steps, total_steps = self._prepare_and_validate_steps(df, step_size)

        def process_window(w):
            train_df, val_df = self._split_train_val_window(
                df, time_steps, total_steps, w, step_size
            )

            fcst = self._fit_predict_window(train_df, val_df, static_features)
            return fcst, val_df

        results = Parallel(n_jobs=n_jobs)(
            delayed(process_window)(w) for w in reversed(range(self.n_windows))
        )

        window_scores_by_model = {}
        for fcst, val_df in results:
            self._compute_window_residuals(
                fcst, val_df, n_series, window_scores_by_model
            )

        return window_scores_by_model

    def _stack_window_scores_by_series(
        self,
        window_scores_by_model: dict[str, list[np.ndarray]],
        series_ids: list,
    ) -> dict:
        """Stack window scores into horizon matrices keyed by model and series."""
        return {
            model: {
                series_id: np.vstack([window_scores[row] for window_scores in windows])
                for row, series_id in enumerate(series_ids)
            }
            for model, windows in window_scores_by_model.items()
        }

    @requires_extra("series")
    def fit(
        self,
        df: pd.DataFrame,
        horizon: int,
        n_windows: int = 15,
        step_size: int | None = None,
        static_features: list | None = None,
        nexcp: bool = True,
        decay: float = 0.99,
        weighted_refit: bool = True,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        n_jobs: int = -1,
    ):
        """
        Fits the conformal regressor by extracting nonconformity scores across backtest
        windows and training the final base learner on the complete dataset.

        Parameters:
        ----------
        df : pd.DataFrame
            The target and feature time series dataset.
        horizon : int
            Maximum forecast horizon calibrated by each backtesting window.
        n_windows : int, default=15
            Number of rolling-origin windows used for calibration.
        step_size : int, optional
            Step size between calibration windows. If None, defaults to `self.horizon`.
        static_features : list, optional
            Columns that identify time-invariant properties of each series. They are
            passed to the base learner during fitting and excluded from future
            dynamic exogenous inputs. Every other non-structural column is treated
            as a dynamic exogenous feature.
        nexcp : bool, default=True
            Apply exponential recency weights to calibration windows.
        decay : float, default=0.99
            Recency decay factor in ``(0, 1)`` when ``nexcp=True``.
        weighted_refit : bool, default=True
            Pass compatible recency weights to learner fits when ``nexcp=True``.
        id_col : str, default="unique_id"
            Series identifier column.
        time_col : str, default="ds"
            Timestamp column.
        target_col : str, default="y"
            Target column.
        n_jobs : int, default=-1
            Number of parallel jobs used to process calibration windows.

        Notes
        -----
        Prediction horizons must not exceed the horizon calibrated here. If the
        requested coverage is unattainable for the finite calibration sample, the
        conformal rank is clipped to the observed scores and a ``RuntimeWarning``
        is emitted.

        Returns:
        -------
        self : MultiStepConformalTimeSeriesRegressor
            Fitted instance of the conformal regressor.
        """

        self.horizon = horizon
        self.n_windows = n_windows
        self.nexcp = nexcp
        self.decay = decay
        self.weighted_refit = weighted_refit
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col

        self._get_horizon()
        self._validate_fit_configuration()
        if (
            not isinstance(self.n_windows, (int, np.integer))
            or isinstance(self.n_windows, bool)
            or self.n_windows <= 0
        ):
            raise ValueError("n_windows must be a positive integer.")

        self.static_features_ = [] if static_features is None else list(static_features)

        self.exog_cols_ = [
            col
            for col in df.columns
            if col
            not in (
                self.id_col,
                self.time_col,
                self.target_col,
                *self.static_features_,
            )
        ]

        series_ids = pd.unique(df[self.id_col]).tolist()
        window_scores_by_model = self._sequential_backtesting(
            df,
            step_size=step_size,
            static_features=self.static_features_ or None,
            n_jobs=n_jobs,
            n_series=len(series_ids),
        )

        self.ncscores_ = self._stack_window_scores_by_series(
            window_scores_by_model, series_ids
        )
        if not self.ncscores_:
            raise RuntimeError(
                f"No nonconformity scores were extracted during backtesting. "
                f"Verify that 'n_windows' ({self.n_windows}) and 'horizon' ({self.horizon}) "
                f"are compatible with the time series length."
            )

        first_model = next(iter(self.ncscores_))
        first_series_scores = next(iter(self.ncscores_[first_model].values()))
        self.n = len(first_series_scores)
        self.calibration_weights_ = (
            temporal_decay_weights(self.n, self.decay) if self.nexcp else None
        )

        self._fit_forecaster(
            self.learner, df, static_features=self.static_features_ or None
        )

        return self
