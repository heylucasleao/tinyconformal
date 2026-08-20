# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import mean_absolute_error


class BaseTimeSeriesConformalRegressor(ABC):
    """
    Base class for Time Series Conformal Prediction wrapping Nixtla estimators
    (StatsForecast / MLForecast) using Temporal Cross-Validation (Backtesting).
    """

    def __init__(
        self,
        learner: BaseEstimator,
        horizon: int,
        n_windows: int = 3,
        alpha: float = 0.05,
    ):
        """
        Parameters
        ----------
        learner : BaseEstimator
            Unfitted or fitted Nixtla-compatible estimator (StatsForecast or MLForecast).
        horizon : int
            Forecast horizon step count (H).
        n_windows : int, default=3
            Number of backtesting rolling windows used to extract calibration residuals.
        alpha : float, default=0.05
            Significance level for conformal prediction.
        """
        self.learner = learner
        self.alpha = alpha
        self.horizon = horizon
        self.n_windows = n_windows
        self.residuals_ = None
        self.ncscore = None
        self.n = None

        self.target_col = None
        self.time_col = None
        self.id_col = None
        self.model_col = None

    @abstractmethod
    def _generate_residuals(
        self, preds_val: np.ndarray, y_val_arr: np.ndarray
    ) -> np.ndarray:
        """
        Computes nonconformity scores/residuals from predictions and true targets.
        To be implemented by subclasses.
        """
        pass

    @abstractmethod
    def fit(self, df: pd.DataFrame, *args, **kwargs):
        """Fits the conformal model."""
        pass

    @abstractmethod
    def predict_interval(self, *args, **kwargs) -> np.ndarray:
        """
        Generates prediction intervals for the input data.
        To be implemented by subclasses.
        """
        pass

    def _compute_qhat(self, ncscore: np.ndarray, q_level: float, axis: int = None):
        """
        Computes the q-hat quantile value based on nonconformity scores and target level.
        """
        return np.quantile(ncscore, q_level, method="higher", axis=axis)

    def _get_alpha(self, alpha: Optional[float] = None) -> float:
        """Helper to retrieve active alpha value."""
        return alpha if alpha is not None else self.alpha

    def _coverage_rate(self, y_true: np.ndarray, y_pred_intervals: np.ndarray) -> float:
        """Evaluates empirical interval coverage rate."""
        lower, upper = y_pred_intervals[..., 0], y_pred_intervals[..., 1]
        coverages = (y_true >= lower) & (y_true <= upper)
        return float(np.mean(coverages))

    def _interval_width_mean(self, y_pred_intervals: np.ndarray) -> float:
        """Calculates mean interval width."""
        widths = y_pred_intervals[..., 1] - y_pred_intervals[..., 0]
        return float(np.mean(widths))

    def _mwi_score(
        self, y_true: np.ndarray, y_pred_intervals: np.ndarray, alpha: float
    ) -> float:
        """
        Calculates the Mean Winkler Interval Score (MWIS) for prediction intervals.

        Parameters
        ----------
        y_true : ndarray
            True target values.
        y_pred_intervals : ndarray
            Prediction intervals array where the last axis contains [lower_bound, upper_bound].
        alpha : float
            Significance level, where (1 - alpha) represents the target coverage rate.

        Returns
        -------
        score : float
            The computed Mean Winkler Interval Score across all predictions.
        """
        lower, upper = y_pred_intervals[..., 0], y_pred_intervals[..., 1]
        width = upper - lower
        penalty_lower = (2.0 / alpha) * (lower - y_true) * (y_true < lower)
        penalty_upper = (2.0 / alpha) * (y_true - upper) * (y_true > upper)
        return float(np.mean(width + penalty_lower + penalty_upper))

    def _extract_predictions(self, fcst_df: pd.DataFrame) -> np.ndarray:
        """
        Pivots Nixtla long-format DataFrame prediction into a 2D NumPy array.
        Ensures strict row and column alignment sorting.
        """
        if self.model_col is None:
            model_cols = [
                c for c in fcst_df.columns if c not in [self.id_col, self.time_col]
            ]
            if not model_cols:
                raise ValueError(
                    "No prediction model column was detected in the model output DataFrame."
                )
            self.model_col = model_cols[0]

        pivoted = fcst_df.pivot(
            index=self.id_col, columns=self.time_col, values=self.model_col
        )
        pivoted = pivoted.sort_index(axis=0).sort_index(axis=1)
        return pivoted.values

    def _extract_target(self, target_df: pd.DataFrame) -> np.ndarray:
        """
        Pivots ground-truth DataFrames into a 2D NumPy array (n_series, horizon).
        Ensures strict row and column alignment sorting matching predictions.
        """
        pivoted = target_df.pivot(
            index=self.id_col, columns=self.time_col, values=self.target_col
        )
        pivoted = pivoted.sort_index(axis=0).sort_index(axis=1)
        return pivoted.values

    def predict(
        self,
        X_df: Optional[pd.DataFrame] = None,
        h: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """
        Generates point predictions as the center of prediction intervals.
        """
        intervals = self.predict_interval(X_df=X_df, h=h, alpha=alpha)
        return np.mean(intervals, axis=-1)

    def evaluate(
        self,
        df_test: pd.DataFrame,
        h: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates the regressor performance on the given dataset.

        Parameters
        ----------
        df_test : pd.DataFrame
            Evaluation dataset containing input features and target values.
        h : int, optional
            Forecast horizon for multi-step predictions.
        alpha : float, optional
            Significance level for prediction intervals.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing metrics (coverage_rate, interval_width_mean, mwis, mae, mbe, mse).
        """
        alpha = self._get_alpha(alpha)
        y_true = self._extract_target(df_test)

        y_pred_intervals = self.predict_interval(X_df=df_test, h=h, alpha=alpha)
        y_pred = np.mean(y_pred_intervals, axis=-1)

        def rounded(val):
            return float(np.round(val, 3))

        return {
            "total": len(df_test),
            "alpha": alpha,
            "coverage_rate": rounded(self._coverage_rate(y_true, y_pred_intervals)),
            "interval_width_mean": rounded(self._interval_width_mean(y_pred_intervals)),
            "mwis": rounded(self._mwi_score(y_true, y_pred_intervals, alpha)),
            "mae": rounded(mean_absolute_error(y_true.ravel(), y_pred.ravel())),
            "mbe": rounded(np.mean(y_pred - y_true)),
            "mse": rounded(np.mean((y_pred - y_true) ** 2)),
        }
