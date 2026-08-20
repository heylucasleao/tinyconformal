# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import List, Union, Any, Dict, Optional
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
        self.beta = None

        self.target_col = None
        self.time_col = None
        self.id_col = None
        self.model_col = None

    @abstractmethod
    def fit(self, *args, **kwargs):
        """Fits the conformal model."""
        pass

    @abstractmethod
    def predict_interval(self, *args, **kwargs) -> np.ndarray:
        """
        Generate prediction intervals for the input data.
        To be implemented by subclasses.
        """
        pass

    def _compute_qhat(self, ncscore: np.ndarray, q_level: float, axis: int = None):
        """
        Compute the q-hat value based on the nonconformity scores and the quantile level.
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

        If the target observation falls inside the prediction interval, the score equal
        to the interval width (narrower intervals are rewarded). If the observation falls
        outside, a penalty proportional to the distance from the nearest bound is added.

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

    def _extract_predictions(self, preds: pd.DataFrame) -> np.ndarray:
        """
        Pivots Nixtla long-format DataFrame prediction into a 2D/3D NumPy array.
        """
        target_models = (
            [self.model_col]
            if isinstance(self.model_col, str)
            else list(self.model_col)
        )

        if len(target_models) == 1:
            return preds.pivot(
                index=self.id_col, columns=self.time_col, values=target_models[0]
            ).to_numpy()

        return np.stack(
            [
                preds.pivot(
                    index=self.id_col, columns=self.time_col, values=col
                ).to_numpy()
                for col in target_models
            ],
            axis=-1,
        )

    def _extract_target(self, y: pd.DataFrame) -> np.ndarray:
        """
        Pivots Nixtla target DataFrame into a 2D NumPy array (n_series, horizon).
        """
        return y.pivot(
            index=self.id_col, columns=self.time_col, values=self.target_col
        ).to_numpy()

    def _sequential_backtesting(self, df: pd.DataFrame, step_size: int = None):
        """
        Executes sequential backtesting rolling windows over the input DataFrame.
        """
        step_size = step_size if step_size is not None else self.horizon
        total_len = len(df[self.time_col].unique())
        min_required = self.horizon * self.n_windows

        if total_len <= min_required:
            raise ValueError(
                f"Data timeline length ({total_len}) must be greater than required window capacity ({min_required})."
            )

        residuals: List[np.ndarray] = []

        for w in reversed(range(self.n_windows)):
            cutoff_end = total_len - (w * step_size)
            train_end = cutoff_end - self.horizon

            time_steps = sorted(df[self.time_col].unique())
            train_times = time_steps[:train_end]
            val_times = time_steps[train_end:cutoff_end]

            df_tr = df[df[self.time_col].isin(train_times)]
            df_val = df[df[self.time_col].isin(val_times)]

            temp_model = clone(self.learner)
            temp_model.fit(df_tr)

            preds_val = self._extract_predictions(temp_model.predict(h=self.horizon))
            y_val_arr = self._extract_target(df_val)

            residuals.append(y_val_arr - preds_val)

        return residuals

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str,
        time_col: str,
        id_col: str,
        model_col: Union[str, List[str]],
        step_size: int = None,
    ):
        """
        Fits the Nixtla model and computes conformal prediction residuals via backtesting.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Input 'df' must be a pandas DataFrame. Got {type(df)}.")

        for col_name, col_val in [
            ("target_col", target_col),
            ("time_col", time_col),
            ("id_col", id_col),
        ]:
            if col_val not in df.columns:
                raise KeyError(
                    f"Column '{col_val}' specified for '{col_name}' was not found in DataFrame."
                )

        self.target_col = target_col
        self.time_col = time_col
        self.id_col = id_col
        self.model_col = model_col

        residuals = self._sequential_backtesting(df, step_size=step_size)

        self.ncscore = np.vstack(residuals)
        self.learner.fit(df)
        self.n = len(self.ncscore)

        return self

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
        Evaluate the performance the regressor on the given dataset.
        Parameters:
            df_test:
                The evaluation dataset containing both input features and target values.
            h:
                The forecast horizon for multi-step predictions.
            alpha:
                Significance level for prediction intervals. If None, the regressor's default alpha is used.
        Returns:
            A dictionary containing the following evaluation metrics:
            - "total" (int): The total number of samples in the dataset.
            - "alpha" (float): The significance level used for evaluation.
            - "beta" (float): Base model error rate (if `unlabeled_fit` was used)
            - "coverage_rate" (float): The coverage rate of the prediction intervals.
            - "interval_width_mean" (float): The mean width of the prediction intervals.
            - "mwis" (float): The Mean Weighted Interval Score (MWIS).
            - "mae" (float): The Mean Absolute Error (MAE) of the predictions.
            - "mbe" (float): The Mean Bias Error (MBE) of the predictions.
            - "mse" (float): The Mean Squared Error (MSE) of the predictions.
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
            "beta": self.beta,
            "coverage_rate": rounded(self._coverage_rate(y_true, y_pred_intervals)),
            "interval_width_mean": rounded(self._interval_width_mean(y_pred_intervals)),
            "mwis": rounded(self._mwi_score(y_true, y_pred_intervals, alpha)),
            "mae": rounded(mean_absolute_error(y_true.ravel(), y_pred.ravel())),
            "mbe": rounded(np.mean(y_pred - y_true)),
            "mse": rounded(np.mean((y_pred - y_true) ** 2)),
        }
