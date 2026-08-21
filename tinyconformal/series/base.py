# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error
import inspect
import copy


class BaseTimeSeriesConformalRegressor(ABC):
    """
    BaseTimeSeriesConformalRegressor

    Base class for Time Series Conformal Prediction wrapping Nixtla estimators
    (StatsForecast / MLForecast) using Temporal Cross-Validation (Backtesting).

    Time series conformal regressors provide valid prediction intervals for time series
    forecasting models by estimating calibration residuals over rolling windows.
    """

    def __init__(
        self,
        learner: BaseEstimator,
        horizon: int,
        n_windows: int = 10,
        alpha: float = 0.05,
    ):
        """
        Initializes the time series regressor with a learner, horizon, and backtesting configuration.

        Parameters:
        ----------
        learner : BaseEstimator
            Unfitted or fitted Nixtla-compatible estimator (StatsForecast or MLForecast).
        horizon : int
            Forecast horizon step count (H).
        n_windows : int, default=10
            Number of backtesting rolling windows used to extract calibration residuals.
        alpha : float, default=0.05
            Significance level applied in the regressor.

        Attributes:
        ----------
        learner : BaseEstimator
            The base learner employed in the regressor.
        horizon : int
            Forecast horizon step count.
        n_windows : int
            Number of rolling backtesting windows.
        alpha : float
            Significance level applied in the regressor.
        residuals_ : array-like, default=None
            Extracted nonconformity scores/residuals from backtesting windows.
        n : int, default=None
            Number of calibration samples.
        target_col : str, default=None
            Name of the target column in the input DataFrame.
        time_col : str, default=None
            Name of the timestamp/time column in the input DataFrame.
        id_col : str, default=None
            Name of the unique identifier column in the input DataFrame.
        exog_cols_ : list of str, default=None
            Exogenous feature columns present in the dataset.
        """

        self.learner = learner
        self.alpha = alpha
        self.horizon = horizon
        self.n_windows = n_windows
        self.residuals_ = None
        self.n = None

        self.target_col = None
        self.time_col = None
        self.id_col = None
        self.exog_cols_ = None

    @abstractmethod
    def _generate_residuals(
        self, preds_val: np.ndarray, y_val_arr: np.ndarray
    ) -> np.ndarray:
        """
        Computes nonconformity scores or residuals from predictions and true targets.
        To be implemented by subclasses.
        """
        pass

    @abstractmethod
    def fit(self, df: pd.DataFrame, *args, **kwargs):
        """
        Fits the conformal model to the time series dataset.
        To be implemented by subclasses.
        """
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
        Compute the q-hat quantile value based on nonconformity scores and the quantile level.
        """
        return np.quantile(ncscore, q_level, method="higher", axis=axis)

    def _infer_model_cols(self, df: pd.DataFrame) -> List[str]:
        """
        Dynamically infers model prediction columns from the output DataFrame.
        """
        if self.model_col_ is not None:
            return (
                [self.model_col_]
                if isinstance(self.model_col_, str)
                else self.model_col_
            )

        excluded = {self.id_col, self.time_col, *self.exog_cols_}
        model_cols = [c for c in df.columns if c not in excluded]

        if not model_cols:
            raise ValueError(
                "Could not infer any prediction model column from the returned DataFrame."
            )
        return model_cols

    def _get_alpha(self, alpha: Optional[float] = None) -> float:
        """Helper to retrieve the active significance level (alpha)."""
        return alpha if alpha is not None else self.alpha

    def _get_horizon(self, h: Optional[float] = None) -> float:
        """Helper to retrieve and validate the forecast horizon."""
        h = h if h is not None else self.horizon
        if h > self.horizon:
            raise ValueError(
                f"Requested forecast horizon h={h} exceeds fitted calibration horizon ({self.horizon})."
            )
        return h

    def _invoke(self, method, **kwargs):
        """
        Executes a callable (fit, predict, cross_validation) injecting only
        the parameters accepted by its signature.
        """
        sig = inspect.signature(method)
        has_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )

        if has_var_kw:
            filtered = {k: v for k, v in kwargs.items() if v is not None}
        else:
            filtered = {
                k: v for k, v in kwargs.items() if k in sig.parameters and v is not None
            }

        return method(**filtered)

    def _sequential_backtesting(
        self,
        df: pd.DataFrame,
        step_size: int = None,
        static_features: list = None,
    ) -> list:
        """
        Executes sequential rolling-window backtesting across calibration windows.

        Parameters:
        ----------
        df : pd.DataFrame
            The calibration DataFrame containing time series values.
        step_size : int, optional
            Step size for window shift. Defaults to self.horizon if None.
        static_features : list, optional
            List of static feature column names.

        Returns:
        -------
        dict
            Dictionary containing list of 2D residual matrices per model extracted
            from backtesting windows.
        """
        residuals_by_model: Dict[List] = {}

        if step_size is None:
            step_size = self.horizon

        df = df.sort_values(by=[self.id_col, self.time_col]).reset_index(drop=True)

        unique_times = np.sort(df[self.time_col].unique())
        total_times = len(unique_times)

        for w in reversed(range(self.n_windows)):
            val_end_idx = total_times - (w * step_size)
            val_start_idx = val_end_idx - self.horizon

            if val_start_idx <= 0:
                raise ValueError(
                    f"Time series length is too short for the specified n_windows ({self.n_windows}) "
                    f"and horizon ({self.horizon})."
                )

            train_cutoff = unique_times[val_start_idx - 1]
            val_cutoff = unique_times[val_end_idx - 1]

            train_df = df[df[self.time_col] <= train_cutoff].copy()
            val_df = df[
                (df[self.time_col] > train_cutoff) & (df[self.time_col] <= val_cutoff)
            ].copy()

            temp_model = copy.deepcopy(self.learner)

            self._invoke(
                temp_model.fit,
                df=train_df,
                id_col=self.id_col,
                time_col=self.time_col,
                target_col=self.target_col,
                static_features=static_features,
            )

            predict_cols = [self.id_col, self.time_col] + self.exog_cols_
            X_val = val_df[predict_cols] if self.exog_cols_ else None

            fcst = self._invoke(
                temp_model.predict,
                h=self.horizon,
                X_df=X_val,
            )

            model_cols = self._infer_model_cols(fcst)
            y_true = self._extract_target(val_df)
            n_series = fcst[self.id_col].nunique()

            for model in model_cols:
                y_hat = fcst[model].to_numpy().reshape(n_series, self.horizon)
                r = self._generate_residuals(y_hat, y_true)

                if model not in residuals_by_model:
                    residuals_by_model[model] = []
                residuals_by_model[model].append(r)

        return residuals_by_model

    def _coverage_rate(
        self, y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
    ) -> float:
        """
        Evaluate empirical coverage of prediction intervals.
        """
        coverages = (y_true >= lower) & (y_true <= upper)
        return float(np.mean(coverages))

    def _interval_width_mean(self, lower: np.ndarray, upper: np.ndarray) -> float:
        """
        Calculates the mean width of the prediction intervals.
        """
        widths = upper - lower
        return float(np.mean(widths))

    def _mwi_score(
        self,
        y_true: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        alpha: float,
    ) -> float:
        """
        Calculate the Winkler interval score for prediction intervals.

        If the observation falls outside the prediction interval, the score increases
        with the distance from the interval bounds.

        If the observation falls inside the prediction interval, the score depends on
        the width of the interval (narrower intervals are better).

        Parameters:
        ----------
        y_true : np.ndarray
            True target values.
        lower : np.ndarray
            Lower bounds of prediction intervals.
        upper : np.ndarray
            Upper bounds of prediction intervals.
        alpha : float
            Significance level, where (1 - alpha) is the desired coverage.

        Returns:
        -------
        float
            The mean Winkler interval score.
        """
        width = upper - lower
        penalty_lower = (2.0 / alpha) * (lower - y_true) * (y_true < lower)
        penalty_upper = (2.0 / alpha) * (y_true - upper) * (y_true > upper)
        return float(np.mean(width + penalty_lower + penalty_upper))

    def _extract_predictions(self, fcst_df: pd.DataFrame) -> np.ndarray:
        """
        Pivots Nixtla long-format DataFrame predictions into a 2D NumPy array.
        Ensures strict row and column alignment sorting.
        """
        if self.model_col_ is None:
            model_cols = [
                c for c in fcst_df.columns if c not in [self.id_col, self.time_col]
            ]
            if not model_cols:
                raise ValueError(
                    "No prediction model column was detected in the model output DataFrame."
                )
            self.model_col_ = model_cols[0]

        pivoted = fcst_df.pivot(
            index=self.id_col, columns=self.time_col, values=self.model_col_
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

    def evaluate(
        self,
        df_test: pd.DataFrame,
        h: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Evaluate the performance of the regressor on the given dataset.

        Parameters:
        ----------
        df_test : pd.DataFrame
            Evaluation dataset containing input features and target values.
        h : int, optional
            Forecast horizon step count. If None, the default fitted horizon is used.
        alpha : float, optional
            Significance level for prediction intervals. If None, default alpha is used.

        Returns:
        -------
        pd.DataFrame
            A DataFrame containing the evaluation metrics for each model and level:
            - "model" (str): Name of the evaluated model.
            - "level" (str): Coverage percentage string (e.g., '95%').
            - "alpha" (float): The significance level used for evaluation.
            - "coverage_rate" (float): The coverage rate of the prediction intervals.
            - "interval_width_mean" (float): The mean width of the prediction intervals.
            - "mwis" (float): The Mean Weighted Interval Score (MWIS).
            - "mae" (float): The Mean Absolute Error (MAE) of the predictions.
            - "mbe" (float): The Mean Bias Error (MBE) of the predictions.
            - "mse" (float): The Mean Squared Error (MSE) of the predictions.
        """
        alpha = self._get_alpha(alpha)

        eval_df = self.predict_interval(X_df=df_test, h=h, alpha=alpha)

        eval_df = eval_df.merge(
            df_test[[self.id_col, self.time_col, self.target_col]],
            on=[self.id_col, self.time_col],
            how="inner",
        )
        y_true = eval_df[self.target_col].to_numpy()
        mask = lambda c: any(pattern in c for pattern in ("-lo-", "-hi-"))
        bounds = [c for c in eval_df.columns if mask(c)]
        lo_cols = [model for model in bounds if "-lo-" in model]
        records = []

        def rounded(value):
            return np.round(value, 3)

        for lo_col in lo_cols:
            model_name, level_str = lo_col.split("-lo-")
            hi_col = f"{model_name}-hi-{level_str}"

            lower = eval_df[lo_col].to_numpy()
            upper = eval_df[hi_col].to_numpy()
            y_pred = eval_df[model_name].to_numpy()
            mae = mean_absolute_error(y_true, y_pred)
            mbe = np.mean(y_pred - y_true)
            mse = np.mean((y_pred - y_true) ** 2)

            records.append(
                {
                    "model": model_name,
                    "level": f"{level_str}%",
                    "alpha": alpha,
                    "coverage_rate": rounded(self._coverage_rate(y_true, lower, upper)),
                    "interval_width_mean": rounded(
                        self._interval_width_mean(lower, upper)
                    ),
                    "mwis": rounded(self._mwi_score(y_true, lower, upper, alpha)),
                    "mae": rounded(mae),
                    "mbe": rounded(mbe),
                    "mse": rounded(mse),
                }
            )

        return pd.DataFrame(records)
