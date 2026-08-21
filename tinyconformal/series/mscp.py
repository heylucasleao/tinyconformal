# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

import copy
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from .base import BaseTimeSeriesConformalRegressor


class ConformalDistributionTimeSeriesRegressor(
    RegressorMixin, BaseEstimator, BaseTimeSeriesConformalRegressor
):
    """Multi-Step Conformal Distribution Regressor for Time Series.

    Applies conformal prediction over multi-step horizons for Nixtla-style
    estimators (MLForecast or StatsForecast) using sequential backtesting
    to build empirical nonconformity scores (residuals).
    """

    def __init__(
        self,
        learner: BaseEstimator,
        horizon: int,
        n_windows: int = 3,
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
        )
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col

        self.model_col_ = None
        self.exog_cols_ = []
        self.ncscores_: Dict[str, np.ndarray] = {}
        self.n = 0

    def _validate_columns(self, df: pd.DataFrame):
        """Validates presence of required structural columns in input DataFrames."""
        required_cols = [self.id_col, self.time_col, self.target_col]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(
                f"The following required columns are missing from the DataFrame: {missing}"
            )

    def _generate_residuals(self, y_hat: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Computes nonconformity scores (residuals) via conformal distribution:

        R_{t,h} = \\hat{y}_{t,h} - y_{t,h}
        """
        return y_hat - y_true

    def _sequential_backtesting(
        self,
        df: pd.DataFrame,
        step_size: Optional[int] = None,
        static_features: Optional[List[str]] = None,
    ) -> Dict[str, List[np.ndarray]]:
        """Executes sequential rolling-window backtesting across calibration windows."""
        residuals_by_model: Dict[str, List[np.ndarray]] = {}

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
                    f"Time series length is too short for specified n_windows ({self.n_windows}) "
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
            y_true_raw = self._extract_target(val_df)
            n_series = fcst[self.id_col].nunique()
            y_true = np.asarray(y_true_raw).reshape(n_series, self.horizon)

            for model in model_cols:
                y_hat = fcst[model].to_numpy().reshape(n_series, self.horizon)
                r = self._generate_residuals(y_hat, y_true)

                if model not in residuals_by_model:
                    residuals_by_model[model] = []
                residuals_by_model[model].append(r)

        return residuals_by_model

    def fit(
        self,
        df: pd.DataFrame,
        step_size: Optional[int] = None,
        static_features: Optional[List[str]] = None,
    ):
        """Fits the conformal regressor by extracting nonconformity scores across backtest windows."""
        self._validate_columns(df)

        self.exog_cols_ = [
            col
            for col in df.columns
            if col not in (self.id_col, self.time_col, self.target_col)
        ]

        residuals_by_model = self._sequential_backtesting(
            df,
            step_size=step_size,
            static_features=static_features,
        )

        self.ncscores_ = {
            model: np.vstack(res_list) for model, res_list in residuals_by_model.items()
        }

        if not self.ncscores_:
            raise RuntimeError(
                f"No nonconformity scores were extracted during backtesting. "
                f"Verify that 'n_windows' ({self.n_windows}) and 'horizon' ({self.horizon}) "
                f"are compatible with the time series length in the provided DataFrame."
            )

        self._invoke(
            self.learner.fit,
            df=df,
            id_col=self.id_col,
            time_col=self.time_col,
            target_col=self.target_col,
            static_features=static_features,
        )

        return self

    def _sample_correction(self, alpha: float) -> Tuple[float, float]:
        """Computes finite-sample quantile adjustment for exact coverage bounds."""
        n = self.n
        low_q = max(0.0, alpha / 2.0 - 1.0 / (2.0 * n))
        high_q = min(1.0, 1.0 - alpha / 2.0 + 1.0 / (2.0 * n))
        return low_q, high_q

    def _predict_raw(
        self,
        h: Optional[int] = None,
        X_df: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """Generates base model point predictions into standard ndarray."""
        h = h if h is not None else self.horizon

        preds_df = self._invoke(
            self.learner.predict,
            h=h,
            X_df=X_df,
        )
        return self._extract_predictions(preds_df)

    def _get_conformal_distribution(
        self,
        h: Optional[int] = None,
        X_df: Optional[pd.DataFrame] = None,
        model_name: Optional[str] = None,
    ) -> np.ndarray:
        """Generates the 3D empirical trajectory tensor: shape (n_series, n_residuals, horizon)."""
        h = h if h is not None else self.horizon

        if h > self.horizon:
            raise ValueError(
                f"Requested forecast horizon h={h} exceeds fitted calibration horizon ({self.horizon})."
            )

        preds = self._predict_raw(h=h, X_df=X_df)  # shape: (n_series, horizon)

        target_model = model_name or next(iter(self.ncscores_))
        ncscore_sliced = self.ncscores_[target_model][:, :h]

        # Inversão R = y_hat - y => y = y_hat - R
        return preds[:, np.newaxis, :] - ncscore_sliced[np.newaxis, :, :]

    def _compute_bounds(
        self,
        y_hat: np.ndarray,
        model_name: str,
        h: int,
        n_series: int,
        alpha: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calcula os limites inferior e superior conformais sobre vetores 1D."""
        alpha = self._get_alpha(alpha)
        low_q, high_q = self._sample_correction(alpha)
        ncscore = self.ncscores_[model_name]
        ncscore_sliced = ncscore[:, :h]

        q_low_h = self._compute_qhat(ncscore_sliced, low_q, axis=0)
        q_high_h = self._compute_qhat(ncscore_sliced, high_q, axis=0)

        q_low_tiled = np.tile(q_low_h, n_series)
        q_high_tiled = np.tile(q_high_h, n_series)

        lower_bound = y_hat - q_high_tiled
        upper_bound = y_hat - q_low_tiled

        return lower_bound, upper_bound

    def predict(
        self,
        h: Optional[int] = None,
        X_df: Optional[pd.DataFrame] = None,
        alpha: Optional[float] = None,
    ) -> pd.DataFrame:
        """Generates prediction intervals [lower, upper] appended to forecast DataFrame."""
        h = h if h is not None else self.horizon
        if h > self.horizon:
            raise ValueError(
                f"Requested forecast horizon h={h} exceeds fitted calibration horizon ({self.horizon})."
            )

        pred_df = (
            self._invoke(
                self.learner.predict,
                h=h,
                X_df=X_df,
            )
            .sort_values(by=[self.id_col, self.time_col])
            .reset_index(drop=True)
        )
        model_cols = self._infer_model_cols(pred_df)

        for model in model_cols:
            y_hat = pred_df[model].to_numpy()
            n_series = pred_df[self.id_col].nunique()

            lower_bound, upper_bound = self._compute_bounds(
                y_hat=y_hat,
                model_name=model,
                h=h,
                n_series=n_series,
                alpha=alpha,
            )
            eff_alpha = self._get_alpha(alpha)
            level = int(round((1 - eff_alpha) * 100))

            pred_df[f"{model}-lo-{level}"] = lower_bound
            pred_df[f"{model}-hi-{level}"] = upper_bound

        return pred_df
