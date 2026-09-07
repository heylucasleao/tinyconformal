"""Shared rolling-origin residual calibration for time-series estimators."""

import copy

import numpy as np
import pandas as pd

from tinyconformal.core import conformal as core_conformal

from .base import BaseConformalTimeSeriesRegressor


class ResidualConformalTimeSeriesRegressor(BaseConformalTimeSeriesRegressor):
    """Reusable signed-residual backtesting layer for MSCP and TSCPS."""

    def _generate_residuals(self, y_hat: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return core_conformal.signed_forecast_residuals(y_true, y_hat)

    def _finalize_residuals(
        self,
        residuals_by_model: dict[str, list[np.ndarray]],
        series_ids: list,
    ) -> dict[str, dict[object, np.ndarray]]:
        return {
            model: {
                series_id: np.vstack([window_scores[row] for window_scores in windows])
                for row, series_id in enumerate(series_ids)
            }
            for model, windows in residuals_by_model.items()
        }

    def _prepare_and_validate_steps(
        self, df: pd.DataFrame, step_size: int | None
    ) -> tuple[np.ndarray, int, int]:
        df = df.sort_values(by=[self.id_col, self.time_col])
        time_steps = np.sort(df[self.time_col].unique())
        total_steps = len(time_steps)
        n_series = df[self.id_col].nunique()
        val_end_idx = total_steps - ((self.n_windows - 1) * step_size)
        val_start_idx = val_end_idx - self.horizon
        if val_start_idx <= 0:
            raise ValueError(
                f"Time series length is too short for the specified n_windows "
                f"({self.n_windows}) and horizon ({self.horizon})."
            )
        return time_steps, total_steps, n_series

    def _split_train_val_window(
        self,
        df: pd.DataFrame,
        time_steps: np.ndarray,
        total_steps: int,
        w: int,
        step_size: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        val_end_idx = total_steps - (w * step_size)
        val_start_idx = val_end_idx - self.horizon
        train_cutoff = time_steps[val_start_idx - 1]
        val_cutoff = time_steps[val_end_idx - 1]
        train_df = df[df[self.time_col] <= train_cutoff].copy()
        val_df = df[
            (df[self.time_col] > train_cutoff) & (df[self.time_col] <= val_cutoff)
        ].copy()
        return train_df, val_df

    def _fit_predict_window(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        static_features: list | None = None,
    ) -> pd.DataFrame:
        temp_model = copy.deepcopy(self.learner)
        self._fit_forecaster(temp_model, train_df, static_features=static_features)
        predict_cols = [self.id_col, self.time_col, *self.exog_cols_]
        X_val = val_df[predict_cols] if self.exog_cols_ else None
        return self._invoke(temp_model.predict, h=self.horizon, X_df=X_val)

    def _extract_target(self, target_df: pd.DataFrame) -> np.ndarray:
        return self._pivot_panel(target_df, self.target_col).to_numpy()

    def _compute_window_residuals(
        self,
        fcst: pd.DataFrame,
        val_df: pd.DataFrame,
        n_series: int,
        residuals_by_model: dict[str, list],
    ) -> None:
        model_cols = self._infer_model_cols(fcst)
        target_pivot, y_true = self._extract_target_panel(val_df, n_series)
        for model in model_cols:
            forecast_pivot = self._pivot_panel(fcst, model)
            y_hat = forecast_pivot.to_numpy()
            self._validate_calibration_forecasts(
                forecast_pivot.index, target_pivot, y_hat
            )
            residuals_by_model.setdefault(model, []).append(
                self._generate_residuals(y_hat, y_true)
            )
