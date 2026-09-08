"""Shared rolling-origin residual calibration for time-series estimators."""

import numpy as np
import pandas as pd

from tinyconformal.core import conformal as core_conformal

from .base import BaseConformalTimeSeriesRegressor


class ResidualConformalTimeSeriesRegressor(BaseConformalTimeSeriesRegressor):
    """Reusable signed-residual backtesting layer for MSCP and TSCPS."""

    def _generate_residuals(self, y_hat: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Return signed forecast residuals in predictive-distribution orientation."""
        return core_conformal.signed_forecast_residuals(y_true, y_hat)

    def _compute_window_residuals(
        self,
        fcst: pd.DataFrame,
        val_df: pd.DataFrame,
        n_series: int,
        residuals_by_model: dict[str, list],
    ) -> None:
        """Validate, align, and collect residuals for one forecast window."""
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
