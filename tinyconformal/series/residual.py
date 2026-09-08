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
        window_scores_by_model: dict[str, list],
    ) -> None:
        """Collect residuals for one forecast window."""
        model_cols = self._infer_model_cols(fcst)
        shape = (n_series, self.horizon)
        y_true = val_df[self.target_col].to_numpy().reshape(shape)
        for model in model_cols:
            y_hat = fcst[model].to_numpy().reshape(shape)
            window_scores_by_model.setdefault(model, []).append(
                self._generate_residuals(y_hat, y_true)
            )
