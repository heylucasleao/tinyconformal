# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

import numpy as np
import pandas as pd
from typing import Optional
from sklearn.base import RegressorMixin, BaseEstimator
from .base import BaseTimeSeriesConformalRegressor


class ConformalQuantileTimeSeriesRegressor(
    RegressorMixin, BaseEstimator, BaseTimeSeriesConformalRegressor
):
    """
    Time Series Conformal Quantile Regression (TSCQR) wrapping Nixtla estimators.

    Requires `model_col` to be a list/tuple of two column names corresponding
    to the lower and upper quantile predictions, e.g., ['model_q_low', 'model_q_high'].
    """

    def __init__(
        self,
        learner: BaseEstimator,
        horizon: int,
        n_windows: int = 3,
        alpha: float = 0.05,
    ):
        super().__init__(
            learner=learner, horizon=horizon, n_windows=n_windows, alpha=alpha
        )

    def _sample_correction(self, alpha: float) -> float:
        """
        Calculates the finite-sample corrected quantile level for CQR.
        """
        q_level = np.ceil((self.n + 1) * (1.0 - alpha)) / self.n
        return float(np.clip(q_level, 0.0, 1.0))

    def _generate_residuals(
        self, preds_val: np.ndarray, y_val_arr: np.ndarray
    ) -> np.ndarray:
        """
        Computes the CQR nonconformity score E_t = max(q_low - y, y - q_high).

        preds_val shape: (n_series, horizon, 2)
        y_val_arr shape: (n_series, horizon)
        """
        q_low = preds_val[..., 0]
        q_high = preds_val[..., 1]

        return np.maximum(q_low - y_val_arr, y_val_arr - q_high)

    def _predict_raw(
        self, X_test: Optional[pd.DataFrame] = None, h: Optional[int] = None
    ) -> np.ndarray:
        """
        Generates base model quantile predictions from Nixtla estimator.
        Returns array of shape (n_series, horizon, 2).
        """
        h = h if h is not None else self.horizon
        preds_df = self.learner.predict(
            h=h, X_df=X_test if X_test is not None else None
        )
        return self._extract_predictions(preds_df)

    def predict_interval(
        self,
        X_test: Optional[pd.DataFrame] = None,
        h: Optional[int] = None,
        alpha: float = None,
    ) -> np.ndarray:
        """
        Generates prediction intervals [lower_bound, upper_bound] by applying
        conformal correction to base quantile predictions.

        Returns
        -------
        intervals : ndarray of shape (n_series, horizon, 2)
        """
        alpha = self._get_alpha(alpha)

        raw_quantiles = self._predict_raw(X_test=X_test, h=h)
        q_low = raw_quantiles[..., 0]
        q_high = raw_quantiles[..., 1]

        q_level = self._sample_correction(alpha)
        q_hat = self._compute_qhat(self.ncscore, q_level)

        lower_bound = q_low - q_hat
        upper_bound = q_high + q_hat

        return np.stack([lower_bound, upper_bound], axis=-1)
