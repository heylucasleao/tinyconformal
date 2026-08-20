# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

import numpy as np
import pandas as pd
from typing import Union, Optional
from sklearn.base import RegressorMixin, BaseEstimator
from .base import BaseTimeSeriesConformalRegressor


class ConformalDistributionTimeSeriesRegressor(
    RegressorMixin, BaseEstimator, BaseTimeSeriesConformalRegressor
):
    """
    Multi-step Conformal Distribution Regressor for Time Series (Nixtla approach).
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

    def _generate_residuals(self, y_pred, y_true) -> np.ndarray:
        """
        Generates residuals for the conformal distribution using the Nixtla approach.
        """
        return y_pred - y_true

    def _sample_correction(self, alpha: float):
        n = self.n
        low_q = max(0.0, alpha / 2.0 - 1.0 / (2.0 * n))
        high_q = min(1.0, 1.0 - alpha / 2.0 + 1.0 / (2.0 * n))
        return low_q, high_q

    def _predict_raw(
        self, X_test: Optional[pd.DataFrame] = None, h: Optional[int] = None
    ) -> np.ndarray:
        """
        Generates base model point predictions from Nixtla estimator into standard ndarray.
        """
        h = h if h is not None else self.horizon
        preds_df = self.learner.predict(
            h=h, X_df=X_test if X_test is not None else None
        )

        return self._extract_predictions(preds_df)

    def _get_conformal_distribution(
        self, X_test: Optional[pd.DataFrame] = None, h: Optional[int] = None
    ) -> np.ndarray:
        """
        Generates the 3D empirical trajectory tensor: shape (n_series, n_residuals, horizon)
        """
        preds = self._predict_raw(X_test=X_test, h=h)  # shape: (n_series, horizon)
        return preds[:, np.newaxis, :] + self.ncscore[np.newaxis, :, :]

    def predict_interval(
        self,
        X_test: Optional[pd.DataFrame] = None,
        h: Optional[int] = None,
        alpha: float = None,
    ) -> np.ndarray:
        """
        Generates prediction intervals [lower, upper] for Nixtla inputs.

        Returns
        -------
        intervals : ndarray of shape (n_series, horizon, 2)
        """
        alpha = self._get_alpha(alpha)
        low_q, high_q = self._sample_correction(alpha)

        conformal_dist = self._get_conformal_distribution(X_test=X_test, h=h)

        lower_bound = self._compute_qhat(conformal_dist, low_q, axis=1)
        upper_bound = self._compute_qhat(conformal_dist, high_q, axis=1)

        return np.stack([lower_bound, upper_bound], axis=-1)
