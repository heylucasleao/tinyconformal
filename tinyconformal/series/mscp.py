# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator
from .base_ts import BaseTimeSeriesConformalRegressor


class MultiStepTimeSeriesRegressor(
    RegressorMixin, BaseEstimator, BaseTimeSeriesConformalRegressor
):
    """
    Multi-step Conformal Distribution Regressor for Time Series (Nixtla approach).

    Generates empirical forecast trajectories by adding historical backtesting signed errors
    (e = y - y_hat) to point predictions, naturally adapting to asymmetric and heteroscedastic
    error distributions across the forecast horizon.
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
            The base time series regressor learner.
        horizon : int
            Forecast horizon step count (H).
        n_windows : int, default=3
            Number of backtesting rolling windows.
        alpha : float, default=0.05
            Significance level.
        """
        super().__init__(
            learner=learner, horizon=horizon, n_windows=n_windows, alpha=alpha
        )

    def _store_nonconformity_scores(self, raw_residuals: np.ndarray):
        """
        Stores signed residuals (Y_val - preds_val) to preserve distribution shape and bias.
        """
        self.residuals_ = raw_residuals  # Shape: (N_residuals, horizon)
        self.ncscore = self.residuals_

    def predict_interval(self, X_test, alpha: float = None) -> np.ndarray:
        """
        Generates lower and upper prediction interval bounds for each horizon step.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test features matrix.
        alpha : float, optional
            Significance level. If None, uses self.alpha.

        Returns
        -------
        intervals : ndarray of shape (n_samples, horizon, 2)
            Prediction bounds containing [lower, upper] for each sample and horizon step.
        """
        alpha = self._get_alpha(alpha)

        # Point predictions from primary fitted model. Shape: (N_test, horizon)
        preds = self.learner.predict(X_test)
        if preds.ndim == 1:
            preds = preds[:, np.newaxis]

        # 1. Simulate empirical paths: Point Forecasts + Historical Backtest Residuals
        # Shape: (N_test, N_residuals, horizon)
        conformal_dist = preds[:, np.newaxis, :] + self.residuals_[np.newaxis, :, :]

        # 2. Conformal tail quantile calculation with small-sample correction
        n = self.n
        low_q = max(0.0, alpha / 2.0 - 1.0 / (2.0 * n))
        high_q = min(1.0, 1.0 - alpha / 2.0 + 1.0 / (2.0 * n))

        # 3. Extract percentiles across calibration sample dimension (axis=1)
        lower_bound = np.quantile(conformal_dist, low_q, axis=1)
        upper_bound = np.quantile(conformal_dist, high_q, axis=1)

        # Output shape matching multi-step intervals: (N_test, horizon, 2)
        return np.stack([lower_bound, upper_bound], axis=-1)
