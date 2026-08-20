# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator
from .base import BaseTimeSeriesConformalRegressor


class ConformalDistributionTimeSeriesRegressor(
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

    def _sample_correction(self, alpha):
        """
        Computes the small-sample correction for quantile levels based on the number of calibration samples.

        Parameters
        ----------
        alpha : float
            Significance level.

        Returns
        -------
        low_q, high_q : tuple of floats
            Corrected lower and upper quantile levels.
        """
        n = self.n
        low_q = max(0.0, alpha / 2.0 - 1.0 / (2.0 * n))
        high_q = min(1.0, 1.0 - alpha / 2.0 + 1.0 / (2.0 * n))
        return low_q, high_q

    def _predict_raw(self, X_test: np.ndarray) -> np.ndarray:
        """
        Generates base model point predictions ensured to be a 2D array.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test features matrix.

        Returns
        -------
        preds : ndarray of shape (n_samples, horizon)
            2D array of point forecasts.
        """
        preds = self.learner.predict(X_test)
        if preds.ndim == 1:
            preds = preds[:, np.newaxis]
        return preds

    def _get_conformal_distribution(self, X_test: np.ndarray) -> np.ndarray:
        """
        Generates the 3D empirical trajectory tensor combining point forecasts
        with historical calibration residuals.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test features matrix.

        Returns
        -------
        conformal_dist : ndarray of shape (n_samples, n_residuals, horizon)
            3D array of simulated prediction trajectories.
        """
        preds = self._predict_raw(X_test)
        return preds[:, np.newaxis, :] + self.ncscore[np.newaxis, :, :]

    def _generate_conformal_bounds(self, X_test, alpha=None):
        """
        Generates the conformal distribution trajectories for X_test and extracts
        the lower and upper quantile bounds.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test features matrix.
        alpha : float, optional
            Significance level for conformal prediction. If None, uses self.alpha.

        Returns
        -------
        lower_bound, upper_bound : tuple of ndarrays
            Lower and upper confidence bounds for the test predictions.
        """
        alpha = self._get_alpha(alpha)
        low_q, high_q = self._sample_correction(alpha)
        conformal_dist = self._get_conformal_distribution(X_test)
        lower_bound = self._compute_qhat(conformal_dist, low_q, axis=1)
        upper_bound = self._compute_qhat(conformal_dist, high_q, axis=1)

        return lower_bound, upper_bound

    def predict_interval(self, X_test, alpha=None) -> np.ndarray:
        """
        Generates prediction interval boundaries [lower, upper] for input data.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test feature matrix.
        alpha : float, optional
            Significance level. If None, uses self.alpha.

        Returns
        -------
        intervals : ndarray of shape (n_samples, horizon, 2)
            Prediction bounds containing [lower, upper] for each sample and horizon step.
        """
        alpha = self._get_alpha(alpha)
        lower_bound, upper_bound = self._generate_conformal_bounds(X_test, alpha)

        return np.stack([lower_bound, upper_bound], axis=-1)
