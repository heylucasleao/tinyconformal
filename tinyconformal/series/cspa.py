# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class ConformalSeasonalPoolsAdaptive(RegressorMixin, BaseEstimator):
    """
    Adaptive Conformal Seasonal Pools (CSP-Adaptive) Regressor.

    A non-parametric generative conformal method for time series. It constructs
    empirical prediction distributions by combining recency-weighted seasonal pools
    with bootstrap sampling of seasonal naive residuals (y_t - y_{t-m}).
    """

    def __init__(
        self,
        season_period: int,
        decay_lambda: float = 0.05,
        alpha: float = 0.05,
        n_samples: int = 1000,
        initial_weight: float = 0.5,
    ):
        """
        Parameters
        ----------
        season_period : int
            Seasonal period / periodicity (m) of the time series (e.g., 12 for monthly, 7 for daily).
        decay_lambda : float, default=0.05
            Exponential decay rate for weighting recent seasonal observations.
        alpha : float, default=0.05
            Significance level for prediction intervals.
        n_samples : int, default=1000
            Number of Monte Carlo paths to sample.
        initial_weight : float, default=0.5
            Weight ratio balancing pure seasonal draw vs. residual-augmented naive draw.
        """
        self.season_period = season_period
        self.decay_lambda = decay_lambda
        self.alpha = alpha
        self.n_samples = n_samples
        self.initial_weight = initial_weight

        self.history_ = None
        self.signed_residuals_ = None
        self.n = None

    def fit(self, y: np.ndarray):
        """
        Fits the seasonal pool history and extracts seasonal naive residuals.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Historical time series target values.

        Returns
        -------
        self : object
            Fitted CSP-Adaptive instance.
        """
        if y is None or len(y) == 0:
            raise ValueError("Target array 'y' cannot be empty.")

        self.history_ = np.asarray(y, dtype=np.float64)
        self.n = len(self.history_)
        m = self.season_period

        # Compute signed residuals against seasonal naive model: y_t - y_{t-m}
        if self.n > m:
            self.signed_residuals_ = self.history_[m:] - self.history_[:-m]
        else:
            self.signed_residuals_ = np.zeros_like(self.history_)

        return self

    def predict_samples(self, horizon: int) -> np.ndarray:
        """
        Generates simulated forecast trajectories via adaptive seasonal pooling.

        Parameters
        ----------
        horizon : int
            Forecast horizon step count (H).

        Returns
        -------
        samples : ndarray of shape (n_samples, horizon)
            Matrix of simulated future paths.
        """
        if self.history_ is None:
            raise ValueError("This instance is not fitted yet. Call 'fit' first.")

        T = self.n
        m = self.season_period
        samples = np.zeros((self.n_samples, horizon))

        for h in range(1, horizon + 1):
            target_phase = (T + h - 1) % m

            # 1. Empirical Seasonal Pool (Same phase with exponential recency weighting)
            indices = list(range(target_phase, T, m))
            same_season_vals = self.history_[indices]

            recency = np.arange(len(same_season_vals))
            weights = np.exp(self.decay_lambda * recency)
            weights /= np.sum(weights)

            draws_seasonal = np.random.choice(
                same_season_vals, size=self.n_samples, p=weights
            )

            # 2. Conformal Residual Draws around Seasonal Naive
            naive_point = self.history_[T - m + ((h - 1) % m)]
            res_draws = np.random.choice(
                self.signed_residuals_, size=self.n_samples, replace=True
            )
            draws_conformal = naive_point + res_draws

            # 3. Adaptive Mixture Strategy
            mix_mask = np.random.binomial(1, self.initial_weight, size=self.n_samples)
            samples[:, h - 1] = (
                mix_mask * draws_seasonal + (1 - mix_mask) * draws_conformal
            )

        return samples

    def predict_interval(self, horizon: int, alpha: float = None) -> np.ndarray:
        """
        Generates lower and upper prediction interval bounds.

        Parameters
        ----------
        horizon : int
            Forecast horizon step count (H).
        alpha : float, optional
            Significance level. If None, uses self.alpha.

        Returns
        -------
        intervals : ndarray of shape (horizon, 2)
            Prediction bounds containing [lower, upper] for each horizon step.
        """
        alpha = alpha if alpha is not None else self.alpha
        samples = self.predict_samples(horizon)

        # Conformal percentile bounds calculation
        low_pct = (alpha / 2.0) * 100.0
        high_pct = (1.0 - alpha / 2.0) * 100.0

        lower_bound = np.percentile(samples, low_pct, axis=0)
        upper_bound = np.percentile(samples, high_pct, axis=0)

        return np.column_stack([lower_bound, upper_bound])

    def predict(self, horizon: int) -> np.ndarray:
        """
        Generates point forecasts (median of simulated trajectories).
        """
        samples = self.predict_samples(horizon)
        return np.median(samples, axis=0)
