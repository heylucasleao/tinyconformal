# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

from abc import ABC, abstractmethod
import numpy as np
from typing import List
import numpy as np
from sklearn.base import BaseEstimator, clone
from regressor.base import BaseConformalRegressor


class BaseTimeSeriesConformalRegressor(BaseConformalRegressor, ABC):
    """
    Base class for Time Series Conformal Prediction using Temporal Cross-Validation
    (Backtesting) to collect nonconformity scores/residuals across historical windows.
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
            Unfitted or fitted scikit-learn compatible time series regressor.
        horizon : int
            Forecast horizon step count (H).
        n_windows : int, default=3
            Number of backtesting rolling windows used to extract calibration residuals.
        alpha : float, default=0.05
            Significance level for conformal prediction.
        """
        super().__init__(learner=learner, alpha=alpha)
        self.horizon = horizon
        self.n_windows = n_windows
        self.residuals_ = None
        self.alpha = alpha

    def _sequential_backtesting(self, X, y, step_size: int = None):
        """
        Performs sequential backtesting to compute calibration residuals without data leakage.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Full feature matrix ordered chronologically.
        y : array-like of shape (n_samples, horizon) or (n_samples,)
            Target matrix/vector ordered chronologically.
        step_size : int, optional
            Step size between rolling windows. Defaults to `self.horizon`.

        Returns
        -------
        residuals : List[np.ndarray]
            List of residual arrays for each backtesting window.
        """
        if X is None or y is None:
            raise ValueError(
                "Both training data (X) and true labels (y) must be provided."
            )

        step_size = step_size if step_size is not None else self.horizon
        total_len = len(X)
        min_required = self.horizon * self.n_windows

        if total_len <= min_required:
            raise ValueError(
                f"Data length ({total_len}) must be greater than required window capacity ({min_required})."
            )

        residuals: List[np.ndarray] = []

        # Sequential Backtesting (Rolling Windows)
        for w in reversed(range(self.n_windows)):
            cutoff_end = total_len - (w * step_size)
            train_end = cutoff_end - self.horizon

            X_tr, Y_tr = X[:train_end], y[:train_end]
            X_val, Y_val = X[train_end:cutoff_end], y[train_end:cutoff_end]

            # Fit temporary clone on historical window
            temp_model = clone(self.learner)
            temp_model.fit(X_tr, Y_tr)

            # Predict validation horizon
            preds_val = temp_model.predict(X_val)

            # Store raw residuals (y_true - y_pred)
            residual = Y_val - preds_val
            residuals.append(residual)

        return residuals

    def fit(self, X, y, step_size: int = None):
        """
        Executes Temporal Cross-Validation (Backtesting) across historical data to
        compute calibration residuals without data leakage.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Full feature matrix ordered chronologically.
        y : array-like of shape (n_samples, horizon) or (n_samples,)
            Target matrix/vector ordered chronologically.
        step_size : int, optional
            Step size between rolling windows. Defaults to `self.horizon`.

        Returns
        -------
        self : object
            The calibrated time series conformal regressor.
        """
        if X is None or y is None:
            raise ValueError(
                "Both training data (X) and true labels (y) must be provided."
            )

        step_size = step_size if step_size is not None else self.horizon
        total_len = len(X)
        min_required = self.horizon * self.n_windows

        if total_len <= min_required:
            raise ValueError(
                f"Data length ({total_len}) must be greater than required window capacity ({min_required})."
            )

        residuals: List[np.ndarray] = []

        residuals = self._sequential_backtesting(X, y, step_size=step_size)

        self.ncscore = np.vstack(residuals)

        self.learner.fit(X, y)
        self.n = len(self.ncscore)

        return self
