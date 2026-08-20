# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

from abc import ABC, abstractmethod
import numpy as np
from sklearn.base import BaseEstimator, clone
from .base import BaseConformalRegressor


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

    def fit_calibration(self, X_full, Y_full, step_size: int = None):
        """
        Executes Temporal Cross-Validation (Backtesting) across historical data to
        compute calibration residuals without data leakage.

        Parameters
        ----------
        X_full : array-like of shape (n_samples, n_features)
            Full feature matrix ordered chronologically.
        Y_full : array-like of shape (n_samples, horizon) or (n_samples,)
            Target matrix/vector ordered chronologically.
        step_size : int, optional
            Step size between rolling windows. Defaults to `self.horizon`.

        Returns
        -------
        self : object
            The calibrated time series conformal regressor.
        """
        if X_full is None or Y_full is None:
            raise ValueError("Both 'X_full' and 'Y_full' must be provided.")

        step_size = step_size if step_size is not None else self.horizon
        total_len = len(X_full)
        min_required = self.horizon * self.n_windows

        if total_len <= min_required:
            raise ValueError(
                f"Data length ({total_len}) must be greater than required window capacity ({min_required})."
            )

        raw_residuals = []

        # Sequential Backtesting (Rolling Windows)
        for w in reversed(range(self.n_windows)):
            cutoff_end = total_len - (w * step_size)
            train_end = cutoff_end - self.horizon

            X_tr, Y_tr = X_full[:train_end], Y_full[:train_end]
            X_val, Y_val = X_full[train_end:cutoff_end], Y_full[train_end:cutoff_end]

            # Fit temporary clone on historical window
            temp_model = clone(self.learner)
            temp_model.fit(X_tr, Y_tr)

            # Predict validation horizon
            preds_val = temp_model.predict(X_val)

            # Store raw residuals (y_true - y_pred)
            res = Y_val - preds_val
            raw_residuals.append(res)

        # Concatenate residuals across windows -> Shape: (n_windows * samples_per_win, horizon)
        raw_residuals = np.vstack(raw_residuals)

        # Abstract method call to allow concrete implementations (e.g., signed vs absolute scores)
        self._store_nonconformity_scores(raw_residuals)

        # Fit main single learner on 100% of the dataset
        self.learner.fit(X_full, Y_full)
        self.n = len(self.residuals_)

        return self

    @abstractmethod
    def _store_nonconformity_scores(self, raw_residuals: np.ndarray):
        """
        Process and store residuals from backtesting into self.residuals_ / self.ncscore.
        """
        pass

    def fit(self, X_full, Y_full, step_size: int = None):
        """
        Alias for fit_calibration to maintain scikit-learn API consistency.
        """
        return self.fit_calibration(X_full, Y_full, step_size=step_size)
