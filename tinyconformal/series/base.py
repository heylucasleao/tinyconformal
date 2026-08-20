# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import List, Union, Optional
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
            Unfitted or fitted scikit-learn or Nixtla-compatible time series regressor.
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

        self.target_col = None
        self.time_col = None
        self.id_col = None
        self.model_col = None

    def _extract_predictions(
        self, preds: Union[np.ndarray, pd.DataFrame, pd.Series]
    ) -> np.ndarray:
        """
        Extracts raw numerical prediction values from scikit-learn or Nixtla outputs.
        """
        if isinstance(preds, (pd.DataFrame, pd.Series)):
            if isinstance(preds, pd.DataFrame):
                target_models = (
                    [self.model_col]
                    if isinstance(self.model_col, str)
                    else list(self.model_col)
                )

                if self.id_col and self.id_col in preds.columns:
                    if len(target_models) == 1:
                        preds_arr = preds.pivot(
                            index=self.id_col,
                            columns=self.time_col,
                            values=target_models[0],
                        ).to_numpy()
                    else:
                        preds_arr = np.stack(
                            [
                                preds.pivot(
                                    index=self.id_col, columns=self.time_col, values=col
                                ).to_numpy()
                                for col in target_models
                            ],
                            axis=-1,
                        )
                else:
                    preds_arr = preds[target_models].to_numpy()
            else:
                preds_arr = preds.to_numpy()
        else:
            preds_arr = np.asarray(preds)

        if preds_arr.ndim == 1:
            preds_arr = preds_arr[:, np.newaxis]

        return preds_arr

    def _extract_target(
        self, y: Union[np.ndarray, pd.DataFrame, pd.Series]
    ) -> np.ndarray:
        """
        Extracts ground truth target array from DataFrame, Series, or NumPy array.
        """
        if isinstance(y, pd.DataFrame):
            if self.id_col and self.id_col in y.columns and self.time_col in y.columns:
                y_arr = y.pivot(
                    index=self.id_col, columns=self.time_col, values=self.target_col
                ).to_numpy()
            else:
                y_arr = y[[self.target_col]].to_numpy()
        elif isinstance(y, pd.Series):
            y_arr = y.to_numpy()
        else:
            y_arr = np.asarray(y)

        if y_arr.ndim == 1:
            y_arr = y_arr[:, np.newaxis]

        return y_arr

    def _sequential_backtesting(self, X, y, step_size: int = None):
        """
        Performs sequential backtesting to compute calibration residuals without data leakage.
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
            preds_val = self._extract_predictions(temp_model.predict(X_val))
            Y_val_arr = self._extract_target(Y_val)

            residual = Y_val_arr - preds_val
            residuals.append(residual)

        return residuals

    def fit(
        self,
        X,
        y=None,
        target_col: str = "y",
        time_col: str = "ds",
        id_col: Optional[str] = "unique_id",
        model_col: Optional[Union[str, List[str]]] = None,
        step_size: int = None,
    ):
        """
        Executes Temporal Cross-Validation (Backtesting) across historical data to
        compute calibration residuals without data leakage.
        """
        # Validate model_col requirement at entry
        if isinstance(X, pd.DataFrame) or isinstance(y, pd.DataFrame):
            if model_col is None:
                raise ValueError(
                    "The parameter 'model_col' must be explicitly provided when fitting with DataFrames."
                )

        self.target_col = target_col
        self.time_col = time_col
        self.id_col = id_col
        self.model_col = model_col

        if y is None and isinstance(X, pd.DataFrame):
            y = X

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

        residuals = self._sequential_backtesting(X, y, step_size=step_size)

        self.ncscore = np.vstack(residuals)
        self.learner.fit(X, y)
        self.n = len(self.ncscore)

        return self
