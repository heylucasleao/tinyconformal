# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

from itertools import pairwise

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted


class MultiQuantileRegressor(BaseEstimator, RegressorMixin):
    """
    A multi-quantile meta-estimator wrapper for mono-quantile regressors.

    Wraps any single-quantile regressor (e.g., LightGBM, XGBoost, CatBoost,
    HistGradientBoostingRegressor) and manages multiple quantile models internally.

    Parameters
    ----------
    base_estimator : BaseEstimator
        The base scikit-learn compatible estimator instance to clone.
    quantiles : tuple of float, default=(0.025, 0.5, 0.975)
        The target quantiles to fit during training. At least two quantiles are
        required.
    """

    def __init__(self, base_estimator: BaseEstimator, quantiles=(0.025, 0.5, 0.975)):
        self.base_estimator = base_estimator
        self.quantiles = tuple(quantiles)

        self._validate_quantiles(self.quantiles)

    def _validate_quantiles(self, quantiles):
        if len(quantiles) < 2:
            raise ValueError("At least two quantiles are required.")
        if not all(
            isinstance(q, (int, float, np.integer, np.floating)) for q in quantiles
        ):
            raise TypeError("Quantiles must be numeric values.")
        if not all(0.0 < float(q) < 1.0 for q in quantiles):
            raise ValueError("Quantiles must be strictly between 0 and 1.")
        if any(left >= right for left, right in pairwise(quantiles)):
            raise ValueError("Quantiles must be unique and in strictly increasing order.")

    def _set_quantile_param(self, model: BaseEstimator, q: float) -> BaseEstimator:
        """
        Attempts to inject the target quantile parameter into the cloned model
        handling LightGBM, XGBoost, CatBoost, and scikit-learn conventions.
        """
        try:
            return model.set_params(alpha=q)
        except (ValueError, TypeError, KeyError):
            pass

        try:
            return model.set_params(quantile=q)
        except (ValueError, TypeError, KeyError):
            pass

        try:
            return model.set_params(loss_function=f"Quantile:alpha={q}")
        except (ValueError, TypeError, KeyError):
            pass

        raise AttributeError(
            f"Unable to set target quantile '{q}' on base estimator "
            f"'{model.__class__.__name__}'. Verify that the estimator supports quantile regression."
        )

    def fit(self, X, y):
        """
        Fits an independent clone of the base estimator for each quantile in `self.quantiles`.
        """
        self.models_ = {}
        for q in self.quantiles:
            model = clone(self.base_estimator)
            model = self._set_quantile_param(model, q)
            model.fit(X, y)
            self.models_[q] = model

        return self

    def predict(self, X, quantiles=None):
        """
        Generates predictions for all or a subset of trained quantiles.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        quantiles : list or tuple of float or None, default=None
            Subset of quantiles to evaluate. If None, predicts on all fitted quantiles.

        Returns
        -------
        y_pred : ndarray of shape (n_samples, len(quantiles))
            Stacked quantile predictions.
        """
        check_is_fitted(self, "models_")

        if quantiles is None:
            quantiles = self.quantiles

        preds = []
        for q in quantiles:
            if q in self.models_:
                preds.append(self.models_[q].predict(X))
            else:
                raise ValueError(
                    f"Quantile {q} was not fitted. Available quantiles: {list(self.models_.keys())}"
                )

        return np.column_stack(preds)
