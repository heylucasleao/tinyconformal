# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License


import warnings

import numpy as np
from sklearn.base import BaseEstimator

from tinyconformal.core.conformal import (
    cqr_bounds,
    cqr_scores,
    validate_calibration_values,
)

from .base import BaseConformalRegressor

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class ConformalizedQuantileRegressor(BaseEstimator, BaseConformalRegressor):
    """
    A conformalized quantile regressor that provides valid prediction intervals
    using a specified quantile regression model as the learner. It ensures statistical validity
    under the assumption of exchangeability, based on the conformalized quantile regression (CQR) method.
    Note:
    -----
    This class is designed to work with learner models such as those provided by
    the Quantile Forest library: https://github.com/zillow/quantile-forest
    """

    def __init__(
        self,
        learner: BaseEstimator,
        alpha: float = 0.05,
    ):
        """
        Initializes the conformalized regressor with a specified learner and significance level.
        Parameters:
        ----------
        learner : BaseEstimator
            The base learner to be used in the regressor.
        alpha : float, default=0.05
            The significance level applied in the regressor.
        """
        super().__init__(learner, alpha)

    def fit_from_scores(self, scores):
        """Calibrate from precomputed out-of-sample CQR scores."""
        self.ncscore = validate_calibration_values(scores, "scores")
        self.n = self.ncscore.size
        return self

    def _predict_quantiles(self, X, quantiles, oob_score=False):
        """
        Routes prediction calls directly to the learner interface.
        """
        if hasattr(self.learner, "predict"):
            if oob_score:
                return self.learner.predict(X, quantiles=quantiles, oob_score=True)
            return self.learner.predict(X, quantiles=quantiles)

        preds = self.learner.predict(X)
        if isinstance(preds, np.ndarray) and preds.ndim == 2:
            return preds

        raise TypeError(
            f"Learner of type '{type(self.learner).__name__}' cannot be parsed. "
            "Ensure it supports `.predict(X, quantiles=[...])` or wrap it with `MultiQuantileRegressor`."
        )

    def fit(self, X, y, oob=False):
        """
        Fit the conformalized regressor by calculating nonconformity scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Training target vector.
        oob : bool, default=False
            Whether to use out-of-bag predictions (if supported by the learner).

        Returns
        -------
        self : object
            The fitted conformalized regressor.
        """
        if X is None or y is None:
            raise ValueError(
                "Both training data (X) and true labels (y) must be provided."
            )

        q_low = self.alpha / 2.0
        q_high = 1.0 - (self.alpha / 2.0)

        if oob:
            if not hasattr(self.learner, "oob_prediction_"):
                raise ValueError(
                    "OOB predictions are not available for the provided learner."
                )

            self.decision_function_ = self._predict_quantiles(
                X, quantiles=[q_low, q_high], oob_score=True
            )
        else:
            self.decision_function_ = self._predict_quantiles(
                X, quantiles=[q_low, q_high]
            )

        return self.fit_from_scores(
            cqr_scores(
                y, self.decision_function_[:, 0], self.decision_function_[:, -1]
            )
        )

    def predict_interval(self, X_test, alpha=None):
        """
        Generates conformalized prediction intervals.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test feature matrix.
        alpha : float or None, default=None
            Significance level override.
        Returns
        -------
        intervals : ndarray of shape (n_samples, 2)
            Conformalized lower and upper bounds.
        """
        alpha = self._get_alpha(alpha)
        qhat = self.generate_conformal_quantile(alpha)

        q_low = alpha / 2.0
        q_high = 1.0 - (alpha / 2.0)

        y_pred = self._predict_quantiles(X_test, quantiles=[q_low, q_high])

        q_low_base = y_pred[:, 0]
        q_high_base = y_pred[:, -1]

        lower_bound, upper_bound = cqr_bounds(q_low_base, q_high_base, qhat)

        return np.column_stack([lower_bound, upper_bound])
