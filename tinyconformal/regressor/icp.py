# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License


import numpy as np
from sklearn.base import BaseEstimator

from tinyconformal.core import conformal as core_conformal

from .base import BaseConformalRegressor


class ConformalizedRegressor(BaseEstimator, BaseConformalRegressor):
    """
    ConformalizedRegressor
    This class implements a conformalized regressor that provides valid prediction intervals
    using a specified regression model as the learner. It is based on the split inductive conformal prediction (ICP) method,
    ensuring statistical validity under the assumption of exchangeability.
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
            Already-fitted point regressor used to obtain calibration and test
            predictions.
        alpha : float, default=0.05
            The significance level applied in the regressor.
        """
        super().__init__(learner, alpha)

    def fit_from_scores(self, scores):
        """Calibrate from precomputed out-of-sample absolute-residual scores."""
        scores = core_conformal.validate_calibration_values(scores, "scores")
        if np.any(scores < 0.0):
            raise ValueError("ICP scores must be non-negative.")
        self.ncscore = scores
        self.n = self.ncscore.size
        return self

    def fit(self, X=None, y=None, oob=False):
        """Calibrate ICP scores from predictions of the fitted learner.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), optional
            Calibration features. Required unless ``oob=True``.
        y : array-like of shape (n_samples,)
            Calibration targets.
        oob : bool, default=False
            Use the learner's ``oob_prediction_`` instead of predicting ``X``.

        Returns
        -------
        self
            Fitted conformal regressor.
        """

        if y is None:
            raise ValueError("The true labels (y) must be provided.")
        if oob:
            if not hasattr(self.learner, "oob_prediction_"):
                raise ValueError(
                    "OOB predictions are not available for the provided learner."
                )
            self.decision_function_ = self.learner.oob_prediction_
        else:
            if X is None:
                raise ValueError(
                    "Training data (X) must be provided if OOB is not used."
                )

            self.decision_function_ = self.learner.predict(X)

        return self.fit_from_scores(
            core_conformal.absolute_residual_scores(y, self.decision_function_)
        )

    def predict_interval(self, X_test, alpha=None):
        """Generate symmetric conformal prediction intervals.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Features for which intervals are requested.
        alpha : float or None, default=None
            Significance-level override. Uses ``self.alpha`` when omitted.

        Returns
        -------
        ndarray of shape (n_samples, 2)
            Lower and upper conformal bounds.
        """

        alpha = self._get_alpha(alpha)
        qhat = self.generate_conformal_quantile(alpha)
        y_pred = self.learner.predict(X_test)

        # Calculate the lower and upper bounds of the prediction intervals
        lower_bound, upper_bound = core_conformal.symmetric_bounds(y_pred, qhat)

        return np.array([lower_bound, upper_bound]).T
