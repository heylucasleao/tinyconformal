# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License


from sklearn.base import RegressorMixin, BaseEstimator
import numpy as np
from .base import BaseConformalRegressor


class ConformalizedRegressor(RegressorMixin, BaseEstimator, BaseConformalRegressor):
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
            The base learner to be used in the regressor.
        alpha : float, default=0.05
            The significance level applied in the regressor.
        """
        super().__init__(learner, alpha)

    def unlabeled_fit(self, X, tilde_beta: float):
        """
        Calibrates the nonconformity scores using unlabeled data (X) based on a specified
        model exactness measure (tilde_beta, beta) as in Flechsig & Pilz (2025).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Unlabeled calibration features.
        tilde_beta : float
            The error threshold/bound (e.g., MedAE or a specific error quantile estimated during training/OOB).

        Returns
        -------
        self : object
            The fitted regressor.
        """
        if X is None:
            raise ValueError("Unlabeled calibration data (X) must be provided.")

        self.is_unlabeled = True
        self.tilde_beta = float(tilde_beta)
        self.n = len(X)
        self.ncscore = np.full(shape=self.n, fill_value=self.tilde_beta)

        return self

    def fit(self, X=None, y=None, oob=False):

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

        self.n = len(self.decision_function_)

        self.ncscore = np.abs(y - self.decision_function_)

        return self

    def predict_interval(self, X_test, alpha=None):
        """
        Generate prediction intervals for the given model and calibration data.
        """

        alpha = self._get_alpha(alpha)
        qhat = self.generate_conformal_quantile(alpha)
        y_pred = self.learner.predict(X_test)

        # Calculate the lower and upper bounds of the prediction intervals
        lower_bound = y_pred - qhat
        upper_bound = y_pred + qhat

        return np.array([lower_bound, upper_bound]).T
