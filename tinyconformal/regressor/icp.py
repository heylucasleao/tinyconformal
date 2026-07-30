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

    def unlabeled_fit(
        self,
        X=None,
        tilde_beta: float = None,
        beta: float = None,
    ):
        """Fits the conformal regressor using unlabeled data via model exactness
        bounds (Flechsig & Pilz, 2025).

        Standard CP guarantees coverage >= 1 - alpha using labeled data.
        With unlabeled data, the model exactness error (beta) degrades the bound:
        Coverage >= 1 - alpha - beta

        Parameters:
        ----------
        X : array-like of shape (n_samples, n_features)
            Unlabeled calibration features[cite: 1].
        tilde_beta : float
            Prediction error bound (e.g., MedAE or q-th error quantile)[cite: 1].
        beta : float, default=0.50
            Probability bound complementary to accuracy (e.g., 0.50 for MedAE)
        [cite: 1].

        Returns:
        -------
        self : object
            The fitted regressor.
        """
        if X is None:
            raise ValueError("Unlabeled calibration data (X) must be provided.")

        if tilde_beta is None:
            raise ValueError(
                "The error bound 'tilde_beta' (e.g., MedAE or error quantile) must be provided[cite: 1]. "
                "Example: `tilde_beta = np.median(np.abs(y_tr - y_pred_cv))`[cite: 1]"
            )

        if beta is None:
            raise ValueError(
                "The parameter 'beta' must be provided. "
                "Without 'beta', the actual lower coverage bound (1 - alpha - beta) cannot be determined. "
                "Consider using `tilde_beta, beta = ExactnessBound.estimate_icp_bound(...)`."
            )

        self.is_unlabeled = True
        self.tilde_beta = float(tilde_beta)
        self.beta = float(beta)
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
