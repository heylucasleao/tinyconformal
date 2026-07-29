# Copyright (c) 2024-2025 Lucas Leão
# tinyCP - A small toolbox for conformal prediction
# Licensed under the MIT License


from sklearn.base import RegressorMixin, BaseEstimator
import numpy as np
from .base import BaseConformalRegressor
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class ConformalizedQuantileRegressor(
    RegressorMixin, BaseEstimator, BaseConformalRegressor
):
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

    def unlabeled_fit(self, X, tilde_beta: float):
        """
        Calibrates the CQR nonconformity scores using unlabeled data (X) based on a specified
        quantile model exactness measure (tilde_beta, beta) as in Flechsig & Pilz (2025).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Unlabeled calibration features.
        tilde_beta : float
            The quantile model error bound (e.g., residual bound computed on OOB/CV quantile predictions).

        Returns
        -------
        self : object
            The fitted conformalized quantile regressor.
        """
        if X is None:
            raise ValueError("Unlabeled calibration data (X) must be provided.")

        self.is_unlabeled = True
        self.tilde_beta = float(tilde_beta)
        self.n = len(X)
        self.ncscore = np.full(shape=self.n, fill_value=self.tilde_beta)

        return self

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

        if oob:
            if not hasattr(self.learner, "oob_prediction_"):
                raise ValueError(
                    "OOB predictions are not available for the provided learner."
                )

            # Use out-of-bag predictions if available
            self.decision_function_ = self.learner.predict(
                X, quantiles=[self.alpha / 2, 1 - self.alpha / 2], oob_score=True
            )
        else:
            self.decision_function_ = self.learner.predict(
                X, quantiles=[self.alpha / 2, 1 - self.alpha / 2]
            )

        self.n = len(self.decision_function_)
        self.ncscore = np.maximum(
            self.decision_function_[:, 0] - y, y - self.decision_function_[:, 1]
        )

        return self

    def predict_interval(self, X_test, alpha=None):
        """
        Generate prediction intervals for the given model and calibration data.
        """

        alpha = self._get_alpha(alpha)
        qhat = self.generate_conformal_quantile(alpha)
        y_pred = self.learner.predict(X_test)

        lower_bound = y_pred[:, 0] - qhat
        upper_bound = y_pred[:, 1] + qhat

        return np.array([lower_bound, upper_bound]).T
