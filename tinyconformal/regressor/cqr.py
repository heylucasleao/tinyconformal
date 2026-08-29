# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
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

    def unlabeled_fit(
        self,
        X,
        tilde_beta: float,
        beta: float = None,
    ):
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

        if tilde_beta is None:
            raise ValueError(
                "The error bound 'tilde_beta' (e.g., a quantile residual bound) "
                "must be provided. Consider using "
                "`tilde_beta, beta = ExactnessBound.estimate_cqr_bound(...)`."
            )

        if beta is None:
            raise ValueError(
                "The parameter 'beta' must be provided. "
                "Without 'beta', the actual lower coverage bound (1 - alpha - beta) cannot be determined. "
                "Consider using `tilde_beta, beta = ExactnessBound.estimate_cqr_bound(...)`."
            )

        self.beta = beta
        self.is_unlabeled = True
        self.tilde_beta = float(tilde_beta)
        self.n = len(X)
        self.ncscore = np.full(shape=self.n, fill_value=self.tilde_beta)

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

        self.n = len(self.decision_function_)
        self.ncscore = np.maximum(
            self.decision_function_[:, 0] - y, y - self.decision_function_[:, -1]
        )

        return self

    def predict(
        self,
        X_test,
        alpha=None,
    ):
        """
        Generates skewness-adjusted median predictions (P50) for input samples.

        This method retrieves the conformalized prediction interval alongside the
        calibrated median, which incorporates the relative position (skew ratio)
        of the base model's uncalibrated P50.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test feature matrix.

        Returns
        -------
        p50_adj : ndarray of shape (n_samples,)
            1D array containing the adjusted median predictions.

        Raises
        ------
        ValueError
            If the base learner was not fitted with or cannot predict the 0.50 quantile.
        """
        alpha = self._get_alpha(alpha)
        intervals = self.predict_interval(X_test, alpha, return_p50=True)

        return intervals[:, 1]

    def predict_interval(self, X_test, alpha=None, return_p50=False):
        """
        Generates conformalized prediction intervals and optional skewness-adjusted P50 estimates.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test feature matrix.
        alpha : float or None, default=None
            Significance level override.
        return_p50 : bool, default=False
            If True and the base learner provides 3 quantiles, returns a (n_samples, 3) matrix:
            [lower_bound, p50_adjusted, upper_bound].
            If False or if the base learner only returns 2 quantiles, returns (n_samples, 2):
            [lower_bound, upper_bound].

        Returns
        -------
        intervals : ndarray of shape (n_samples, 2) or (n_samples, 3)
            Conformalized bounds and optional adjusted median.
        """
        alpha = self._get_alpha(alpha)
        qhat = self.generate_conformal_quantile(alpha)

        q_low = alpha / 2.0
        q_high = 1.0 - (alpha / 2.0)

        req_quantiles = [q_low, 0.5, q_high] if return_p50 else [q_low, q_high]
        y_pred = self._predict_quantiles(X_test, quantiles=req_quantiles)

        q_low_base = y_pred[:, 0]
        q_high_base = y_pred[:, -1]

        lower_bound = q_low_base - qhat
        upper_bound = q_high_base + qhat

        if not return_p50:
            return np.column_stack([lower_bound, upper_bound])

        p50_base = y_pred[:, 1]

        base_spread = np.maximum(q_high_base - q_low_base, 1e-6)
        skew_ratio = np.clip((p50_base - q_low_base) / base_spread, 0.01, 0.99)

        cqr_spread = np.maximum(upper_bound - lower_bound, 1e-6)
        p50_adj = lower_bound + skew_ratio * cqr_spread

        return np.column_stack([lower_bound, p50_adj, upper_bound])
