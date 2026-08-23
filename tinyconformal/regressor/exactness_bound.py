# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.base import BaseEstimator


class ExactnessBound:
    """
    Utility class to estimate model exactness bounds (tilde_beta) for unlabeled
    conformal prediction calibration as described in Flechsig & Pilz (2025).
    """

    @staticmethod
    def estimate_icp_bound(
        learner: BaseEstimator,
        X_train,
        y_train,
        p: float = 0.95,
        cv: int = 5,
    ) -> tuple[float, float]:
        """
        Estimates the exactness bound (tilde_beta) for standard Inductive Conformal Prediction (ICP)
        using mean regression learners.

        Calculates out-of-fold absolute prediction residuals |y - y_hat| via
        cross-validation, returning the residual bound at quantile p along with
        the derived error rate beta.

        Parameters
        ----------
        learner : BaseEstimator
            An unfitted scikit-learn compatible mean regressor returning 1D predictions (N,).
        X_train : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y_train : array-like of shape (n_samples,)
            Training target values.
        p : float, default=0.95
            The target error percentile for training residuals. Higher values (e.g., 0.90 or 0.95)
            yield stronger theoretical guarantees.
        cv : int, default=5
            Number of cross-validation folds.

        Returns
        -------
        tilde_beta : float
            The absolute residual bound |y - y_hat| at percentile p.
        beta : float
            The complementary error rate (1.0 - p)

        Examples
        --------
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> from tinyconformal import ConformalizedRegressor, ExactnessBound
        >>>
        >>> rf = RandomForestRegressor(random_state=42)
        >>> p = 0.95
        >>> tilde_beta, beta = ExactnessBound.estimate_icp_bound(rf, X_train, y_train, p=p, cv=5)
        >>>
        >>> rf.fit(X_train, y_train)
        >>> reg = ConformalizedRegressor(rf, alpha=0.05)
        >>> reg.unlabeled_fit(X_unlabeled, tilde_beta=tilde_beta, beta=beta)
        """
        if not (0.0 < p < 1.0):
            raise ValueError(
                f"The quantile probability 'p' must be in (0, 1), got {p}."
            )
        y_cv_pred = cross_val_predict(learner, X_train, y_train, cv=cv)
        cv_residuals = np.abs(y_train - y_cv_pred)

        return float(np.quantile(cv_residuals, p, method="higher")), round(
            float(1.0 - p), 3
        )

    @staticmethod
    def estimate_cqr_bound(
        learner: BaseEstimator,
        X_train,
        y_train,
        p: float = 0.95,
        cv: int = 5,
    ) -> tuple[float, float]:
        """
        Estimates the exactness bound (tilde_beta) for Conformalized Quantile Regression (CQR)
        using quantile learners.

        Parameters
        ----------
        learner : BaseEstimator
            An unfitted quantile regressor outputting lower and upper quantile predictions (N, 2).
        X_train : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y_train : array-like of shape (n_samples,)
            Training target vector.
        p : float, default=0.95
            The target percentile of CQR residuals to use as exactness bound.
        cv : int, default=5
            Number of cross-validation folds.

        Returns
        -------
        tilde_beta : float
            The CQR quantile error bound at percentile p.
        beta : float
            The complementary error rate (1.0 - p)

        Examples
        --------
        >>> from quantile_forest import RandomForestQuantileRegressor
        >>> from tinyconformal import ConformalizedQuantileRegressor, ExactnessBound
        >>>
        >>> qf = RandomForestQuantileRegressor(default_quantiles=[0.025, 0.975], random_state=42)
        >>> p = 0.95
        >>> tilde_beta, beta = ExactnessBound.estimate_cqr_bound(qf, X_train, y_train, p=p, cv=5)
        >>>
        >>> qf.fit(X_train, y_train)
        >>> cqr = ConformalizedQuantileRegressor(qf, alpha=0.05)
        >>> cqr.unlabeled_fit(X_unlabeled, tilde_beta=tilde_beta, beta=beta)
        """
        if not (0.0 < p < 1.0):
            raise ValueError(
                f"The quantile probability 'p' must be in (0, 1), got {p}."
            )
        preds = cross_val_predict(learner, X_train, y_train, cv=cv)
        cqr_residuals = np.maximum(preds[:, 0] - y_train, y_train - preds[:, -1])

        return float(np.quantile(cqr_residuals, p, method="higher")), round(
            float(1.0 - p), 3
        )
