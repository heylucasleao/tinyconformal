# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License


from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from tinyconformal.core import conformal as core_conformal
from tinyconformal.core.quantiles import conformal_quantile_level, validate_alpha


class BaseConformalRegressor(ABC):
    """
    BaseRegressor

    A base class for conformal regression using a model as the learner
    to provide valid prediction intervals with a specified significance level (alpha).

    Conformal regressors aim to quantify uncertainty in predictions by generating
    prediction intervals that adapt to the data and model.
    """

    def __init__(
        self,
        learner: BaseEstimator,
        alpha: float = 0.05,
    ):
        """
        Initializes the regressor with a specified learner and significance level.

        Parameters:
        ----------
        learner : BaseEstimator
            Already-fitted base learner used by the conformal regressor.
        alpha : float, default=0.05
            The significance level applied in the regressor.

        Attributes:
        ----------
        learner : BaseEstimator
            The base learner employed in the regressor.
        alpha : float
            The significance level applied in the regressor.
        decision_function_ : array-like, default=None
            The decision function values after fitting the model.
        ncscore : array-like, default=None
            Nonconformity scores used for conformal prediction.
        n : int, default=None
            Number of calibration samples.
        """

        self.learner = learner
        self.alpha = alpha
        self.decision_function_ = None
        self.ncscore = None
        self.n = None
        self._quantile_warning_registry = set()

        # Ensure the learner is fitted
        check_is_fitted(learner)

    @abstractmethod
    def fit(self, y):
        """Calibrate the regressor from targets or subclass-specific inputs."""

    @abstractmethod
    def predict_interval(self, X, alpha=None):
        """
        Generate prediction intervals for the input data.
        To be implemented by subclasses.
        """

    def _compute_qhat(self, ncscore, q_level):
        """
        Compute the q-hat value based on the nonconformity scores and the quantile level.
        """

        return np.quantile(ncscore, q_level, method="higher")

    def _get_alpha(self, alpha):
        """Helper to retrieve the alpha value."""
        return validate_alpha(self.alpha if alpha is None else alpha)

    def generate_conformal_quantile(self, alpha=None):
        """
        Generate the conformal quantile for conformal prediction.

        This method calculates the conformal quantile based on the nonconformity scores
        of the calibration samples. The quantile serves as a threshold to determine
        the prediction intervals in conformal prediction.

        Parameters:
        -----------
        alpha : float, optional
            The significance level for conformal prediction. If None, the default
            value of self.alpha is used.

        Returns:
        --------
        float
            The computed conformal quantile.

        Notes:
        ------
        - The order statistic has rank ceil((n + 1) * (1 - alpha)), clipped to
          the observed score range when the requested coverage is unattainable.
        - This method relies on the self.ncscore attribute, which should contain the
          nonconformity scores of the calibration samples.
        """

        alpha = self._get_alpha(alpha)

        q_level = conformal_quantile_level(
            self.n,
            alpha,
            warning_registry=self._quantile_warning_registry,
            context=self.__class__.__name__,
        )

        return self._compute_qhat(self.ncscore, q_level)

    def evaluate(self, X, y, alpha=None):
        """Evaluate interval coverage, width, and mean Winkler score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features passed to ``predict_interval``.
        y : array-like of shape (n_samples,)
            Observed target values.
        alpha : float or None, default=None
            Significance-level override. Uses ``self.alpha`` when omitted.

        Returns
        -------
        dict
            Evaluation summary containing ``total``, ``alpha``,
            ``coverage_rate``, ``interval_width_mean``, and ``mwis``.
        """

        alpha = self._get_alpha(alpha)

        y_pred_intervals = self.predict_interval(X, alpha)
        lower, upper = y_pred_intervals[:, 0], y_pred_intervals[:, -1]

        return {
            "total": len(X),
            "alpha": alpha,
            **core_conformal.interval_metrics(y, lower, upper, alpha),
        }
