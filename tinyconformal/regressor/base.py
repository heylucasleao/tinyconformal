# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License


from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from tinyconformal.utils.quantiles import conformal_quantile_level, validate_alpha


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
            The base learner to be used in the regressor.
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
        """
        Fits the classifier to the training data.
        """

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

    def _coverage_rate(self, y, y_pred):
        """
        Evaluate coverage of prediction intervals.

        """

        coverages = (y >= y_pred[:, 0]) & (y <= y_pred[:, 1])

        return np.mean(coverages)

    def _interval_width_mean(self, y_pred):
        """
        Calculates the mean width of the prediction intervals.
        """
        widths = y_pred[:, 1] - y_pred[:, 0]
        return np.mean(widths)

    def _mwi_score(self, y, y_pred, alpha):
        """
        Calculate the Winkler interval score for prediction intervals.

        If the observation falls outside the prediction interval, the score increases
        with the distance from the interval bounds.

        If the observation falls inside the prediction interval, the score depends on
        the width of the interval (narrower intervals are better).

        Parameters:
        ----------
        y : array-like
            True target values.
        y_pred : array-like
            Prediction intervals, where each row contains [lower_bound, upper_bound].
        alpha : float
            Significance level, where (1 - alpha) is the desired coverage.

        Returns:
        -------
        float
            The mean Winkler interval score.
        """

        # Extract lower and upper bounds of the intervals
        lower, upper = y_pred[:, 0], y_pred[:, 1]

        # Calculate the width of the intervals
        width = upper - lower

        # Calculate penalties for predictions below the lower bound
        penalty_lower = 2 / alpha * (lower - y) * (y < lower)

        # Calculate penalties for predictions above the upper bound
        penalty_upper = 2 / alpha * (y - upper) * (y > upper)

        # Return the mean Winkler interval score
        return np.mean(width + penalty_lower + penalty_upper)

    def evaluate(self, X, y, alpha=None):
        """
        Evaluate the performance the regressor on the given dataset.
        Parameters:
            X:
                The input features for the evaluation dataset.
            y:
                The true target values corresponding to the input features.
            alpha:
                Significance level for prediction intervals. If None, the regressor's default alpha is used.
        Returns:
            A dictionary containing the following evaluation metrics:
            - "total" (int): The total number of samples in the dataset.
            - "alpha" (float): The significance level used for evaluation.
            - "coverage_rate" (float): The coverage rate of the prediction intervals.
            - "interval_width_mean" (float): The mean width of the prediction intervals.
            - "mwis" (float): The Mean Weighted Interval Score (MWIS).
        """

        alpha = self._get_alpha(alpha)

        y_pred_intervals = self.predict_interval(X, alpha)
        bounds = np.column_stack([y_pred_intervals[:, 0], y_pred_intervals[:, -1]])

        def rounded(value):
            return np.round(value, 3)

        total = len(X)
        coverage_rate = rounded(self._coverage_rate(y, bounds))
        interval_width_mean = rounded(self._interval_width_mean(bounds))
        mwi_score = rounded(self._mwi_score(y, bounds, alpha))

        return {
            "total": total,
            "alpha": alpha,
            "coverage_rate": coverage_rate,
            "interval_width_mean": interval_width_mean,
            "mwis": mwi_score,
        }
