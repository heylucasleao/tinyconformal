# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License


import warnings
from abc import ABC, abstractmethod

import numpy as np
import sklearn.metrics
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from venn_abers import VennAbers

from tinyconformal.core import conformal as core_conformal
from tinyconformal.core.quantiles import validate_alpha

warnings.filterwarnings("ignore", category=RuntimeWarning, module="venn_abers")


class BaseConformalClassifier(ABC):
    """
    BaseConformalClassifier

    A base class for conformal prediction using a model as the learner
    and Venn-Abers calibration for confidence estimation.
    This approach provides valid predictions with a specified significance level (alpha).

    Conformal classifiers aim to quantify uncertainty in predictions.
    """

    def __init__(
        self,
        learner: BaseEstimator,
        alpha: float = 0.05,
    ):
        """

        Initializes the classifier with a specified learner and a Venn-Abers calibration layer.

        Parameters
        ----------
        learner : BaseEstimator
            Already-fitted binary classifier implementing ``predict_proba``.
        alpha : float, default=0.05
            The significance level applied in the classifier.

        Attributes
        ----------
        learner : BaseEstimator
            The base learner employed in the classifier.
        calibration_layer : VennAbers
            The calibration layer utilized in the classifier.
        decision_function_ : ndarray or None
            Out-of-sample class probabilities used for calibration.
        hinge : array-like of shape (n_samples,), default=None
            The non-conformity scores of the calibration samples.
        alpha : float, default=0.05
            The significance level applied in the classifier.
        n : int or None
            The number of calibration samples.
        """

        self.learner = learner
        self.alpha = alpha
        self.calibration_layer = VennAbers()
        self.classes = getattr(self.learner, "classes_", [0, 1])
        self.decision_function_ = None
        check_is_fitted(learner)

        if len(self.classes) != 2:
            raise ValueError("This classifier supports only binary classification.")

        self.hinge = None
        self.n = None
        self._quantile_warning_registry = set()

    @abstractmethod
    def fit(self, X=None, y=None, oob=False):
        """
        Fits the classifier to the training data.
        """

    @abstractmethod
    def predict_set(self, X, alpha=None):
        """
        Generate a prediction set for the given input.
        This method must be implemented by subclasses.
        """

    @abstractmethod
    def _compute_qhat(self, ncscore, q_level):
        """
        Compute the q-hat value based on the nonconformity scores and the quantile level.
        """

    @abstractmethod
    def _compute_set(self, ncscore, qhat):
        """
        Compute a set based on the given ncscore and qhat.
        """

    @abstractmethod
    def _compute_q_level(self, n, alpha):
        """Compute the quantile level from sample size and significance level."""

    @abstractmethod
    def _store_calibration_scores(self, scores, labels):
        """Store marginal or class-conditional calibration scores."""

    def fit_from_probabilities(self, probabilities, y):
        """Calibrate from out-of-sample probabilities and observed labels.

        Parameters
        ----------
        probabilities : array-like of shape (n_samples, 2)
            Probabilities produced without fitting on the corresponding rows.
            Columns must follow ``self.classes`` order.
        y : array-like of shape (n_samples,)
            Observed binary class labels.

        Returns
        -------
        self
            Fitted conformal classifier.
        """
        probabilities = core_conformal.validate_probabilities(probabilities)
        y = np.asarray(y)
        if probabilities.shape != (y.size, len(self.classes)):
            raise ValueError(
                "probabilities must contain one row per label and one column per class."
            )
        indices = core_conformal.class_indices(y, self.classes)
        if any(not np.any(indices == index) for index in range(len(self.classes))):
            raise ValueError("Calibration requires samples from both classes.")
        self.decision_function_ = probabilities
        self.calibration_layer.fit(probabilities, indices)
        calibrated, _ = self.calibration_layer.predict_proba(probabilities)
        scores = core_conformal.true_class_probability_scores(
            calibrated, y, self.classes
        )
        self._store_calibration_scores(scores, y)
        return self

    def _compute_prediction(self, prediction_set):
        """Return the class label contained in every singleton prediction set."""
        positions = np.argmax(prediction_set, axis=1)
        return np.asarray(self.classes)[positions]

    def _bookmaker_informedness(self, y, y_pred):
        """
        Calculate the bookmaker informedness score for the given true and predicted labels.
        """
        return sklearn.metrics.balanced_accuracy_score(y, y_pred, adjusted=True)

    def _select_scoring_function(self, scoring_func):
        """
        Select the scoring function based on the provided string.
        """

        if scoring_func == "bm":
            func = self._bookmaker_informedness
        elif scoring_func == "mcc":
            func = sklearn.metrics.matthews_corrcoef
        else:
            raise ValueError("Invalid metric function. Please use 'bm' or 'mcc'.")
        return func

    def _get_alpha(self, alpha):
        """Helper to retrieve the alpha value."""
        return validate_alpha(self.alpha if alpha is None else alpha)

    def generate_non_conformity_score(self, y_prob):
        """
        Generates the non-conformity score based on the hinge loss.

        This function calculates the non-conformity score for conformal prediction
        using the hinge loss approach.
        """
        return core_conformal.probability_scores(y_prob)

    def generate_conformal_quantile(self, alpha=None):
        """
        Generate the conformal quantile for conformal prediction.

        This method calculates the conformal quantile based on the nonconformity scores
        of the calibration samples. The quantile serves as a threshold to determine
        the prediction sets in conformal prediction.

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

        q_level = self._compute_q_level(self.n, alpha)

        return self._compute_qhat(self.hinge, q_level)

    def predict_proba(self, X):
        """
        Return Venn-Abers calibrated class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        p_prime : ndarray of shape (n_samples, 2)
            The calibrated class probabilities.
        """
        y_score = self.learner.predict_proba(X)

        p_prime, _ = self.calibration_layer.predict_proba(y_score)
        return p_prime

    def calibrate(self, X, y, max_alpha=0.2, func="mcc"):
        """
        Calibrates the alpha value to optimize the specified metric.

        This method evaluates a range of alpha values (from 0.01 to `max_alpha`)
        to determine the optimal significance level based on the provided scoring
        function. The alpha value that maximizes the scoring function is selected.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples used for calibration.
        y : array-like of shape (n_samples,)
            True labels corresponding to the input samples.
        max_alpha : float, optional, default=0.2
            The maximum alpha value to consider during calibration. The range of
            alpha values tested will be from 0.01 to `max_alpha`, inclusive.
        func : str, optional, default="mcc"
            The name of the scoring function to use for optimization. Supported
            functions should be implemented in the `_select_scoring_function` method.

        Raises
        ------
            If an invalid scoring function name is provided in the `func` parameter.

        Returns
            The optimal alpha value that maximizes the scoring function.
        """

        scoring_func = self._select_scoring_function(func)

        alphas = {k: None for k in np.round(np.arange(0.01, max_alpha + 0.01, 0.01), 2)}

        for alpha in alphas:
            y_pred = self.predict(X, alpha)
            alphas[alpha] = scoring_func(y, y_pred)

        self.alpha = max(alphas, key=alphas.get)

        return self.alpha

    def predict(self, X, alpha=None):
        """
        Predicts the classes for the input samples.

        Parameters:
        -----------
        X: np.ndarray of shape (n_samples, n_features)
            Input samples.
        alpha: float, optional
            Significance level. If None, defaults to the classifier's alpha value.

        Returns:
        --------
        np.ndarray of shape (n_samples,)
            Predicted class labels, where 1 indicates the model's certainty.
        """

        alpha = self._get_alpha(alpha)

        prediction_set = self.predict_set(X, alpha)
        singletons = prediction_set.sum(axis=1) == 1
        predictions = np.asarray(self.learner.predict(X)).copy()
        predictions[singletons] = self._compute_prediction(prediction_set[singletons])
        return predictions
