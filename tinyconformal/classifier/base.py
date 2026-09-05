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
    def fit(self, y):
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

    def _expected_calibration_error(self, y, y_prob, M=5):
        """
        Generate the expected calibration error (ECE) of the classifier.

        Parameters:
        y: array-like of shape (n_samples,)
            The true labels.
        y_prob: array-like of shape (n_samples, n_classes)
            The predicted probabilities.
        M: int, default=5
            The number of bins for the uniform binning approach.

        Returns:
        ece: float
            The expected calibration error.

        The function works as follows:
        - It first creates M bins with uniform width over the interval [0, 1].
        - For each sample, it computes the maximum predicted probability and makes a prediction.
        - It then checks whether each prediction is correct or not.
        - For each bin, it calculates the empirical probability of a sample falling into the bin.
        - If the empirical probability is greater than 0, it computes the accuracy and average confidence of the bin.
        - It then calculates the absolute difference between the accuracy and the average confidence, multiplies it by the empirical probability, and adds it to the total ECE.
        """

        # uniform binning approach with M number of bins
        bin_boundaries = np.linspace(0, 1, M + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        # get max probability per sample i
        confidences = np.max(y_prob, axis=1)
        # get predictions from confidences (positional in this case)
        predicted_label = np.argmax(y_prob, axis=1)

        # get a boolean list of correct/false predictions
        predictions = predicted_label == core_conformal.class_indices(y, self.classes)

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # determine if sample is in bin m (between bin lower & upper)
            in_bin = np.logical_and(
                confidences > bin_lower.item(), confidences <= bin_upper.item()
            )
            # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
            prob_in_bin = np.mean(in_bin)

            if prob_in_bin > 0:
                # get the accuracy of bin m: acc(Bm)
                avg_pred = np.mean(predictions[in_bin])
                # get the average confidence of bin m: conf(Bm)
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
                ece += np.abs(avg_pred - avg_confidence_in_bin) * prob_in_bin
        return ece

    def _false_positive_rate(self, y, y_pred):
        """
        Computes the false positive rate (FPR).
        """
        tn, fp, _, _ = sklearn.metrics.confusion_matrix(y, y_pred).ravel()
        return fp / (fp + tn)

    def _coverage_rate(self, X, y, alpha=None):
        """
        Compute the coverage rate from conformal prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
        y : array-like of shape (n_samples,)
            True labels.
        alpha : float, optional
            Significance level (1 - desired coverage). If None, the default value of self.alpha is used.

        Returns
        -------
        float
            The average coverage rate, which represents the proportion of true labels covered by the prediction sets.
        """

        alpha = self._get_alpha(alpha)
        predict_sets = self.predict_set(X, alpha)
        indices = core_conformal.class_indices(y, self.classes)
        coverages = predict_sets[np.arange(len(y)), indices]

        return np.mean(coverages)

    def evaluate(self, X, y, alpha=None):
        """
        Evaluate the classifier on the given dataset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        y : array-like of shape (n_samples,)
            True labels for the input samples.
        alpha : float, optional
            Significance level for prediction sets. If None, the classifier's default alpha is used.

        Returns
        -------
        results : dict
            A dictionary containing the following evaluation metrics:
            - "total" (int): Total number of samples.
            - "alpha" (float): Significance level used.
            - "coverage_rate" (float): Coverage rate of the prediction sets.
            - "one_c" (float): Proportion of prediction sets containing exactly one element.
            - "avg_c" (float): Average size of the prediction sets.
            - "empty" (float): Proportion of empty prediction sets.
            - "error" (float): Classification error rate.
            - "log_loss" (float): Log loss of the predictions.
            - "ece" (float): Expected calibration error.
            - "bm" (float): Bookmaker informedness score.
            - "mcc" (float): Matthews correlation coefficient.
            - "f1" (float): F1 score.
            - "fpr" (float): False positive rate.
        """

        alpha = self._get_alpha(alpha)

        # Helper function for rounding
        def rounded(value):
            return np.round(value, 3)

        y_prob = self.predict_proba(X)
        y_pred = self.predict(X, alpha)
        predict_set = self.predict_set(X, alpha)
        total = X.shape[0] if hasattr(X, "shape") else len(X)
        coverage_rate = rounded(self._coverage_rate(X, y, alpha))
        one_c = rounded(np.mean([np.sum(p) == 1 for p in predict_set]))
        avg_c = rounded(np.mean([np.sum(p) for p in predict_set]))
        empty = rounded(np.mean([np.sum(p) == 0 for p in predict_set]))
        indices = core_conformal.class_indices(y, self.classes)
        error = rounded(1 - np.mean(predict_set[np.arange(len(y)), indices]))
        log_loss = rounded(sklearn.metrics.log_loss(y, y_prob, labels=self.classes))
        ece = rounded(self._expected_calibration_error(y, y_prob))
        fpr = rounded(self._false_positive_rate(y, y_pred))
        bookmaker_informedness = rounded(self._bookmaker_informedness(y, y_pred))
        matthews_corr = rounded(sklearn.metrics.matthews_corrcoef(y, y_pred))
        f1 = rounded(
            sklearn.metrics.f1_score(
                y, self.predict(X, alpha), pos_label=self.classes[1]
            )
        )

        # Results aggregation
        results = {
            "total": total,
            "alpha": alpha,
            "coverage_rate": coverage_rate,
            "one_c": one_c,
            "avg_c": avg_c,
            "empty": empty,
            "error": error,
            "log_loss": log_loss,
            "ece": ece,
            "bm": bookmaker_informedness,
            "mcc": matthews_corr,
            "f1": f1,
            "fpr": fpr,
        }

        return results
