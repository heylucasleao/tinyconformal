# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from tinyconformal.core.conformal import (
    class_indices,
    conformal_p_values,
    threshold_prediction_set,
)
from tinyconformal.core.quantiles import conformal_quantile_level

from .base import BaseConformalClassifier


class BinaryClassConditionalConformalClassifier(
    ClassifierMixin, BaseEstimator, BaseConformalClassifier
):
    """
    A Mondrian class-conditional conformal classifier using a binary learner.
    This class is inspired by the WrapperClassifier classes from the Crepes library.
    """

    def __init__(
        self,
        learner: BaseEstimator,
        alpha: float = 0.05,
    ):
        """
        Constructs the classifier with a specified learner and a Venn-Abers calibration layer.

        Parameters:
        ----------
        learner : BaseEstimator
            Already-fitted binary classifier implementing ``predict_proba``.
        alpha : float, default=0.05
            The significance level applied in the classifier.

        Attributes:
        ----------
        learner : BaseEstimator
            The base learner employed in the classifier.
        calibration_layer : VennAbers
            The calibration layer utilized in the classifier.
        classes : array-like of shape (n_classes,), default=None
            The unique class labels identified during training.
        hinge : list of array-like, default=None
            Per-class calibration scores ``1 - p_true``.
        n : array-like of shape (n_classes,), default=None
            The number of calibration points for each class.
        alpha : float, default=0.05
            The significance level applied in the classifier.
        """

        super().__init__(learner, alpha)

    def fit(self, X=None, y=None, oob=False):
        """
        Calibrate the classifier from class-conditional nonconformity scores.

        Parameters:
        ----------
        X : array-like of shape (n_samples, n_features), optional
            The training data. Required if OOB predictions are not used.
        y : array-like of shape (n_samples,)
            The true labels. Required in all cases.
        oob : bool, default=False
            Whether to use Out-of-Bag (OOB) predictions if available.

        Returns:
        -------
        self : object
            The fitted classifier.

        Raises:
        ------
        ValueError:
            If OOB is enabled but not supported by the learner,
            or if `X` and `y` are not provided when `oob=False`.
        """
        if y is None:
            raise ValueError("The true labels (y) must be provided.")

        if oob:
            if (
                not hasattr(self.learner, "oob_decision_function_")
                or self.learner.oob_decision_function_ is None
            ):
                raise ValueError(
                    "OOB predictions are not available for the provided learner."
                )
            if X is not None:
                raise ValueError(
                    "Training data (X) should not be provided when OOB is used. Ensure that 'y' is the same as the labels used during training."
                )

            # Use OOB predictions
            self.decision_function_ = self.learner.oob_decision_function_
        else:
            if X is None:
                raise ValueError(
                    "Training data (X) must be provided if OOB is not used."
                )

            # Use predict_proba for training data
            self.decision_function_ = self.learner.predict_proba(X)

        return self.fit_from_probabilities(self.decision_function_, y)

    def _store_calibration_scores(self, scores, labels):
        indices = class_indices(labels, self.classes)
        self.hinge = [scores[indices == index] for index in range(len(self.classes))]
        self.n = [values.size for values in self.hinge]
        if any(size == 0 for size in self.n):
            raise ValueError(
                "Class-conditional calibration requires samples from both classes."
            )

    def _compute_q_level(self, n, alpha):
        """
        Compute the quantile level for each class based on the number of samples and significance level.
        """
        alpha = self._get_alpha(alpha)
        q_level = np.zeros(len(self.classes))
        for index, label in enumerate(self.classes):
            q_level[index] = conformal_quantile_level(
                n[index],
                alpha,
                warning_registry=self._quantile_warning_registry,
                context=f"{self.__class__.__name__} class {label!r}",
            )
        return q_level

    def _compute_qhat(self, ncscore, q_level):
        """
        Compute the q-hat value based on the nonconformity scores and the quantile level.
        """
        qhat = np.zeros(len(self.classes))
        for index in range(len(self.classes)):
            qhat[index] = np.quantile(
                ncscore[index], q_level[index], method="higher"
            )
        return qhat

    def _compute_set(self, ncscore, qhat):
        """
        Compute a predict set based on the given ncscore and qhat.
        """
        return threshold_prediction_set(ncscore, qhat)

    def predict_set(self, X, alpha=None):
        """
        Predicts the possible set of classes for the instances in X based on the predefined significance level.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        alpha : float or None, default=None
            The significance level. If None, the value of self.alpha is used.

        Returns
        -------
        prediction_set : ndarray of shape (n_samples, 2)
            The predicted set of classes. A class is included in the set if its non-conformity score is less
            than or equal to the quantile of the hinge loss distribution at the (n+1)*(1-alpha)/n level.
        """

        alpha = self._get_alpha(alpha)

        y_prob = self.predict_proba(X)
        ncscore = self.generate_non_conformity_score(y_prob)
        qhat = self.generate_conformal_quantile(alpha)

        return self._compute_set(ncscore, qhat)

    def predict_p(self, X):
        """
        Calculate the p-values for each instance in the input data X using a non-conformity score.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input data for which the p-values need to be predicted.

        Returns:
        --------
        p_values : array-like of shape (n_samples, n_classes)
            The p-values for each instance in X for each class.

        """
        y_prob = self.predict_proba(X)
        ncscore = self.generate_non_conformity_score(y_prob)
        return np.column_stack(
            [
                conformal_p_values(self.hinge[index], ncscore[:, index])
                for index in range(len(self.classes))
            ]
        )
