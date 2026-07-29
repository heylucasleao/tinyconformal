# Copyright (c) 2024-2025 Lucas Leão
# tinyCP - A small toolbox for conformal prediction
# Licensed under the MIT License


from sklearn.base import ClassifierMixin, BaseEstimator
import numpy as np
import warnings
from .base import BaseConformalClassifier

warnings.filterwarnings("ignore", category=RuntimeWarning, module="venn_abers")


class BinaryMarginalConformalClassifier(
    ClassifierMixin, BaseEstimator, BaseConformalClassifier
):
    """
    A marginal coverage conformal classifier methodology utilizing a classifier as the underlying learner.
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
        learner: BaseEstimator
            The base learner to be used in the classifier.
        alpha: float, default=0.05
            The significance level applied in the classifier.

        Attributes:
        learner: BaseEstimator
            The base learner employed in the classifier.
        calibration_layer: VennAbers
            The calibration layer utilized in the classifier.
        feature_importances_: array-like of shape (n_features,)
            The feature importances derived from the learner.
        hinge : array-like of shape (n_samples,), default=None
            Nonconformity scores based on the predicted probabilities. Measures the confidence margin
            between the predicted probability of the true class and the most likely incorrect class.
        alpha: float, default=0.05
            The significance level applied in the classifier.
        """

        super().__init__(learner, alpha)

    def unlabeled_fit(self, X=None):
        """
        Calibrates the nonconformity scores using unlabeled data (X) based on the model's
        own predictions (Flechsig & Pilz, 2025).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Unlabeled calibration features.
        """
        if X is None:
            raise ValueError("Unlabeled calibration data (X) must be provided.")

        self.is_unlabeled = True
        y_prob = self.learner.predict_proba(X)
        self.hinge = 1.0 - np.max(y_prob, axis=1)
        self.n = len(X)

        return self

    def fit(self, X=None, y=None, oob=False):
        """
        Fits the classifier to the training data. Calculates the conformity score for each training instance.

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
            If OOB is enabled but the learner does not support OOB predictions,
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

        self.calibration_layer.fit(self.decision_function_, y)

        y_prob, _ = self.calibration_layer.predict_proba(self.decision_function_)

        y_prob = y_prob[np.arange(len(y)), y]

        self.hinge = self.generate_non_conformity_score(y_prob)
        self.n = len(y)

        return self

    def _compute_qhat(self, ncscore, q_level):
        """
        Compute the q-hat value based on the nonconformity scores and the quantile level.
        """
        return np.quantile(ncscore, q_level, method="higher")

    def _compute_q_level(self, n, alpha=None):
        """
        Compute the quantile level based on the number of samples and significance level.
        """
        alpha = self._get_alpha(alpha)
        return np.ceil((n + 1) * (1 - alpha)) / n

    def _compute_set(self, ncscore, qhat):
        """
        Compute a predict set based on the given ncscore and qhat.
        """
        return (ncscore <= qhat).astype(int)

    def predict_set(self, X, alpha=None):
        """
        Predicts the possible set of classes for the instances in X based on the predefined significance level.

        Parameters:
        X: array-like of shape (n_samples, n_features)
            The input samples.
        alpha: float, default=None
            The significance level. If None, the value of self.alpha is used.

        Returns:
        prediction_set: array-like of shape (n_samples, n_classes)
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
        p_values = np.zeros_like(ncscore)

        for i in range(ncscore.shape[0]):
            for j in range(ncscore.shape[1]):
                numerator = np.sum(self.hinge >= ncscore[i][j]) + 1
                denumerator = self.n + 1
                p_values[i, j] = numerator / denumerator

        return p_values
