# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

"""Evaluation of already-produced binary classification predictions."""

from __future__ import annotations

from itertools import pairwise
from numbers import Integral, Real

import numpy as np
import pandas as pd
from sklearn import metrics


class ClassifierEvaluator:
    """Evaluate binary classification outputs whose classes are zero and one."""

    @staticmethod
    def evaluate_set(y_true, prediction_sets, coverage: float) -> pd.DataFrame:
        """Evaluate binary conformal prediction sets.

        Parameters
        ----------
        y_true : array-like of shape (n_observations,)
            Observed binary labels.
        prediction_sets : array-like of shape (n_observations, 2)
            Boolean or zero-one membership indicators. Column zero represents
            class 0 and column one represents class 1.
        coverage : float
            Nominal coverage, strictly between zero and one.

        Returns
        -------
        pandas.DataFrame
            Single-row evaluation summary. The ``n_obs`` column has integer
            dtype.

        Columns
        -------
        **coverage** : ``float``
            Requested nominal coverage.
        **coverage_rate** : ``float``
            Fraction of observed labels included in their prediction sets.
        **set_size_mean** : ``float``
            Mean number of included classes per prediction set.
        **singleton_rate** : ``float``
            Fraction of prediction sets containing exactly one class.
        **empty_rate** : ``float``
            Fraction of prediction sets containing no classes.
        **n_obs** : ``int``
            Number of evaluated observations.
        """
        observed = ClassifierEvaluator._validate_labels(y_true, "y_true")
        coverage = ClassifierEvaluator._validate_coverage(coverage)
        sets = ClassifierEvaluator._validate_prediction_sets(
            prediction_sets, len(observed)
        )
        sizes = sets.sum(axis=1)

        result = pd.DataFrame(
            [
                {
                    "coverage": coverage,
                    "coverage_rate": float(
                        np.mean(sets[np.arange(len(observed)), observed])
                    ),
                    "set_size_mean": float(np.mean(sizes)),
                    "singleton_rate": float(np.mean(sizes == 1)),
                    "empty_rate": float(np.mean(sizes == 0)),
                    "n_obs": len(observed),
                }
            ]
        )
        result["n_obs"] = result["n_obs"].astype("int64")
        return result

    @staticmethod
    def evaluate_classification(
        y_true, y_pred, y_prob=None, n_bins: int = 5
    ) -> pd.DataFrame:
        """Evaluate point predictions and, optionally, class probabilities.

        Parameters
        ----------
        y_true : array-like of shape (n_observations,)
            Observed binary labels.
        y_pred : array-like of shape (n_observations,)
            Predicted binary labels.
        y_prob : array-like of shape (n_observations, 2), optional
            Predicted class probabilities, ordered as ``[P(y=0), P(y=1)]``.
        n_bins : int, default=5
            Number of equal-width confidence bins used to compute expected
            calibration error.

        Returns
        -------
        pandas.DataFrame
            Single-row evaluation summary. ``log_loss`` and ``ece`` are
            included only when ``y_prob`` is supplied. The ``n_obs`` column
            has integer dtype.

        Columns
        -------
        **accuracy** : ``float``
            Fraction of correctly predicted labels.
        **balanced_accuracy** : ``float``
            Mean recall across the two classes.
        **bookmaker_informedness** : ``float``
            Balanced accuracy adjusted so random performance is zero.
        **mcc** : ``float``
            Matthews correlation coefficient.
        **f1** : ``float``
            F1 score for the positive class.
        **fpr** : ``float``
            False-positive rate among observations from class 0.
        **log_loss** : ``float``
            Cross-entropy loss. Present only when ``y_prob`` is supplied.
        **ece** : ``float``
            Expected calibration error over confidence bins. Present only
            when ``y_prob`` is supplied.
        **n_obs** : ``int``
            Number of evaluated observations.
        """
        observed = ClassifierEvaluator._validate_labels(y_true, "y_true")
        predicted = ClassifierEvaluator._validate_labels(y_pred, "y_pred")
        if predicted.shape != observed.shape:
            raise ValueError("y_pred must have the same shape as y_true.")

        tn, fp, _, _ = metrics.confusion_matrix(
            observed, predicted, labels=[0, 1]
        ).ravel()
        negative_total = tn + fp
        result = {
            "accuracy": metrics.accuracy_score(observed, predicted),
            "balanced_accuracy": metrics.balanced_accuracy_score(observed, predicted),
            "bookmaker_informedness": metrics.balanced_accuracy_score(
                observed, predicted, adjusted=True
            ),
            "mcc": metrics.matthews_corrcoef(observed, predicted),
            "f1": metrics.f1_score(observed, predicted, pos_label=1, zero_division=0),
            "fpr": fp / negative_total if negative_total else 0.0,
        }

        if y_prob is not None:
            probabilities = ClassifierEvaluator._validate_probabilities(
                y_prob, len(observed)
            )
            n_bins = ClassifierEvaluator._validate_n_bins(n_bins)
            result.update(
                {
                    "log_loss": metrics.log_loss(
                        observed, probabilities, labels=[0, 1]
                    ),
                    "ece": ClassifierEvaluator._expected_calibration_error(
                        observed, probabilities, n_bins
                    ),
                }
            )

        result["n_obs"] = len(observed)
        frame = pd.DataFrame([result])
        frame["n_obs"] = frame["n_obs"].astype("int64")
        return frame

    @staticmethod
    def _validate_labels(values, name: str) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim != 1 or array.size == 0:
            raise ValueError(f"{name} must be a non-empty one-dimensional array.")
        if not np.all(np.isin(array, [0, 1])):
            raise ValueError(f"{name} must contain only the binary labels 0 and 1.")
        return array.astype(int, copy=False)

    @staticmethod
    def _validate_coverage(coverage: float) -> float:
        if not isinstance(coverage, Real):
            raise TypeError("coverage must be numeric.")
        coverage = float(coverage)
        if not np.isfinite(coverage) or not 0.0 < coverage < 1.0:
            raise ValueError("coverage must be finite and strictly between 0 and 1.")
        return coverage

    @staticmethod
    def _validate_prediction_sets(prediction_sets, n_obs: int) -> np.ndarray:
        sets = np.asarray(prediction_sets)
        if sets.shape != (n_obs, 2):
            raise ValueError("prediction_sets must have shape (len(y_true), 2).")
        if not np.all(np.isin(sets, [0, 1])):
            raise ValueError("prediction_sets must contain only boolean or 0/1 values.")
        return sets.astype(bool, copy=False)

    @staticmethod
    def _validate_probabilities(y_prob, n_obs: int) -> np.ndarray:
        try:
            probabilities = np.asarray(y_prob, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("y_prob must contain numeric values.") from exc
        if probabilities.shape != (n_obs, 2):
            raise ValueError("y_prob must have shape (len(y_true), 2).")
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("y_prob must contain only finite values.")
        if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
            raise ValueError("y_prob values must be between 0 and 1.")
        if not np.allclose(probabilities.sum(axis=1), 1.0):
            raise ValueError("each y_prob row must sum to 1.")
        return probabilities

    @staticmethod
    def _validate_n_bins(n_bins: int) -> int:
        if not isinstance(n_bins, Integral) or isinstance(n_bins, bool) or n_bins < 1:
            raise ValueError("n_bins must be a positive integer.")
        return int(n_bins)

    @staticmethod
    def _expected_calibration_error(
        y_true: np.ndarray, y_prob: np.ndarray, n_bins: int
    ) -> float:
        confidences = np.max(y_prob, axis=1)
        correct = np.argmax(y_prob, axis=1) == y_true
        boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0

        for index, (lower, upper) in enumerate(pairwise(boundaries)):
            if index == 0:
                in_bin = (confidences >= lower) & (confidences <= upper)
            else:
                in_bin = (confidences > lower) & (confidences <= upper)
            if np.any(in_bin):
                ece += np.mean(in_bin) * abs(
                    np.mean(correct[in_bin]) - np.mean(confidences[in_bin])
                )

        return float(ece)
