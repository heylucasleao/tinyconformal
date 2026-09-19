# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License


import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from tinyconformal.classifier.class_conditional import (
    BinaryClassConditionalConformalClassifier,
)
from tinyconformal.classifier.marginal import BinaryMarginalConformalClassifier
from tinyconformal.core.calibration import CrossValidationCalibration
from tinyconformal.evaluation import ClassifierEvaluator


@pytest.fixture
def dataset():
    weights = [0.4, 0.6]
    seed = 42

    X, y = make_classification(
        n_samples=1500,
        n_features=20,
        n_informative=2,
        weights=weights,
        random_state=seed,
        n_redundant=2,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_train, y_train, test_size=0.25, random_state=seed, stratify=y_train
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_calib": X_calib,
        "y_calib": y_calib,
        "X_test": X_test,
        "y_test": y_test,
    }


@pytest.fixture
def learner(dataset):
    model = RandomForestClassifier(oob_score=True, n_estimators=10)
    model.fit(dataset["X_train"], dataset["y_train"])
    return model


def _assert_classifier_outputs(classifier, dataset):
    classifier.calibrate(dataset["X_calib"], dataset["y_calib"])
    assert 0 < classifier.alpha <= 0.2

    y_proba = classifier.predict_proba(dataset["X_test"])
    assert y_proba.shape == (dataset["X_test"].shape[0], 2)

    prediction_set = classifier.predict_set(dataset["X_test"])
    assert prediction_set.shape == (dataset["X_test"].shape[0], 2)

    p_values = classifier.predict_p(dataset["X_test"])
    assert p_values.shape == (dataset["X_test"].shape[0], 2)

    y_pred = classifier.predict(dataset["X_test"])
    assert y_pred.shape == (dataset["X_test"].shape[0],)

    set_metrics = ClassifierEvaluator.evaluate_set(
        dataset["y_test"], prediction_set, coverage=1 - classifier.alpha
    )
    assert list(set_metrics.columns) == [
        "coverage",
        "coverage_rate",
        "set_size_mean",
        "singleton_rate",
        "empty_rate",
        "n_obs",
    ]

    classification_metrics = ClassifierEvaluator.evaluate_classification(
        dataset["y_test"], y_pred, y_proba
    )
    assert list(classification_metrics.columns) == [
        "accuracy",
        "balanced_accuracy",
        "bookmaker_informedness",
        "mcc",
        "f1",
        "fpr",
        "log_loss",
        "ece",
        "n_obs",
    ]


def test_marginal_classifier(dataset, learner):
    classifier = BinaryMarginalConformalClassifier(learner)
    classifier.fit(dataset["X_calib"], dataset["y_calib"], oob=False)
    _assert_classifier_outputs(classifier, dataset)


def test_class_cond_classifier(dataset, learner):
    classifier = BinaryClassConditionalConformalClassifier(learner)
    classifier.fit(dataset["X_calib"], dataset["y_calib"], oob=False)
    _assert_classifier_outputs(classifier, dataset)


def test_oob_marginal_classifier(dataset, learner):
    classifier = BinaryMarginalConformalClassifier(learner)
    classifier.fit(y=dataset["y_train"], oob=True)
    _assert_classifier_outputs(classifier, dataset)


def test_oob_class_conditional_classifier(dataset, learner):
    classifier = BinaryClassConditionalConformalClassifier(learner)
    classifier.fit(y=dataset["y_train"], oob=True)
    _assert_classifier_outputs(classifier, dataset)


@pytest.mark.parametrize(
    "classifier_cls",
    [BinaryMarginalConformalClassifier, BinaryClassConditionalConformalClassifier],
)
def test_classifier_accepts_oof_probabilities(classifier_cls, dataset):
    unfitted = RandomForestClassifier(n_estimators=10, random_state=42)
    probabilities = CrossValidationCalibration.classification_probabilities(
        unfitted, dataset["X_train"], dataset["y_train"], cv=3
    )
    unfitted.fit(dataset["X_train"], dataset["y_train"])
    classifier = classifier_cls(unfitted).fit_from_probabilities(
        probabilities, dataset["y_train"]
    )

    assert probabilities.shape == (len(dataset["y_train"]), 2)
    assert classifier.predict_p(dataset["X_test"]).shape == (
        len(dataset["X_test"]),
        2,
    )


@pytest.mark.parametrize(
    "classifier_cls",
    [BinaryMarginalConformalClassifier, BinaryClassConditionalConformalClassifier],
)
def test_fit_requires_y(classifier_cls, learner, dataset):
    classifier = classifier_cls(learner)

    with pytest.raises(ValueError, match="true labels"):
        classifier.fit(dataset["X_calib"], y=None, oob=False)


@pytest.mark.parametrize(
    "classifier_cls",
    [BinaryMarginalConformalClassifier, BinaryClassConditionalConformalClassifier],
)
def test_fit_oob_rejects_X(classifier_cls, learner, dataset):
    classifier = classifier_cls(learner)

    with pytest.raises(ValueError, match="should not be provided"):
        classifier.fit(dataset["X_calib"], y=dataset["y_train"], oob=True)


@pytest.mark.parametrize(
    "classifier_cls",
    [BinaryMarginalConformalClassifier, BinaryClassConditionalConformalClassifier],
)
def test_classifier_accepts_learner_without_n_classes_attribute(
    classifier_cls, dataset
):
    learner = LogisticRegression().fit(dataset["X_train"], dataset["y_train"])

    classifier = classifier_cls(learner)

    assert np.array_equal(classifier.classes, learner.classes_)


@pytest.mark.parametrize(
    "classifier_cls",
    [BinaryMarginalConformalClassifier, BinaryClassConditionalConformalClassifier],
)
def test_classifier_supports_non_positional_labels(classifier_cls, dataset):
    y_train = np.where(dataset["y_train"] == 0, "no", "yes")
    y_calib = np.where(dataset["y_calib"] == 0, "no", "yes")
    learner = LogisticRegression().fit(dataset["X_train"], y_train)
    classifier = classifier_cls(learner).fit(dataset["X_calib"], y_calib)

    predictions = classifier.predict(dataset["X_test"])
    assert set(predictions) <= {"no", "yes"}


def test_classifier_evaluator_computes_expected_set_metrics():
    evaluation = ClassifierEvaluator.evaluate_set(
        y_true=[0, 1, 1, 0],
        prediction_sets=[[1, 0], [0, 1], [1, 1], [0, 0]],
        coverage=0.9,
    ).iloc[0]

    assert evaluation["coverage"] == pytest.approx(0.9)
    assert evaluation["coverage_rate"] == pytest.approx(0.75)
    assert evaluation["set_size_mean"] == pytest.approx(1.0)
    assert evaluation["singleton_rate"] == pytest.approx(0.5)
    assert evaluation["empty_rate"] == pytest.approx(0.25)
    assert evaluation["n_obs"] == 4


def test_classifier_evaluator_computes_point_metrics_without_probabilities():
    evaluation = ClassifierEvaluator.evaluate_classification(
        y_true=[0, 0, 1, 1], y_pred=[0, 1, 1, 1]
    )

    assert "log_loss" not in evaluation
    assert "ece" not in evaluation
    assert evaluation.loc[0, "accuracy"] == pytest.approx(0.75)
    assert evaluation.loc[0, "fpr"] == pytest.approx(0.5)


@pytest.mark.parametrize("name", ["y_true", "y_pred"])
def test_classifier_evaluator_requires_zero_one_labels(name):
    values = {"y_true": [0, 1], "y_pred": [0, 1]}
    values[name] = ["no", "yes"]

    with pytest.raises(ValueError, match="labels 0 and 1"):
        ClassifierEvaluator.evaluate_classification(**values)


def test_classifier_evaluator_validates_probability_order_shape_and_sum():
    with pytest.raises(ValueError, match="sum to 1"):
        ClassifierEvaluator.evaluate_classification(
            [0, 1], [0, 1], [[0.8, 0.3], [0.2, 0.8]]
        )


def test_classifier_evaluator_validates_prediction_sets():
    with pytest.raises(ValueError, match="boolean or 0/1"):
        ClassifierEvaluator.evaluate_set([0, 1], [[1, 2], [0, 1]], 0.9)


def test_class_conditional_fit_requires_both_classes(learner, dataset):
    classifier = BinaryClassConditionalConformalClassifier(learner)
    y_single_class = np.zeros_like(dataset["y_calib"])

    with pytest.raises(ValueError, match="samples from both classes"):
        classifier.fit(dataset["X_calib"], y_single_class)


def test_class_conditional_fit_from_probabilities_requires_both_classes(
    learner, dataset
):
    classifier = BinaryClassConditionalConformalClassifier(learner)
    probabilities = learner.predict_proba(dataset["X_calib"])
    labels = np.zeros(len(probabilities), dtype=int)

    with pytest.raises(ValueError, match="samples from both classes"):
        classifier.fit_from_probabilities(probabilities, labels)
