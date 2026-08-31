# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License


import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from tinyconformal.core.calibration import CrossValidationCalibration
from tinyconformal.classifier.class_conditional import (
    BinaryClassConditionalConformalClassifier,
)
from tinyconformal.classifier.marginal import BinaryMarginalConformalClassifier


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

    eval_dict = classifier.evaluate(dataset["X_test"], dataset["y_test"])
    assert isinstance(eval_dict, dict)
    expected_keys = {
        "total",
        "alpha",
        "coverage_rate",
        "one_c",
        "avg_c",
        "empty",
        "error",
        "log_loss",
        "ece",
        "bm",
        "mcc",
        "f1",
        "fpr",
    }
    assert set(eval_dict.keys()) == expected_keys


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
    evaluation = classifier.evaluate(
        dataset["X_test"], np.where(dataset["y_test"] == 0, "no", "yes")
    )
    assert 0.0 <= evaluation["coverage_rate"] <= 1.0


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
