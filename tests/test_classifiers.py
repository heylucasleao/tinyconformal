# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License


import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from tinyconformal.classifier.marginal import BinaryMarginalConformalClassifier
from tinyconformal.classifier.class_conditional import (
    BinaryClassConditionalConformalClassifier,
)


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
        "beta",
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
