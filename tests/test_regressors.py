# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License


import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from tinyconformal.regressor import ConformalizedRegressor
from tinyconformal.regressor import ConformalizedQuantileRegressor
from tinyconformal.regressor import ExactnessBound

quantile_forest = pytest.importorskip("quantile_forest")
RandomForestQuantileRegressor = quantile_forest.RandomForestQuantileRegressor


@pytest.fixture
def regression_dataset():
    X, y = make_regression(
        n_samples=600,
        n_features=12,
        n_informative=6,
        noise=8.0,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42
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
def icp_learner(regression_dataset):
    learner = RandomForestRegressor(
        n_estimators=80,
        oob_score=True,
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )
    learner.fit(regression_dataset["X_train"], regression_dataset["y_train"])
    return learner


@pytest.fixture
def cqr_learner(regression_dataset):
    learner = RandomForestQuantileRegressor(
        n_estimators=80,
        default_quantiles=[0.025, 0.975],
        random_state=42,
        n_jobs=-1,
    )
    learner.fit(regression_dataset["X_train"], regression_dataset["y_train"])
    return learner


def _assert_interval_outputs(regressor, dataset):
    intervals = regressor.predict_interval(dataset["X_test"])
    assert intervals.shape == (dataset["X_test"].shape[0], 2)
    assert np.all(intervals[:, 0] <= intervals[:, 1])

    y_pred = regressor.predict(dataset["X_test"])
    assert y_pred.shape == (dataset["X_test"].shape[0],)

    results = regressor.evaluate(dataset["X_test"], dataset["y_test"])
    assert isinstance(results, dict)
    expected_keys = {
        "total",
        "alpha",
        "beta",
        "coverage_rate",
        "interval_width_mean",
        "mwis",
        "mae",
        "mbe",
        "mse",
    }
    assert set(results.keys()) == expected_keys


def test_icp_regressor_fit_predict_evaluate(regression_dataset, icp_learner):
    reg = ConformalizedRegressor(icp_learner, alpha=0.05)
    reg.fit(regression_dataset["X_calib"], regression_dataset["y_calib"], oob=False)

    assert reg.n == regression_dataset["X_calib"].shape[0]
    assert reg.ncscore.shape == (regression_dataset["X_calib"].shape[0],)
    _assert_interval_outputs(reg, regression_dataset)


def test_icp_regressor_oob_fit(regression_dataset, icp_learner):
    reg = ConformalizedRegressor(icp_learner, alpha=0.05)
    reg.fit(y=regression_dataset["y_train"], oob=True)

    assert reg.n == regression_dataset["y_train"].shape[0]
    assert reg.ncscore.shape == (regression_dataset["y_train"].shape[0],)


def test_icp_unlabeled_fit_requires_inputs(icp_learner, regression_dataset):
    reg = ConformalizedRegressor(icp_learner, alpha=0.05)

    with pytest.raises(ValueError, match="Unlabeled calibration data"):
        reg.unlabeled_fit(X=None, tilde_beta=1.0, beta=0.1)

    with pytest.raises(ValueError, match="tilde_beta"):
        reg.unlabeled_fit(X=regression_dataset["X_calib"], tilde_beta=None, beta=0.1)

    with pytest.raises(ValueError, match="beta"):
        reg.unlabeled_fit(X=regression_dataset["X_calib"], tilde_beta=1.0, beta=None)


def test_icp_unlabeled_fit_sets_constant_quantile(icp_learner, regression_dataset):
    reg = ConformalizedRegressor(icp_learner, alpha=0.05)
    reg.unlabeled_fit(regression_dataset["X_calib"], tilde_beta=2.5, beta=0.1)

    assert reg.is_unlabeled is True
    assert reg.tilde_beta == 2.5
    assert reg.beta == 0.1
    assert reg.n == regression_dataset["X_calib"].shape[0]
    assert np.all(reg.ncscore == 2.5)

    intervals = reg.predict_interval(regression_dataset["X_test"])
    widths = intervals[:, 1] - intervals[:, 0]
    assert np.allclose(widths, 5.0)


def test_icp_fit_requires_y_and_X_when_not_oob(icp_learner, regression_dataset):
    reg = ConformalizedRegressor(icp_learner, alpha=0.05)

    with pytest.raises(ValueError, match="true labels"):
        reg.fit(regression_dataset["X_calib"], y=None, oob=False)

    with pytest.raises(ValueError, match="must be provided"):
        reg.fit(X=None, y=regression_dataset["y_calib"], oob=False)


def test_cqr_fit_predict_evaluate(regression_dataset, cqr_learner):
    reg = ConformalizedQuantileRegressor(cqr_learner, alpha=0.05)
    reg.fit(regression_dataset["X_calib"], regression_dataset["y_calib"], oob=False)

    assert reg.n == regression_dataset["X_calib"].shape[0]
    assert reg.ncscore.shape == (regression_dataset["X_calib"].shape[0],)
    _assert_interval_outputs(reg, regression_dataset)


def test_cqr_unlabeled_fit_requires_X_and_beta(cqr_learner, regression_dataset):
    reg = ConformalizedQuantileRegressor(cqr_learner, alpha=0.05)

    with pytest.raises(ValueError, match="Unlabeled calibration data"):
        reg.unlabeled_fit(X=None, tilde_beta=1.5, beta=0.1)

    with pytest.raises(ValueError, match="beta"):
        reg.unlabeled_fit(X=regression_dataset["X_calib"], tilde_beta=1.5, beta=None)


def test_cqr_unlabeled_fit_sets_constant_quantile(cqr_learner, regression_dataset):
    reg = ConformalizedQuantileRegressor(cqr_learner, alpha=0.05)
    reg.unlabeled_fit(regression_dataset["X_calib"], tilde_beta=3.0, beta=0.1)

    assert reg.is_unlabeled is True
    assert reg.tilde_beta == 3.0
    assert reg.beta == 0.1
    assert reg.n == regression_dataset["X_calib"].shape[0]
    assert np.all(reg.ncscore == 3.0)

    intervals = reg.predict_interval(regression_dataset["X_test"])
    assert intervals.shape == (regression_dataset["X_test"].shape[0], 2)
    assert np.all(intervals[:, 0] <= intervals[:, 1])


def test_exactness_bound_icp_invalid_probability(regression_dataset):
    learner = RandomForestRegressor(n_estimators=10, random_state=42)

    with pytest.raises(ValueError, match=r"must be in \(0, 1\)"):
        ExactnessBound.estimate_icp_bound(
            learner,
            regression_dataset["X_train"],
            regression_dataset["y_train"],
            p=1.0,
            cv=3,
        )


def test_exactness_bound_icp_returns_expected_types(regression_dataset):
    learner = RandomForestRegressor(n_estimators=20, random_state=42)

    tilde_beta, beta = ExactnessBound.estimate_icp_bound(
        learner,
        regression_dataset["X_train"],
        regression_dataset["y_train"],
        p=0.9,
        cv=3,
    )

    assert isinstance(tilde_beta, float)
    assert isinstance(beta, float)
    assert tilde_beta >= 0.0
    assert beta == 0.1


def test_exactness_bound_cqr_invalid_probability(regression_dataset):
    learner = RandomForestQuantileRegressor(
        n_estimators=10,
        default_quantiles=[0.1, 0.9],
        random_state=42,
    )

    with pytest.raises(ValueError, match=r"must be in \(0, 1\)"):
        ExactnessBound.estimate_cqr_bound(
            learner,
            regression_dataset["X_train"],
            regression_dataset["y_train"],
            p=0.0,
            cv=3,
        )


def test_exactness_bound_cqr_returns_expected_types(regression_dataset):
    learner = RandomForestQuantileRegressor(
        n_estimators=20,
        default_quantiles=[0.1, 0.9],
        random_state=42,
    )

    tilde_beta, beta = ExactnessBound.estimate_cqr_bound(
        learner,
        regression_dataset["X_train"],
        regression_dataset["y_train"],
        p=0.9,
        cv=3,
    )

    assert isinstance(tilde_beta, float)
    assert isinstance(beta, float)
    assert beta == 0.1
