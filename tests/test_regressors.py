# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License


import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from tinyconformal.core.calibration import CrossValidationCalibration
from tinyconformal.regressor import (
    ConformalizedQuantileRegressor,
    ConformalizedRegressor,
)

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

    results = regressor.evaluate(dataset["X_test"], dataset["y_test"])
    assert isinstance(results, dict)
    expected_keys = {
        "total",
        "alpha",
        "coverage_rate",
        "interval_width_mean",
        "mwis",
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


def test_icp_fit_from_scores_validates_input(icp_learner):
    reg = ConformalizedRegressor(icp_learner, alpha=0.05)
    with pytest.raises(ValueError, match="one-dimensional"):
        reg.fit_from_scores([[1.0, 2.0]])
    with pytest.raises(ValueError, match="non-negative"):
        reg.fit_from_scores([1.0, -1.0])


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


def test_cqr_fit_from_scores(cqr_learner, regression_dataset):
    reg = ConformalizedQuantileRegressor(cqr_learner, alpha=0.05)
    scores = np.array([-1.0, 0.0, 2.0])
    reg.fit_from_scores(scores)
    np.testing.assert_array_equal(reg.ncscore, scores)
    assert reg.n == 3
    assert reg.predict_interval(regression_dataset["X_test"]).shape[1] == 2


def test_cross_validation_calibration_icp_and_cps(regression_dataset):
    learner = RandomForestRegressor(n_estimators=20, random_state=42)
    icp_scores = CrossValidationCalibration.icp_scores(
        learner,
        regression_dataset["X_train"],
        regression_dataset["y_train"],
        cv=3,
    )
    cps = CrossValidationCalibration.cps_scores(
        learner,
        RandomForestRegressor(n_estimators=20, random_state=43),
        regression_dataset["X_train"],
        regression_dataset["y_train"],
        cv=3,
    )
    assert icp_scores.shape == cps.residuals.shape == regression_dataset["y_train"].shape
    assert cps.scales.shape == cps.standardized_residuals.shape == cps.residuals.shape
    np.testing.assert_allclose(icp_scores, np.abs(cps.residuals))
    np.testing.assert_allclose(cps.standardized_residuals, cps.residuals / cps.scales)


def test_cross_validation_calibration_cqr(regression_dataset):
    learner = RandomForestQuantileRegressor(
        n_estimators=20,
        default_quantiles=[0.1, 0.9],
        random_state=42,
    )

    scores = CrossValidationCalibration.cqr_scores(
        learner,
        regression_dataset["X_train"],
        regression_dataset["y_train"],
        cv=3,
    )
    assert scores.shape == regression_dataset["y_train"].shape
    assert np.all(np.isfinite(scores))
