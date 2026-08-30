import numpy as np
import pytest
from sklearn.base import BaseEstimator

from tinyconformal.distribution import (
    DistributionalConformalPredictiveSystem,
    QuantileGridDistribution,
)


class QuantileLearner(BaseEstimator):
    def fit(self, X, y):
        self.fitted_ = True
        return self

    def predict(self, X, quantiles):
        centers = np.asarray(X, dtype=float).reshape(-1)
        offsets = np.asarray(quantiles) * 4.0 - 2.0
        return centers[:, None] + offsets[None, :]


def test_quantile_grid_repairs_crossing_and_inverts_distribution():
    distribution = QuantileGridDistribution(
        [[0.0, 2.0, 1.0], [10.0, 11.0, 12.0]], [0.1, 0.5, 0.9]
    )
    np.testing.assert_array_equal(distribution.quantile_predictions[0], [0, 2, 2])
    assert distribution.ppf([0.5, 0.5]).shape == (2,)
    assert distribution.cdf([1.0, 11.0]).shape == (2,)
    assert np.all(distribution.cdf(distribution.ppf(0.5)) >= 0.5)


def test_distributional_cps_calibrates_pits_and_returns_full_distribution():
    learner = QuantileLearner().fit([[0]], [0])
    cps = DistributionalConformalPredictiveSystem(
        learner, quantiles=[0.1, 0.5, 0.9]
    )
    X_cal = np.arange(5, dtype=float).reshape(-1, 1)
    y_cal = np.arange(5, dtype=float)
    cps.fit(X_cal, y_cal)

    assert cps.calibration_pits_.shape == (5,)
    distribution = cps.predict_distribution([[10.0], [20.0]])
    assert len(distribution) == 2
    assert distribution.interval(0.8).shape == (2, 2)
    assert distribution.sample(3, random_state=1).shape == (2, 3)
    np.testing.assert_allclose(distribution.ppf(0.5), [10.0, 20.0])


def test_distributional_cps_supports_precomputed_quantile_predictions():
    learner = QuantileLearner().fit([[0]], [0])
    cps = DistributionalConformalPredictiveSystem(learner, [0.1, 0.5, 0.9])
    calibration = np.array([[-1, 0, 1], [0, 1, 2], [1, 2, 3]], dtype=float)
    cps.fit_from_predictions([0, 1, 2], calibration)

    distribution = cps.predict_distribution_from_predictions([[9, 10, 11]])
    assert distribution.ppf([0.1, 0.5, 0.9]).shape == (1, 3)


def test_distributional_cps_accepts_native_cdf_ppf_distribution():
    calibration_base = QuantileGridDistribution(
        [[-1, 0, 1], [0, 1, 2], [1, 2, 3]], [0.1, 0.5, 0.9]
    )
    dcp = DistributionalConformalPredictiveSystem()
    dcp.fit_from_distribution([0, 1, 2], calibration_base)

    test_base = QuantileGridDistribution([[9, 10, 11]], [0.1, 0.5, 0.9])
    calibrated = dcp.predict_distribution_from_base(test_base)

    assert calibrated.ppf([0.1, 0.5, 0.9]).shape == (1, 3)


@pytest.mark.parametrize(
    "levels", [[0.5], [0.1, 0.1], [0.5, 0.1], [0.0, 0.5], [0.5, 1.0]]
)
def test_distributional_cps_rejects_invalid_quantile_grids(levels):
    learner = QuantileLearner().fit([[0]], [0])
    cps = DistributionalConformalPredictiveSystem(learner, levels)
    with pytest.raises(ValueError):
        cps.fit([[0]], [0])
