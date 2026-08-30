import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from tinyconformal.distribution import (
    ContinuousConformalPredictiveSystem,
    DiscreteConformalPredictiveSystem,
)
from tinyconformal.utils.solver import NewsvendorSolver


def _fitted_dummy(value=10.0):
    model = DummyRegressor(strategy="constant", constant=value)
    model.fit(np.arange(8).reshape(-1, 1), np.full(8, value))
    return model


def test_continuous_cps_cdf_ppf_interval_and_sample():
    cps = ContinuousConformalPredictiveSystem(_fitted_dummy())
    cps.fit(np.arange(5).reshape(-1, 1), np.array([8, 9, 10, 11, 12]))
    distribution = cps.predict_distribution(np.array([[20], [21]]))

    np.testing.assert_allclose(distribution.cdf([10, 11]), [0.5, 2 / 3])
    np.testing.assert_allclose(distribution.ppf(0.5), [10, 10])
    assert distribution.ppf([0.25, 0.50, 0.75]).shape == (2, 3)
    assert distribution.interval(0.8).shape == (2, 2)
    assert distribution.sample(4, random_state=42).shape == (2, 4)

    quantile_grid = np.linspace(0.01, 0.99, 11)
    draws = distribution.ppf(quantile_grid)
    assert np.all(distribution.cdf(draws) >= quantile_grid)


def test_discrete_cps_has_integer_nonnegative_support_and_pmf():
    cps = DiscreteConformalPredictiveSystem(_fitted_dummy(value=1.5))
    cps.fit(np.arange(5).reshape(-1, 1), np.array([0, 0, 1, 2, 3]))
    distribution = cps.predict_distribution(np.array([[10], [11]]))

    quantiles = distribution.ppf([0.1, 0.5, 0.9])
    assert np.issubdtype(quantiles.dtype, np.integer)
    assert np.all(quantiles >= 0)
    masses = distribution.pmf(np.array([0, 1]))
    assert masses.shape == (2,)
    assert np.all(masses >= 0)
    np.testing.assert_array_equal(distribution.cdf(-1), [0.0, 0.0])


def test_discrete_cps_rejects_non_integer_targets():
    cps = DiscreteConformalPredictiveSystem(_fitted_dummy())
    with pytest.raises(ValueError, match="integer-valued"):
        cps.fit(np.array([[0], [1]]), np.array([1.0, 1.5]))


def test_cps_accepts_precomputed_forecasting_predictions():
    cps = DiscreteConformalPredictiveSystem(_fitted_dummy())
    cps.fit_from_predictions(
        y=np.array([0, 1, 2]), predictions=np.array([0.5, 1.5, 1.5])
    )

    distribution = cps.predict_distribution_from_predictions(np.array([4.2, 8.1]))

    assert len(distribution) == 2
    assert distribution.interval(0.8).shape == (2, 2)


def test_newsvendor_uses_distribution_ppf_row_wise():
    cps = ContinuousConformalPredictiveSystem(_fitted_dummy())
    cps.fit(np.arange(5).reshape(-1, 1), np.array([8, 9, 10, 11, 12]))
    distribution = cps.predict_distribution(np.array([[20], [21]]))
    frame = pd.DataFrame(
        {"unique_id": ["a", "b"], "ds": [1, 1], "cu": [1.0, 9.0]}
    )

    result = NewsvendorSolver.optimize_distribution(
        frame, distribution, underage_cost="cu", overage_cost=1.0
    )

    np.testing.assert_allclose(result["critical_ratio"], [0.5, 0.9])
    np.testing.assert_allclose(
        result["y_optimal"], distribution.ppf(np.array([0.5, 0.9]))
    )


def test_newsvendor_rejects_distribution_batch_size_mismatch():
    cps = ContinuousConformalPredictiveSystem(_fitted_dummy())
    cps.fit(np.array([[0], [1]]), np.array([9.0, 11.0]))
    distribution = cps.predict_distribution(np.array([[0]]))

    with pytest.raises(ValueError, match="same number of rows"):
        NewsvendorSolver.optimize_distribution(
            pd.DataFrame({"unique_id": ["a", "b"], "ds": [1, 1]}),
            distribution,
            underage_cost=1,
            overage_cost=1,
        )


def test_newsvendor_marginal_benefit_uses_discrete_cdf():
    cps = DiscreteConformalPredictiveSystem(_fitted_dummy(value=1.5))
    cps.fit(np.arange(5).reshape(-1, 1), np.array([0, 0, 1, 2, 3]))
    distribution = cps.predict_distribution(np.array([[10], [11]]))
    frame = pd.DataFrame(
        {
            "unique_id": ["a", "b"],
            "ds": [1, 1],
            "cu": [10.0, 6.0],
            "co": [2.0, 4.0],
        }
    )

    result = NewsvendorSolver.marginal_benefit_distribution(
        frame,
        distribution,
        underage_cost="cu",
        overage_cost="co",
        max_k=1,
    )

    # max_k + 1 equals the number of rows here, exercising the row/grid
    # ambiguity in the distribution input protocol.
    units = np.array([0, 1])
    thresholds = np.broadcast_to(units - 1, (len(frame), len(units)))
    probability_less = distribution.cdf(thresholds)
    expected = frame["cu"].to_numpy()[:, None] * (1 - probability_less) - frame[
        "co"
    ].to_numpy()[:, None] * probability_less
    np.testing.assert_allclose(result[["MB(k=0)", "MB(k=1)"]], expected)


def test_newsvendor_marginal_benefit_supports_explicit_units():
    cps = DiscreteConformalPredictiveSystem(_fitted_dummy(value=1.5))
    cps.fit(np.arange(5).reshape(-1, 1), np.array([0, 0, 1, 2, 3]))
    distribution = cps.predict_distribution(np.array([[10]]))
    frame = pd.DataFrame({"unique_id": ["a"], "ds": [1]})

    result = NewsvendorSolver.marginal_benefit_distribution(
        frame,
        distribution,
        underage_cost=10.0,
        overage_cost=2.0,
        units=[1, 3, 5],
    )

    assert [column for column in result if column.startswith("MB(")] == [
        "MB(k=1)",
        "MB(k=3)",
        "MB(k=5)",
    ]


def test_newsvendor_marginal_benefit_rejects_continuous_distribution():
    cps = ContinuousConformalPredictiveSystem(_fitted_dummy())
    cps.fit(np.arange(5).reshape(-1, 1), np.array([8, 9, 10, 11, 12]))
    distribution = cps.predict_distribution(np.array([[20]]))

    with pytest.raises(TypeError, match="only for discrete"):
        NewsvendorSolver.marginal_benefit_distribution(
            pd.DataFrame({"unique_id": ["a"], "ds": [1]}),
            distribution,
            underage_cost=1.0,
            overage_cost=1.0,
        )


def test_predictive_distribution_evaluates_coverage():
    cps = ContinuousConformalPredictiveSystem(_fitted_dummy())
    cps.fit(np.arange(5).reshape(-1, 1), np.array([8, 9, 10, 11, 12]))
    distribution = cps.predict_distribution(np.array([[20], [21]]))

    result = distribution.evaluate([10, 10], coverages=[0.5, 0.9])

    assert list(result.columns) == [
        "coverage",
        "empirical_coverage",
        "mean_width",
        "winkler_score",
    ]
    assert len(result) == 2
