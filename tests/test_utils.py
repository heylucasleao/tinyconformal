# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from tinyconformal.utils import MultiQuantileRegressor, NewsvendorSolver


def test_mqr_fit_and_predict_requested_quantiles():
    X = np.arange(30, dtype=float).reshape(-1, 1)
    y = np.sin(X[:, 0] / 5.0)
    regressor = MultiQuantileRegressor(
        GradientBoostingRegressor(loss="quantile", random_state=42),
        quantiles=(0.1, 0.5, 0.9),
    )

    with pytest.raises(NotFittedError):
        check_is_fitted(regressor)

    regressor.fit(X, y)
    predictions = regressor.predict(X[:4], quantiles=(0.1, 0.9))

    assert predictions.shape == (4, 2)
    assert set(regressor.models_) == {0.1, 0.5, 0.9}


def test_mqr_supports_quantile_grid():
    X = np.arange(30, dtype=float).reshape(-1, 1)
    y = np.sin(X[:, 0] / 5.0)
    quantiles = (0.01, 0.1, 0.5, 0.9, 0.99)
    regressor = MultiQuantileRegressor(
        GradientBoostingRegressor(loss="quantile", random_state=42),
        quantiles=quantiles,
    ).fit(X, y)

    predictions = regressor.predict(X[:4])

    assert predictions.shape == (4, len(quantiles))
    assert tuple(regressor.models_) == quantiles


@pytest.mark.parametrize(
    ("quantiles", "error", "message"),
    [
        ((), ValueError, "At least two"),
        ((0.1,), ValueError, "At least two"),
        ((0.0, 0.9), ValueError, "between 0 and 1"),
        ((0.9, 0.1), ValueError, "increasing order"),
        ((0.1, 0.1), ValueError, "increasing order"),
        (("low", "high"), TypeError, "numeric"),
    ],
)
def test_mqr_rejects_invalid_quantiles(quantiles, error, message):
    with pytest.raises(error, match=message):
        MultiQuantileRegressor(GradientBoostingRegressor(), quantiles=quantiles)


def test_mqr_rejects_quantile_that_was_not_fitted():
    X = np.arange(12, dtype=float).reshape(-1, 1)
    regressor = MultiQuantileRegressor(
        GradientBoostingRegressor(loss="quantile"), quantiles=(0.1, 0.9)
    ).fit(X, X[:, 0])

    with pytest.raises(ValueError, match="was not fitted"):
        regressor.predict(X, quantiles=(0.5,))


@pytest.fixture
def forecast_df():
    return pd.DataFrame(
        {
            "unique_id": ["B", "A"],
            "ds": [2, 1],
            "model-lo-90-cqr": [-2.0, 20.0],
            "model-hi-90-cqr": [8.0, 15.0],
        }
    )


def test_solver_computes_ratio_and_enforces_monotonicity_without_mutating_input(
    forecast_df,
):
    original = forecast_df.copy(deep=True)

    result = NewsvendorSolver.optimize(
        forecast_df,
        interval_pair=("model-lo-90-cqr", "model-hi-90-cqr"),
        underage_cost=3.0,
        overage_cost=1.0,
    )

    pd.testing.assert_frame_equal(forecast_df, original)
    assert np.allclose(result["critical_ratio"], 0.75)
    assert np.all(result["model-lo-90-cqr"] >= 0)
    assert np.all(result["model-lo-90-cqr"] <= result["model-hi-90-cqr"])
    assert np.all(result["y_optimal"] >= result["model-lo-90-cqr"])
    assert np.all(result["y_optimal"] <= result["model-hi-90-cqr"])


def test_solver_supports_cost_columns_and_sorting(forecast_df):
    df = forecast_df.assign(underage=[1.0, 3.0], overage=[3.0, 1.0])

    result = NewsvendorSolver.optimize(
        df,
        interval_pair=("model-lo-90-cqr", "model-hi-90-cqr"),
        underage_cost="underage",
        overage_cost="overage",
        assume_sorted=False,
    )

    assert result["unique_id"].tolist() == ["A", "B"]
    assert np.allclose(result["critical_ratio"], [0.75, 0.25])


def test_solver_uses_interval_midpoint_when_both_costs_are_zero(forecast_df):
    result = NewsvendorSolver.optimize(
        forecast_df,
        interval_pair=("model-lo-90-cqr", "model-hi-90-cqr"),
        underage_cost=0.0,
        overage_cost=0.0,
    )

    assert np.allclose(result["critical_ratio"], 0.5)
    assert np.allclose(
        result["y_optimal"],
        (result["model-lo-90-cqr"] + result["model-hi-90-cqr"]) / 2,
    )


@pytest.mark.parametrize("cost", [-1.0, np.inf, -np.inf])
def test_solver_rejects_invalid_costs(forecast_df, cost):
    with pytest.raises(ValueError, match="non-negative finite"):
        NewsvendorSolver.optimize(
            forecast_df,
            interval_pair=("model-lo-90-cqr", "model-hi-90-cqr"),
            underage_cost=cost,
            overage_cost=1.0,
        )


@pytest.mark.parametrize(
    ("interval_pair", "message"),
    [
        (("lo-90", "model-hi-90"), "Invalid lower"),
        (("model-lo-90", "other-hi-90"), "model mismatch"),
        (("model-lo-90", "model-hi-80"), "coverage level mismatch"),
        (("model-lo-90-cqr", "model-hi-90"), "suffix mismatch"),
        (("model-lo-100", "model-hi-100"), "between 1 and 99"),
    ],
)
def test_solver_rejects_invalid_interval_names(forecast_df, interval_pair, message):
    with pytest.raises(ValueError, match=message):
        NewsvendorSolver.optimize(
            forecast_df,
            interval_pair=interval_pair,
            underage_cost=1.0,
            overage_cost=1.0,
        )
