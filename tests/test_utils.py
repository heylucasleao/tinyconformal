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


@pytest.mark.parametrize(
    ("quantiles", "error", "message"),
    [
        ((0.1,), ValueError, "only 2 or 3"),
        ((0.1, 0.2, 0.9), ValueError, "median"),
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
            "lo": [-2.0, 20.0],
            "median": [5.0, 10.0],
            "hi": [8.0, 15.0],
        }
    )


def test_solver_computes_ratio_and_enforces_monotonicity_without_mutating_input(
    forecast_df,
):
    original = forecast_df.copy(deep=True)

    result = NewsvendorSolver.optimize(
        forecast_df,
        interval_pair=("lo", "hi"),
        underage_cost=3.0,
        overage_cost=1.0,
        median_col="median",
    )

    pd.testing.assert_frame_equal(forecast_df, original)
    assert np.allclose(result["critical_ratio"], 0.75)
    assert np.all(result["lo"] >= 0)
    assert np.all(result["lo"] <= result["median"])
    assert np.all(result["median"] <= result["hi"])
    assert np.all(result["y_optimal"] >= result["lo"])
    assert np.all(result["y_optimal"] <= result["hi"])


def test_solver_supports_cost_columns_and_sorting(forecast_df):
    df = forecast_df.assign(underage=[1.0, 3.0], overage=[3.0, 1.0])

    result = NewsvendorSolver.optimize(
        df,
        interval_pair=("lo", "hi"),
        underage_cost="underage",
        overage_cost="overage",
        assume_sorted=False,
    )

    assert result["unique_id"].tolist() == ["A", "B"]
    assert np.allclose(result["critical_ratio"], [0.75, 0.25])


def test_solver_uses_median_ratio_when_both_costs_are_zero(forecast_df):
    result = NewsvendorSolver.optimize(
        forecast_df,
        interval_pair=("lo", "hi"),
        underage_cost=0.0,
        overage_cost=0.0,
        median_col="median",
    )

    assert np.allclose(result["critical_ratio"], 0.5)
    assert np.allclose(result["y_optimal"], result["median"])


@pytest.mark.parametrize("cost", [-1.0, np.inf, -np.inf])
def test_solver_rejects_invalid_costs(forecast_df, cost):
    with pytest.raises(ValueError, match="non-negative finite"):
        NewsvendorSolver.optimize(
            forecast_df,
            interval_pair=("lo", "hi"),
            underage_cost=cost,
            overage_cost=1.0,
        )


@pytest.mark.parametrize("level", [0, 100, -1, 101])
def test_solver_rejects_invalid_level(forecast_df, level):
    with pytest.raises(ValueError, match="strictly between"):
        NewsvendorSolver.optimize(
            forecast_df,
            interval_pair=("lo", "hi"),
            underage_cost=1.0,
            overage_cost=1.0,
            level=level,
        )
