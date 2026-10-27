# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from tinyconformal.utils import MultiQuantileRegressor


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
