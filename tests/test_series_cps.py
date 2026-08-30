from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from tinyconformal.series import (
    ContinuousConformalPredictiveSystemTimeSeriesRegressor,
    DiscreteConformalPredictiveSystemTimeSeriesRegressor,
)


@pytest.fixture
def panel():
    dates = pd.date_range("2026-01-01", periods=12, freq="D")
    return pd.DataFrame(
        {
            "unique_id": np.repeat(["a", "b"], 12),
            "ds": list(dates) * 2,
            "y": np.r_[np.arange(12), np.arange(12) + 10].astype(float),
        }
    )


@pytest.fixture
def nixtla_learner():
    learner = MagicMock()
    learner.fit.return_value = learner

    def predict(h, X_df=None):
        ids = ["a", "b"] if X_df is None else sorted(X_df["unique_id"].unique())
        dates = pd.date_range("2026-01-13", periods=h, freq="D")
        return pd.DataFrame(
            {
                "unique_id": np.repeat(ids, h),
                "ds": list(dates) * len(ids),
                "Model": np.tile(np.arange(h, dtype=float) + 10.0, len(ids)),
            }
        )

    learner.predict.side_effect = predict
    return learner


def test_series_cps_uses_sequential_backtesting_and_horizon_residuals(
    nixtla_learner, panel
):
    cps = ContinuousConformalPredictiveSystemTimeSeriesRegressor(
        nixtla_learner, horizon=2, n_windows=2
    )
    cps.fit(panel, n_jobs=1)

    assert cps.ncscores_["Model"].shape == (4, 2)
    frame, distributions = cps.predict_distribution(h=2)
    assert list(distributions) == ["Model"]
    assert len(distributions["Model"]) == len(frame) == 4
    np.testing.assert_array_equal(
        distributions["Model"].horizon_steps, np.array([0, 1, 0, 1])
    )


def test_series_cps_quantiles_intervals_and_evaluation(nixtla_learner, panel):
    cps = ContinuousConformalPredictiveSystemTimeSeriesRegressor(
        nixtla_learner, horizon=2, n_windows=2, alpha=0.1
    ).fit(panel, n_jobs=1)

    quantiles = cps.predict_quantiles([0.1, 0.25, 0.5, 0.9], h=2)
    assert {"Model-q-10", "Model-q-25", "Model-q-50", "Model-q-90"} <= set(
        quantiles
    )

    intervals = cps.predict_interval(h=2)
    assert {"Model-lo-90", "Model-hi-90"} <= set(intervals)

    test = intervals[["unique_id", "ds"]].copy()
    test["y"] = 10.0
    evaluation = cps.evaluate(test, h=2)
    assert evaluation.loc[0, "model"] == "Model"
    assert evaluation.loc[0, "level"] == "90%"


def test_discrete_series_cps_supports_pmf_and_integer_quantiles(
    nixtla_learner, panel
):
    cps = DiscreteConformalPredictiveSystemTimeSeriesRegressor(
        nixtla_learner, horizon=2, n_windows=2
    ).fit(panel, n_jobs=1)

    _, distributions = cps.predict_distribution(h=2)
    distribution = distributions["Model"]
    assert np.issubdtype(distribution.ppf(0.5).dtype, np.integer)
    assert np.all(distribution.ppf(0.01) >= 0)
    assert np.all(distribution.pmf(np.array([10, 11, 10, 11])) >= 0)


def test_discrete_series_cps_rejects_noninteger_target(nixtla_learner, panel):
    panel.loc[0, "y"] = 0.5
    cps = DiscreteConformalPredictiveSystemTimeSeriesRegressor(
        nixtla_learner, horizon=2, n_windows=2
    )
    with pytest.raises(ValueError, match="finite integers"):
        cps.fit(panel, n_jobs=1)
