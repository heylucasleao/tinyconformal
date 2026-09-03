from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from tinyconformal.distribution.cross import ContinuousConformalDistribution
from tinyconformal.series import (
    ContinuousTimeSeriesConformalPredictiveSystem,
    DiscreteTimeSeriesConformalPredictiveSystem,
)
from tinyconformal.series.cps import HorizonConformalDistribution
from tinyconformal.utils import NewsvendorSolver


def test_series_public_api_only_exports_modeling_classes():
    import tinyconformal.series as series

    assert series.__all__ == [
        "MultiStepConformalTimeSeriesRegressor ",
        "ConformalizedQuantileTimeSeriesRegressor",
        "ContinuousTimeSeriesConformalPredictiveSystem",
        "DiscreteTimeSeriesConformalPredictiveSystem",
    ]
    assert not hasattr(series, "HorizonConformalDistribution")


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


@pytest.fixture
def dispersion_learner():
    return DummyRegressor(strategy="mean")


def test_series_cps_uses_sequential_backtesting_and_horizon_residuals(
    nixtla_learner, dispersion_learner, panel
):
    cps = ContinuousTimeSeriesConformalPredictiveSystem(
        nixtla_learner, dispersion_learner, horizon=2, n_windows=2
    )
    cps.fit(panel, n_jobs=1)

    assert set(cps.ncscores_["Model"]) == {"a", "b"}
    assert all(scores.shape == (2, 2) for scores in cps.ncscores_["Model"].values())
    for series_id in ("a", "b"):
        np.testing.assert_allclose(
            cps.ncscores_["Model"][series_id],
            cps.raw_residuals_["Model"][series_id]
            / cps.oof_scales_["Model"][series_id],
        )
    forecast = cps.predict_distribution(h=2)
    frame = forecast.to_frame()
    assert forecast.model == "Model"
    assert len(forecast) == len(frame) == 4
    np.testing.assert_array_equal(
        forecast._distribution.horizon_steps, np.array([0, 1, 0, 1])
    )


def test_split_and_single_horizon_cps_share_distribution_semantics():
    locations = np.array([10.0, 20.0])
    residuals = np.array([-2.0, 0.0, 3.0])
    split = ContinuousConformalDistribution(locations, residuals)
    horizon = HorizonConformalDistribution(
        locations, residuals[:, None], horizon_steps=np.zeros(2, dtype=int)
    )

    np.testing.assert_allclose(split.cdf([9.0, 22.0]), horizon.cdf([9.0, 22.0]))
    np.testing.assert_allclose(split.ppf([0.2, 0.5, 0.8]), horizon.ppf([0.2, 0.5, 0.8]))


def test_horizon_distribution_supports_temporal_decay_weights():
    distribution = HorizonConformalDistribution(
        locations=[10.0],
        residuals=np.array([[-10.0], [0.0], [10.0]]),
        horizon_steps=[0],
        weights=np.array([0.01, 0.1, 1.0]),
    )

    np.testing.assert_allclose(distribution.ppf(0.5), [20.0])
    np.testing.assert_allclose(distribution.cdf(10.0), [0.11 / 1.11])


def test_series_cps_distributions_are_calibrated_by_unique_id(
    nixtla_learner, dispersion_learner, panel
):
    cps = ContinuousTimeSeriesConformalPredictiveSystem(
        nixtla_learner, dispersion_learner, horizon=2, n_windows=2
    ).fit(panel, n_jobs=1)
    cps.ncscores_["Model"] = {
        "a": np.array([[-1.0, -2.0], [1.0, 2.0]]),
        "b": np.array([[-10.0, -20.0], [10.0, 20.0]]),
    }
    scale_features = cps._scale_features(["a", "b"])
    cps.dispersion_learners_["Model"] = cps._new_dispersion_pipeline().fit(
        scale_features, np.ones(len(scale_features))
    )

    forecast = cps.predict_distribution(h=2)
    medians = forecast.ppf(0.5)

    np.testing.assert_array_equal(medians["unique_id"], ["a", "a", "b", "b"])
    np.testing.assert_array_equal(medians["Model-q-50"], [11.0, 13.0, 20.0, 31.0])


def test_series_cps_quantiles_intervals_and_evaluation(
    nixtla_learner, dispersion_learner, panel
):
    cps = ContinuousTimeSeriesConformalPredictiveSystem(
        nixtla_learner, dispersion_learner, horizon=2, n_windows=2, alpha=0.1
    ).fit(panel, n_jobs=1)

    forecast = cps.predict_distribution(h=2)
    assert not hasattr(forecast, "sample")
    assert not hasattr(cps, "predict_quantiles")
    assert not hasattr(cps, "predict_interval")
    quantiles = forecast.ppf([0.1, 0.25, 0.5, 0.9])
    assert {"Model-q-10", "Model-q-25", "Model-q-50", "Model-q-90"} <= set(quantiles)
    rowwise_quantiles = forecast.ppf(
        np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]])
    )
    assert {"Model-q-0", "Model-q-1"} <= set(rowwise_quantiles)

    intervals = forecast.interval(0.9)
    assert {"Model-lo-90", "Model-hi-90"} <= set(intervals)

    test = intervals[["unique_id", "ds"]].copy()
    test["y"] = 10.0
    evaluation = cps.evaluate(test, h=2)
    assert evaluation.loc[0, "model"] == "Model"
    assert evaluation.loc[0, "level"] == "90%"

    direct_quantiles = forecast.ppf([0.1, 0.5, 0.9])
    direct_cdf = forecast.cdf(forecast.to_frame()["Model"].to_numpy()[:, None])
    assert {"Model-q-10", "Model-q-50", "Model-q-90"} <= set(direct_quantiles)
    assert "Model-cdf" in direct_cdf


def test_discrete_series_cps_supports_pmf_and_integer_quantiles(
    nixtla_learner, dispersion_learner, panel
):
    cps = DiscreteTimeSeriesConformalPredictiveSystem(
        nixtla_learner, dispersion_learner, horizon=2, n_windows=2
    ).fit(panel, n_jobs=1)

    forecast = cps.predict_distribution(h=2)
    median = forecast.ppf(0.5)
    lower = forecast.ppf(0.01)
    masses = forecast.pmf(np.array([10, 11]))
    assert np.issubdtype(median["Model-q-50"].dtype, np.integer)
    assert np.all(lower["Model-q-1"] >= 0)
    assert np.all(masses[["Model-pmf-10", "Model-pmf-11"]] >= 0)


def test_discrete_series_cps_rejects_noninteger_target(
    nixtla_learner, dispersion_learner, panel
):
    panel.loc[0, "y"] = 0.5
    cps = DiscreteTimeSeriesConformalPredictiveSystem(
        nixtla_learner, dispersion_learner, horizon=2, n_windows=2
    )
    with pytest.raises(ValueError, match="finite integers"):
        cps.fit(panel, n_jobs=1)


def test_series_cps_rejects_multiple_forecast_models(
    nixtla_learner, dispersion_learner, panel
):
    original_predict = nixtla_learner.predict.side_effect

    def predict_with_two_models(*args, **kwargs):
        result = original_predict(*args, **kwargs)
        result["OtherModel"] = result["Model"] + 1.0
        return result

    nixtla_learner.predict.side_effect = predict_with_two_models
    cps = ContinuousTimeSeriesConformalPredictiveSystem(
        nixtla_learner, dispersion_learner, horizon=2, n_windows=2
    )

    with pytest.raises(ValueError, match="exactly one model"):
        cps.fit(panel, n_jobs=1)


def test_newsvendor_accepts_self_contained_series_forecast(
    nixtla_learner, dispersion_learner, panel
):
    cps = DiscreteTimeSeriesConformalPredictiveSystem(
        nixtla_learner, dispersion_learner, horizon=2, n_windows=2
    ).fit(panel, n_jobs=1)
    forecast = cps.predict_distribution(h=2)

    decision = NewsvendorSolver.optimize_distribution(
        forecast, underage_cost=8.0, overage_cost=2.0
    )
    masses = NewsvendorSolver.pmf_distribution(forecast, units=[0, 1])
    benefit = NewsvendorSolver.marginal_benefit_distribution(
        forecast, underage_cost=8.0, overage_cost=2.0, units=[0, 1]
    )

    assert {"critical_ratio", "y_optimal"} <= set(decision)
    assert {"P(Y=0)", "P(Y=1)"} <= set(masses)
    assert {"MB(k=0)", "MB(k=1)"} <= set(benefit)
