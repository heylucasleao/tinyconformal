from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from tinyconformal.series import (
    DistributionalConformalPredictiveSystemTimeSeriesRegressor,
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
def quantile_learner():
    learner = MagicMock()
    learner.fit.return_value = learner

    def predict(h, X_df=None):
        ids = ["a", "b"] if X_df is None else sorted(X_df["unique_id"].unique())
        dates = pd.date_range("2026-01-13", periods=h, freq="D")
        centers = np.tile(np.arange(h, dtype=float) + 10.0, len(ids))
        return pd.DataFrame(
            {
                "unique_id": np.repeat(ids, h),
                "ds": list(dates) * len(ids),
                "Model-q-10": centers - 2.0,
                "Model-q-50": centers,
                "Model-q-90": centers + 2.0,
            }
        )

    learner.predict.side_effect = predict
    return learner


@pytest.fixture
def columns():
    return {0.1: "Model-q-10", 0.5: "Model-q-50", 0.9: "Model-q-90"}


def test_series_dcp_calibrates_horizon_specific_pits(
    quantile_learner, panel, columns
):
    dcp = DistributionalConformalPredictiveSystemTimeSeriesRegressor(
        quantile_learner, horizon=2, n_windows=2, quantile_columns=columns
    ).fit(panel, n_jobs=1)

    assert dcp.ncscores_["Model"].shape == (4, 2)
    assert np.all((dcp.ncscores_["Model"] >= 0) & (dcp.ncscores_["Model"] <= 1))
    frame, distributions = dcp.predict_distribution(h=2)
    assert len(frame) == len(distributions["Model"]) == 4
    assert distributions["Model"].calibration_pits.shape == (4, 4)


def test_series_dcp_quantiles_intervals_and_evaluate(
    quantile_learner, panel, columns
):
    dcp = DistributionalConformalPredictiveSystemTimeSeriesRegressor(
        quantile_learner,
        horizon=2,
        n_windows=2,
        alpha=0.1,
        quantile_columns=columns,
    ).fit(panel, n_jobs=1)

    quantiles = dcp.predict_quantiles([0.1, 0.5, 0.9], h=2)
    assert {"Model-q-10-dcp", "Model-q-50-dcp", "Model-q-90-dcp"} <= set(
        quantiles
    )
    intervals = dcp.predict_interval(h=2)
    assert {"Model-lo-90-dcp", "Model-hi-90-dcp"} <= set(intervals)

    test = intervals[["unique_id", "ds"]].copy()
    test["y"] = 10.0
    result = dcp.evaluate(test, h=2)
    assert result.loc[0, "model"] == "Model"
    assert result.loc[0, "level"] == "90%"


def test_series_dcp_supports_nested_multi_model_mapping(
    quantile_learner, columns
):
    dcp = DistributionalConformalPredictiveSystemTimeSeriesRegressor(
        quantile_learner,
        horizon=2,
        quantile_columns={"Model": columns},
    )
    assert list(dcp.quantile_columns_) == ["Model"]


def test_series_dcp_rejects_invalid_flat_column_names(quantile_learner):
    with pytest.raises(ValueError, match="column names must follow"):
        DistributionalConformalPredictiveSystemTimeSeriesRegressor(
            quantile_learner,
            horizon=2,
            quantile_columns={0.1: "low", 0.9: "high"},
        )
