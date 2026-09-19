# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

import numpy as np
import pandas as pd
import pytest

from tinyconformal.evaluation import (
    ClassifierEvaluator,
    CPSEvaluator,
    FirstStageEvaluator,
    PanelEvaluator,
    RegressorEvaluator,
)


class _Distribution:
    def __init__(self, centers):
        self.centers = np.asarray(centers, dtype=float)

    def __len__(self):
        return len(self.centers)

    def ppf(self, probabilities):
        probabilities = np.atleast_1d(probabilities).astype(float)
        return self.centers[:, None] + 2.0 * (probabilities - 0.5)

    def interval(self, coverage):
        alpha = 1.0 - coverage
        return self.ppf([alpha / 2.0, 1.0 - alpha / 2.0])


class _PanelForecast:
    model = "Model"

    def __init__(self, frame, centers):
        self._frame = frame
        self.distribution = _Distribution(centers)

    def to_frame(self):
        return self._frame.copy()


def test_classifier_evaluator_computes_expected_set_metrics():
    frame = ClassifierEvaluator.evaluate_set(
        y_true=[0, 1, 1, 0],
        prediction_sets=[[1, 0], [0, 1], [1, 1], [0, 0]],
        coverage=0.9,
    )
    result = frame.iloc[0]

    assert result["coverage"] == pytest.approx(0.9)
    assert result["coverage_rate"] == pytest.approx(0.75)
    assert result["set_size_mean"] == pytest.approx(1.0)
    assert result["singleton_rate"] == pytest.approx(0.5)
    assert result["empty_rate"] == pytest.approx(0.25)
    assert result["n_obs"] == 4
    assert pd.api.types.is_integer_dtype(frame["n_obs"])


def test_classifier_evaluator_computes_point_and_probability_metrics():
    result = ClassifierEvaluator.evaluate_classification(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 1, 1],
        y_prob=[[0.8, 0.2], [0.4, 0.6], [0.2, 0.8], [0.1, 0.9]],
    ).iloc[0]

    assert result["accuracy"] == pytest.approx(0.75)
    assert result["balanced_accuracy"] == pytest.approx(0.75)
    assert result["bookmaker_informedness"] == pytest.approx(0.5)
    assert result["fpr"] == pytest.approx(0.5)
    assert result["log_loss"] >= 0.0
    assert 0.0 <= result["ece"] <= 1.0


def test_classifier_evaluator_omits_probability_metrics_without_probabilities():
    result = ClassifierEvaluator.evaluate_classification([0, 1], [0, 1])

    assert "log_loss" not in result
    assert "ece" not in result


@pytest.mark.parametrize("name", ["y_true", "y_pred"])
def test_classifier_evaluator_requires_zero_one_labels(name):
    values = {"y_true": [0, 1], "y_pred": [0, 1]}
    values[name] = ["no", "yes"]

    with pytest.raises(ValueError, match="labels 0 and 1"):
        ClassifierEvaluator.evaluate_classification(**values)


def test_classifier_evaluator_validates_probabilities_and_sets():
    with pytest.raises(ValueError, match="sum to 1"):
        ClassifierEvaluator.evaluate_classification(
            [0, 1], [0, 1], [[0.8, 0.3], [0.2, 0.8]]
        )
    with pytest.raises(ValueError, match="boolean or 0/1"):
        ClassifierEvaluator.evaluate_set([0, 1], [[1, 2], [0, 1]], 0.9)


def test_regressor_evaluator_computes_exact_interval_metrics():
    frame = RegressorEvaluator.evaluate(
        [15.0, 25.0], [[10.0, 20.0], [10.0, 20.0]], coverage=0.9
    )
    result = frame.iloc[0]

    assert result["coverage_rate"] == pytest.approx(0.5)
    assert result["interval_width_mean"] == pytest.approx(10.0)
    assert result["mwis"] == pytest.approx(60.0)
    assert result["n_obs"] == 2
    assert pd.api.types.is_integer_dtype(frame["n_obs"])


def test_regressor_evaluator_rejects_invalid_intervals():
    with pytest.raises(ValueError, match="shape"):
        RegressorEvaluator.evaluate([1.0, 2.0], [[0.0, 2.0]], coverage=0.9)
    with pytest.raises(ValueError, match="lower interval bounds"):
        RegressorEvaluator.evaluate(
            [1.0, 2.0], [[2.0, 0.0], [1.0, 3.0]], coverage=0.9
        )


def test_cps_evaluator_computes_interval_and_distribution_metrics():
    distribution = _Distribution([10.0, 11.0])

    intervals = CPSEvaluator.evaluate_interval(
        [10.0, 11.0], distribution, coverages=[0.5, 0.9]
    )
    result = CPSEvaluator.evaluate_distribution(
        [10.0, 11.0], distribution, scale=2.0
    ).iloc[0]

    assert list(intervals["coverage"]) == [0.5, 0.9]
    assert np.all(intervals["coverage_rate"] == 1.0)
    assert result["crps"] >= 0.0
    assert result["ncrps"] == pytest.approx(result["crps"] / 2.0)
    assert pd.api.types.is_integer_dtype(intervals["n_obs"])


@pytest.mark.parametrize("scale", [0.0, -1.0, np.inf, np.nan])
def test_cps_evaluator_rejects_invalid_ncrps_scale(scale):
    with pytest.raises(ValueError, match="strictly positive"):
        CPSEvaluator.evaluate_distribution([10.0], _Distribution([10.0]), scale)


def test_cps_evaluator_requires_distribution_alignment():
    with pytest.raises(ValueError, match="equal length"):
        CPSEvaluator.evaluate_distribution([10.0, 11.0], _Distribution([10.0]))


def test_panel_evaluator_infers_interval_pairs_and_aligns_by_keys():
    dates = pd.date_range("2024-01-01", periods=2)
    forecast = pd.DataFrame(
        {
            "unique_id": ["a", "a"],
            "ds": dates,
            "Model-lo-90": [10.0, 10.0],
            "Model-hi-90": [20.0, 20.0],
            "Model-lo-90-cqr": [10.0, 10.0],
            "Model-hi-90-cqr": [20.0, 20.0],
        }
    )
    y_true = pd.DataFrame(
        {"unique_id": ["a", "a"], "ds": dates[::-1], "y": [25.0, 15.0]}
    )

    result = PanelEvaluator.evaluate_interval(y_true, forecast)

    assert set(result["model"]) == {"Model", "Model-cqr"}
    assert np.all(result["coverage_rate"] == 0.5)
    assert np.all(result["interval_width_mean"] == 10.0)
    assert np.allclose(result["mwis"], 60.0)


def test_panel_evaluator_computes_per_series_ncrps():
    dates = pd.date_range("2024-01-01", periods=2)
    frame = pd.DataFrame(
        {"unique_id": ["a", "b"], "ds": dates, "Model": [10.0, 20.0]}
    )
    forecast = _PanelForecast(frame, [10.0, 20.0])
    y_true = frame[["unique_id", "ds"]].assign(y=[10.0, 20.0])
    train = pd.DataFrame(
        {"unique_id": ["a", "a", "b", "b"], "y": [9.0, 11.0, 18.0, 22.0]}
    )

    result = PanelEvaluator.evaluate_distribution(y_true, forecast, train)

    assert set(result["unique_id"]) == {"a", "b"}
    assert np.all(result["crps"] >= 0.0)
    assert np.all(np.isfinite(result["ncrps"]))
    assert result["n_obs"].sum() == 2
    assert pd.api.types.is_integer_dtype(result["n_obs"])


def test_panel_evaluator_rejects_duplicate_or_missing_targets():
    dates = pd.date_range("2024-01-01", periods=2)
    forecast = pd.DataFrame(
        {
            "unique_id": ["a", "a"],
            "ds": dates,
            "Model-lo-90": [0.0, 0.0],
            "Model-hi-90": [2.0, 2.0],
        }
    )
    duplicate = pd.DataFrame(
        {"unique_id": ["a", "a"], "ds": [dates[0], dates[0]], "y": [1.0, 1.0]}
    )
    missing = pd.DataFrame(
        {"unique_id": ["a"], "ds": [dates[0]], "y": [1.0]}
    )

    with pytest.raises(ValueError, match="duplicate"):
        PanelEvaluator.evaluate_interval(duplicate, forecast)
    with pytest.raises(ValueError, match="target for every forecast row"):
        PanelEvaluator.evaluate_interval(missing, forecast)


def test_first_stage_evaluator_uses_default_time_series_columns():
    backtest = pd.DataFrame(
        {
            "unique_id": ["a", "b", "a", "b"],
            "ds": [2, 2, 1, 1],
            "y": [2.0, 10.0, 1.0, 10.0],
            "forecast": [3.0, 10.0, 1.0, 10.0],
        }
    )

    result = FirstStageEvaluator.evaluate(backtest, prediction_col="forecast")

    assert result.loc[0, "forecast_instability"] == pytest.approx(0.3333)


def test_first_stage_evaluator_requires_time_series_columns():
    tabular = pd.DataFrame({"y": [1.0, 2.0], "y_pred": [1.0, 2.0]})

    with pytest.raises(KeyError, match="unique_id.*ds"):
        FirstStageEvaluator.evaluate(tabular)
