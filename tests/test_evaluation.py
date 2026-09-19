# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

import pandas as pd
import pytest

from tinyconformal.evaluation import FirstStageEvaluator


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
