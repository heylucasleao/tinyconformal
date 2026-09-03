from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from tinyconformal.series import ConformalizedQuantileTimeSeriesRegressor


@pytest.fixture
def sample_time_series_data():
    """Generates a synthetic time series DataFrame for calibration and testing."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    df_1 = pd.DataFrame(
        {
            "unique_id": ["series_1"] * 30,
            "ds": dates,
            "y": np.linspace(10, 50, 30) + np.random.normal(0, 1, 30),
            "exog_feat": np.random.randn(30),
        }
    )
    df_2 = pd.DataFrame(
        {
            "unique_id": ["series_2"] * 30,
            "ds": dates,
            "y": np.linspace(20, 60, 30) + np.random.normal(0, 1, 30),
            "exog_feat": np.random.randn(30),
        }
    )
    return pd.concat([df_1, df_2], ignore_index=True)


@pytest.fixture
def mock_quantile_learner_single():
    """Mock estimator returning a single interval pair ('LGBM-lo-90', 'LGBM-hi-90')."""
    learner = MagicMock()
    learner.fit.return_value = learner

    def mock_predict(h, X_df=None):
        dates = pd.date_range("2024-01-31", periods=h, freq="D")
        uids = (
            X_df["unique_id"].unique()
            if X_df is not None and "unique_id" in X_df
            else ["series_1", "series_2"]
        )
        records = []
        for uid in uids:
            for d in dates:
                records.append(
                    {"unique_id": uid, "ds": d, "LGBM-lo-90": 10.0, "LGBM-hi-90": 20.0}
                )
        return pd.DataFrame(records)

    learner.predict.side_effect = mock_predict
    return learner


@pytest.fixture
def mock_quantile_learner_multi():
    """Mock estimator returning multiple interval pairs."""
    learner = MagicMock()
    learner.fit.return_value = learner

    def mock_predict(h, X_df=None):
        dates = pd.date_range("2024-01-31", periods=h, freq="D")
        uids = (
            X_df["unique_id"].unique()
            if X_df is not None and "unique_id" in X_df
            else ["series_1", "series_2"]
        )
        records = []
        for uid in uids:
            for d in dates:
                records.append(
                    {
                        "unique_id": uid,
                        "ds": d,
                        "LGBM-lo-90": 5.0,
                        "LGBM-hi-90": 25.0,
                        "LGBM-lo-50": 10.0,
                        "LGBM-hi-50": 20.0,
                    }
                )
        return pd.DataFrame(records)

    learner.predict.side_effect = mock_predict
    return learner


# --- Column Normalization & Renaming Tests ---


def test_normalize_intervals_single_tuple(mock_quantile_learner_single):
    """Verify single tuple input converts to a list of tuples using intervals."""
    cqr = ConformalizedQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_single,
        horizon=5,
        intervals=("LGBM-lo-90", "LGBM-hi-90"),
    )
    assert cqr.intervals_ == [("LGBM-lo-90", "LGBM-hi-90")]


def test_normalize_intervals_list_of_tuples(mock_quantile_learner_multi):
    """Verify normalization when given a list of tuples/lists via intervals."""
    pairs = [("LGBM-lo-90", "LGBM-hi-90"), ["LGBM-lo-50", "LGBM-hi-50"]]
    cqr = ConformalizedQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_multi, horizon=5, intervals=pairs
    )
    assert cqr.intervals_ == [
        ("LGBM-lo-90", "LGBM-hi-90"),
        ("LGBM-lo-50", "LGBM-hi-50"),
    ]


@pytest.mark.parametrize(
    "invalid_cols",
    ["invalid_string", ("only_one_col",), [("low", "high", "extra")], 12345],
)
def test_normalize_intervals_invalid_raises_error(
    mock_quantile_learner_single, invalid_cols
):
    """Ensure ValueError is raised when invalid intervals formats are passed."""
    with pytest.raises(ValueError, match="intervals must be a tuple of 2 column names"):
        ConformalizedQuantileTimeSeriesRegressor(
            learner=mock_quantile_learner_single, horizon=5, intervals=invalid_cols
        )


def test_invalid_interval_col_pattern_raises_error(mock_quantile_learner_single):
    """Ensure ValueError is raised when column names don't match <model>-(lo|hi)-<level>."""
    with pytest.raises(ValueError, match="Invalid lower quantile column name"):
        ConformalizedQuantileTimeSeriesRegressor(
            learner=mock_quantile_learner_single,
            horizon=5,
            intervals=("invalid_lo_format", "LGBM-hi-90"),
        )


# --- Nonconformity Scores and Residual Calculations ---


def test_generate_residuals(mock_quantile_learner_single):
    """Test correctness of CQR nonconformity score computation: max(q_low - y, y - q_high)."""
    cqr = ConformalizedQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_single,
        horizon=3,
        intervals=("LGBM-lo-90", "LGBM-hi-90"),
    )
    q_low = np.array([10.0, 10.0, 10.0])
    q_high = np.array([20.0, 20.0, 20.0])

    y_true = np.array([15.0, 25.0, 5.0])

    residuals = cqr._generate_residuals(q_low, q_high, y_true)
    np.testing.assert_array_equal(residuals, np.array([-5.0, 5.0, 5.0]))


def test_sample_correction(mock_quantile_learner_single):
    """Test finite-sample quantile adjustment computation."""
    cqr = ConformalizedQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_single,
        horizon=5,
        intervals=("LGBM-lo-90", "LGBM-hi-90"),
    )
    cqr.n = 100
    q_level = cqr._sample_correction(alpha=0.05)
    # rank = ceil(101 * 0.95) = 96, mapped exactly for method="higher".
    assert pytest.approx(q_level, abs=1e-4) == 95 / 99


# --- Validation and Backtesting Tests ---


def test_sequential_backtesting_insufficient_time_steps(mock_quantile_learner_single):
    """Raise ValueError if time series lacks sufficient time steps for backtesting."""
    cqr = ConformalizedQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_single,
        horizon=10,
        n_windows=5,
        intervals=("LGBM-lo-90", "LGBM-hi-90"),
    )
    short_df = pd.DataFrame(
        {
            "unique_id": ["s1"] * 10,
            "ds": pd.date_range("2024-01-01", periods=10),
            "y": np.arange(10),
        }
    )
    with pytest.raises(ValueError, match="Time series has 10 unique time steps"):
        cqr.fit(short_df)


def test_sequential_backtesting_missing_quantile_column(
    mock_quantile_learner_single, sample_time_series_data
):
    """Ensure KeyError is raised when configured interval columns are missing from predictions."""
    cqr = ConformalizedQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_single,
        horizon=3,
        n_windows=2,
        intervals=("LGBM-lo-90", "LGBM-hi-90"),
    )
    mock_quantile_learner_single.predict.side_effect = lambda h, X_df=None: (
        pd.DataFrame(
            {
                "unique_id": ["series_1"] * h,
                "ds": pd.date_range("2024-01-31", periods=h),
            }
        )
    )

    with pytest.raises(KeyError, match="were not found in forecast output"):
        cqr.fit(sample_time_series_data)


def test_window_residuals_align_shuffled_forecasts_by_keys(
    mock_quantile_learner_single,
):
    """Quantile residuals must align by series and timestamp, not row order."""
    cqr = ConformalizedQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_single,
        horizon=2,
        intervals=("LGBM-lo-90", "LGBM-hi-90"),
    )
    val_df = pd.DataFrame(
        {
            "unique_id": ["s1", "s1", "s2", "s2"],
            "ds": [1, 2, 1, 2],
            "y": [10.0, 20.0, 30.0, 40.0],
        }
    )
    fcst = pd.DataFrame(
        {
            "unique_id": ["s2", "s1", "s2", "s1"],
            "ds": [2, 1, 1, 2],
            "LGBM-lo-90": [39.0, 9.0, 29.0, 19.0],
            "LGBM-hi-90": [41.0, 11.0, 31.0, 21.0],
        }
    )
    residuals = {}

    cqr._compute_window_residuals(fcst, val_df, 2, residuals)

    np.testing.assert_array_equal(
        residuals["LGBM-lo-90:LGBM-hi-90"][0],
        np.full((2, 2), -1.0),
    )


def test_backtesting_does_not_fit_original_learner_per_window(
    mock_quantile_learner_single, sample_time_series_data
):
    """Only the final full-data fit should mutate the user-provided learner."""
    cqr = ConformalizedQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_single,
        horizon=3,
        n_windows=2,
        intervals=("LGBM-lo-90", "LGBM-hi-90"),
    )

    cqr.fit(sample_time_series_data, n_jobs=1)

    assert mock_quantile_learner_single.fit.call_count == 1


# --- Fitting and Interval Prediction Tests ---


def test_fit_and_ncscores_structure(
    mock_quantile_learner_single, sample_time_series_data
):
    """Verify that fitting populates ncscores_ correctly and updates calibration sample size n."""
    cqr = ConformalizedQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_single,
        horizon=3,
        n_windows=2,
        intervals=("LGBM-lo-90", "LGBM-hi-90"),
    )
    cqr.fit(sample_time_series_data)

    pair_key = "LGBM-lo-90:LGBM-hi-90"
    assert pair_key in cqr.ncscores_
    assert set(cqr.ncscores_[pair_key]) == {"series_1", "series_2"}
    assert all(scores.shape == (2, 3) for scores in cqr.ncscores_[pair_key].values())
    assert cqr.n == 2


def test_tscqr_bounds_are_calibrated_by_unique_id(mock_quantile_learner_single):
    """Each series must use only its own CQR nonconformity scores."""
    cqr = ConformalizedQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_single,
        horizon=2,
        n_windows=2,
        intervals=("LGBM-lo-90", "LGBM-hi-90"),
    )
    pair_key = "LGBM-lo-90:LGBM-hi-90"
    cqr.ncscores_ = {
        pair_key: {
            "stable": np.array([[1.0, 2.0], [1.0, 2.0]]),
            "volatile": np.array([[10.0, 20.0], [10.0, 20.0]]),
        }
    }
    cqr.n = 2

    lower, upper = cqr._compute_bounds(
        q_low=np.full(4, 90.0),
        q_high=np.full(4, 110.0),
        pair_key=pair_key,
        h=2,
        prediction_ids=np.array(["stable", "stable", "volatile", "volatile"]),
        alpha=0.1,
    )

    np.testing.assert_array_equal(lower, [89.0, 88.0, 80.0, 70.0])
    np.testing.assert_array_equal(upper, [111.0, 112.0, 120.0, 130.0])


def test_predict_interval_single_pair_formatting(
    mock_quantile_learner_single, sample_time_series_data
):
    """Validate output column formatting with -cqr suffix for a single interval pair."""
    cqr = ConformalizedQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_single,
        horizon=3,
        n_windows=2,
        intervals=("LGBM-lo-90", "LGBM-hi-90"),
    )
    cqr.fit(sample_time_series_data)
    pred_df = cqr.predict_interval(h=3)

    assert "LGBM-lo-90-cqr" in pred_df.columns
    assert "LGBM-hi-90-cqr" in pred_df.columns


def test_predict_interval_multi_pair_formatting(
    mock_quantile_learner_multi, sample_time_series_data
):
    """Validate output column formatting with -cqr suffix for multiple interval pairs."""
    pairs = [("LGBM-lo-90", "LGBM-hi-90"), ("LGBM-lo-50", "LGBM-hi-50")]
    cqr = ConformalizedQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_multi,
        horizon=3,
        n_windows=2,
        intervals=pairs,
    )
    cqr.fit(sample_time_series_data)
    pred_df = cqr.predict_interval(h=3)

    assert "LGBM-lo-90-cqr" in pred_df.columns
    assert "LGBM-hi-90-cqr" in pred_df.columns
    assert "LGBM-lo-50-cqr" in pred_df.columns
    assert "LGBM-hi-50-cqr" in pred_df.columns


def test_evaluate_output_structure_and_metrics(
    mock_quantile_learner_multi, sample_time_series_data
):
    """Verify structure, columns, and metric calculations in evaluate() output."""
    pairs = [("LGBM-lo-90", "LGBM-hi-90"), ("LGBM-lo-50", "LGBM-hi-50")]
    cqr = ConformalizedQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_multi,
        horizon=3,
        n_windows=2,
        intervals=pairs,
    )
    cqr.fit(sample_time_series_data)

    test_dates = pd.date_range("2024-01-31", periods=3, freq="D")
    df_test = pd.DataFrame(
        {
            "unique_id": ["series_1"] * 3 + ["series_2"] * 3,
            "ds": list(test_dates) * 2,
            "y": [15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
            "exog_feat": [0.0] * 6,
        }
    )

    eval_df = cqr.evaluate(df_test=df_test, h=3)

    expected_cols = [
        "model",
        "level",
        "alpha",
        "coverage_rate",
        "interval_width_mean",
        "mwis",
    ]
    assert list(eval_df.columns) == expected_cols

    assert len(eval_df) == 4
    assert set(eval_df["model"]) == {"LGBM", "LGBM-cqr"}
    assert set(eval_df["level"]) == {"90%", "50%"}
    assert np.allclose(eval_df.loc[eval_df["level"] == "90%", "alpha"], 0.10)
    assert np.allclose(eval_df.loc[eval_df["level"] == "50%", "alpha"], 0.50)
    prediction_input = mock_quantile_learner_multi.predict.call_args.kwargs["X_df"]
    assert list(prediction_input.columns) == ["unique_id", "ds", "exog_feat"]
    assert "y" not in prediction_input


def test_predict_rejects_unbalanced_forecast_panel(
    mock_quantile_learner_single, sample_time_series_data
):
    cqr = ConformalizedQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_single,
        horizon=2,
        n_windows=2,
        intervals=("LGBM-lo-90", "LGBM-hi-90"),
    )
    cqr.fit(sample_time_series_data)
    mock_quantile_learner_single.predict.side_effect = None
    mock_quantile_learner_single.predict.return_value = pd.DataFrame(
        {
            "unique_id": ["series_1", "series_1", "series_2"],
            "ds": pd.to_datetime(["2024-01-31", "2024-02-01", "2024-01-31"]),
            "LGBM-lo-90": [10.0] * 3,
            "LGBM-hi-90": [20.0] * 3,
        }
    )

    with pytest.raises(ValueError, match="exactly 2 rows for every series"):
        cqr.predict_interval(h=2)


def test_predict_rejects_crossing_quantiles(
    mock_quantile_learner_single, sample_time_series_data
):
    cqr = ConformalizedQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_single,
        horizon=1,
        n_windows=2,
        intervals=("LGBM-lo-90", "LGBM-hi-90"),
    )
    cqr.fit(sample_time_series_data)
    mock_quantile_learner_single.predict.side_effect = None
    mock_quantile_learner_single.predict.return_value = pd.DataFrame(
        {
            "unique_id": ["series_1", "series_2"],
            "ds": pd.to_datetime(["2024-01-31", "2024-01-31"]),
            "LGBM-lo-90": [21.0, 10.0],
            "LGBM-hi-90": [20.0, 20.0],
        }
    )

    with pytest.raises(ValueError, match="Crossing quantiles detected"):
        cqr.predict_interval(h=1)


def test_evaluate_metric_values_correctness(mock_quantile_learner_single):
    """Test exact mathematical outputs of evaluate() on deterministic bounds."""
    cqr = ConformalizedQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_single,
        horizon=2,
        intervals=("LGBM-lo-90", "LGBM-hi-90"),
    )

    # Mock do predict_interval contendo tanto os limites não-corrigidos quanto os corrigidos por CQR
    cqr.predict_interval = MagicMock(
        return_value=pd.DataFrame(
            {
                "unique_id": ["s1", "s1"],
                "ds": pd.date_range("2024-01-01", periods=2),
                "LGBM-lo-90": [10.0, 10.0],
                "LGBM-hi-90": [20.0, 20.0],
                "LGBM-lo-90-cqr": [10.0, 10.0],
                "LGBM-hi-90-cqr": [20.0, 20.0],
            }
        )
    )

    # y = [15.0 (coberto), 25.0 (fora/acima por 5 unidades)]
    df_test = pd.DataFrame(
        {
            "unique_id": ["s1", "s1"],
            "ds": pd.date_range("2024-01-01", periods=2),
            "y": [15.0, 25.0],
        }
    )

    eval_df = cqr.evaluate(df_test=df_test, h=2)

    # Filtra apenas a métrica do CQR para asserção determinística
    cqr_eval = eval_df[eval_df["model"] == "LGBM-cqr"].iloc[0]

    # Coverage: 1 de 2 cobertos -> 0.5
    assert cqr_eval["coverage_rate"] == 0.5
    # Width: (20 - 10) = 10.0
    assert cqr_eval["interval_width_mean"] == 10.0
    # MWIS:
    # Obs 1: 10 + 0 + 0 = 10
    # Obs 2: 10 + (2/0.1)*(25 - 20) = 10 + 20*5 = 110
    # Mean MWIS: (10 + 110) / 2 = 60.0
    assert cqr_eval["mwis"] == 60.0


def test_tscqr_predict_raw(mock_quantile_learner_single, sample_time_series_data):
    """Verify _predict_raw returns valid 2D numpy array format without NaNs."""
    cqr = ConformalizedQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_single,
        horizon=3,
        n_windows=2,
        intervals=("LGBM-lo-90", "LGBM-hi-90"),
    )
    cqr.fit(sample_time_series_data)
    preds_raw = cqr._predict_raw(h=3)

    assert isinstance(preds_raw, np.ndarray)
    assert preds_raw.ndim == 2
    assert preds_raw.shape == (2, 3)
    assert np.issubdtype(preds_raw.dtype, np.number)
    assert not np.isnan(preds_raw).any()
    assert not np.isinf(preds_raw).any()


# --- Pattern Matching & Quantile Validation Errors ---


def test_quantile_pair_mismatch_model_raises_error(mock_quantile_learner_single):
    """Ensure ValueError is raised if pair models don't match (e.g. LGBM vs XGB)."""
    with pytest.raises(
        ValueError, match="Model name mismatch in quantile pair: 'LGBM' vs 'XGB'"
    ):
        ConformalizedQuantileTimeSeriesRegressor(
            learner=mock_quantile_learner_single,
            horizon=3,
            intervals=("LGBM-lo-90", "XGB-hi-90"),
        )


def test_quantile_pair_mismatch_level_raises_error(mock_quantile_learner_single):
    """Ensure ValueError is raised if nominal levels don't match (e.g. 90 vs 50)."""
    with pytest.raises(
        ValueError, match="Coverage level mismatch in quantile pair: '90' vs '50'"
    ):
        ConformalizedQuantileTimeSeriesRegressor(
            learner=mock_quantile_learner_single,
            horizon=3,
            intervals=("LGBM-lo-90", "LGBM-hi-50"),
        )


def test_quantile_pair_invalid_bound_indicator_raises_error(
    mock_quantile_learner_single,
):
    """Ensure ValueError is raised if bound tag is not 'lo' or 'hi'."""
    with pytest.raises(ValueError, match="Invalid lower quantile column name"):
        ConformalizedQuantileTimeSeriesRegressor(
            learner=mock_quantile_learner_single,
            horizon=3,
            intervals=("LGBM-mid-90", "LGBM-hi-90"),
        )


def test_predict_requires_configured_quantile_columns(
    mock_quantile_learner_single, sample_time_series_data
):
    cqr = ConformalizedQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_single,
        horizon=2,
        n_windows=2,
        intervals=("LGBM-lo-90", "LGBM-hi-90"),
    ).fit(sample_time_series_data)
    mock_quantile_learner_single.predict.side_effect = None
    mock_quantile_learner_single.predict.return_value = pd.DataFrame(
        {
            "unique_id": ["series_1"] * 2 + ["series_2"] * 2,
            "ds": list(pd.date_range("2024-01-31", periods=2)) * 2,
            "LGBM-lo-90": [10.0] * 4,
        }
    )

    with pytest.raises(KeyError, match="LGBM-hi-90"):
        cqr.predict_interval(h=2)
