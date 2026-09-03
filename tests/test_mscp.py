from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from tinyconformal.series import MultiStepConformalTimeSeriesRegressor


@pytest.fixture
def sample_distribution_data():
    """Generates time series data for calibrating the distribution regressor."""
    dates = pd.date_range("2024-01-01", periods=25, freq="D")
    df_1 = pd.DataFrame(
        {
            "unique_id": ["id_1"] * 25,
            "ds": dates,
            "y": np.sin(np.linspace(0, 10, 25)) * 10 + 20,
        }
    )
    df_2 = pd.DataFrame(
        {
            "unique_id": ["id_2"] * 25,
            "ds": dates,
            "y": np.cos(np.linspace(0, 10, 25)) * 10 + 20,
        }
    )
    return pd.concat([df_1, df_2], ignore_index=True)


@pytest.fixture
def mock_point_learner():
    """Mock point forecasting learner adhering to the Nixtla interface."""
    learner = MagicMock()
    learner.fit.return_value = learner

    def mock_predict(h, X_df=None):
        dates = pd.date_range("2024-01-26", periods=h, freq="D")
        records = []
        for uid in ["id_1", "id_2"]:
            for d in dates:
                records.append({"unique_id": uid, "ds": d, "LGBMRegressor": 20.0})
        return pd.DataFrame(records)

    learner.predict.side_effect = mock_predict
    return learner


def test_coverage_rate(mock_point_learner):
    """Calculate empirical prediction interval coverage rate."""
    cdr = MultiStepConformalTimeSeriesRegressor(learner=mock_point_learner, horizon=3)
    y_true = np.array([10.0, 15.0, 20.0, 25.0])
    lower = np.array(
        [8.0, 12.0, 18.0, 26.0]
    )  # Last observation (25.0) falls outside bounds
    upper = np.array([12.0, 17.0, 22.0, 30.0])

    coverage = cdr._coverage_rate(y_true, lower, upper)
    assert coverage == 0.75


def test_interval_width_mean(mock_point_learner):
    """Verify mean prediction interval width calculation."""
    cdr = MultiStepConformalTimeSeriesRegressor(learner=mock_point_learner, horizon=3)
    lower = np.array([10.0, 20.0])
    upper = np.array([15.0, 30.0])

    width = cdr._interval_width_mean(lower, upper)
    assert width == 7.5


def test_mwi_score_calculation(mock_point_learner):
    """Evaluate Mean Winkler Interval Score logic with out-of-bounds penalties."""
    cdr = MultiStepConformalTimeSeriesRegressor(learner=mock_point_learner, horizon=3)
    alpha = 0.10

    # Instance 1: inside bounds (width = 10)
    # Instance 2: below lower bound (y=2, lower=5 -> penalty = (2/0.1) * (5 - 2) = 60)
    # Instance 3: above upper bound (y=20, upper=15 -> penalty = (2/0.1) * (20 - 15) = 100)
    y_true = np.array([10.0, 2.0, 20.0])
    lower = np.array([5.0, 5.0, 5.0])
    upper = np.array([15.0, 15.0, 15.0])

    # Widths = [10, 10, 10]
    # Penalties = [0, 60, 100]
    # Individual Scores = [10, 70, 110] -> Mean = 190 / 3 = 63.3333...
    mwis = cdr._mwi_score(y_true, lower, upper, alpha)
    assert pytest.approx(mwis, abs=1e-3) == 63.333


# --- Helper Methods and Validation Tests ---


def test_validate_columns_missing_raises_error(mock_point_learner):
    """Ensure validation error when required structural columns are missing."""
    cdr = MultiStepConformalTimeSeriesRegressor(learner=mock_point_learner, horizon=3)
    invalid_df = pd.DataFrame({"unique_id": ["id_1"], "ds": ["2024-01-01"]})

    with pytest.raises(ValueError, match="required columns are missing"):
        cdr._validate_columns(invalid_df)


def test_get_horizon_exceeds_fitted_horizon(mock_point_learner):
    """Ensure error when requesting a forecast horizon larger than calibrated."""
    cdr = MultiStepConformalTimeSeriesRegressor(learner=mock_point_learner, horizon=5)
    with pytest.raises(ValueError, match="exceeds fitted calibration horizon"):
        cdr._get_horizon(h=10)


def test_sample_correction_finite_bounds(mock_point_learner):
    """Verify finite-sample adjusted low and high quantile bounds."""
    cdr = MultiStepConformalTimeSeriesRegressor(learner=mock_point_learner, horizon=3)
    cdr.n = 20
    alpha = 0.10

    # The conformal ranks are floor(1.05) = 1 and ceil(19.95) = 20.
    # With method="higher", ranks map to (rank - 1) / (n - 1).
    low_q, high_q = cdr._sample_correction(alpha)
    assert pytest.approx(low_q, abs=1e-4) == 0.0
    assert pytest.approx(high_q, abs=1e-4) == 1.0


# --- Full MSCP Pipeline Tests ---


def test_mscp_fit_and_residuals(mock_point_learner, sample_distribution_data):
    """Verify fit process and nonconformity score matrix construction."""
    cdr = MultiStepConformalTimeSeriesRegressor(
        learner=mock_point_learner, horizon=3, n_windows=2
    )
    cdr.fit(sample_distribution_data)

    assert "LGBMRegressor" in cdr.ncscores_
    assert set(cdr.ncscores_["LGBMRegressor"]) == {"id_1", "id_2"}
    assert all(
        scores.shape == (2, 3) for scores in cdr.ncscores_["LGBMRegressor"].values()
    )
    assert cdr.n == 2


def test_mscp_predict_interval_output(mock_point_learner, sample_distribution_data):
    """Validate creation of conformal interval columns (<model>-lo-<level> and <model>-hi-<level>)."""
    cdr = MultiStepConformalTimeSeriesRegressor(
        learner=mock_point_learner, horizon=3, n_windows=2, alpha=0.05
    )
    cdr.fit(sample_distribution_data)
    pred_df = cdr.predict_interval(h=3)

    assert "LGBMRegressor" in pred_df.columns
    assert "LGBMRegressor-lo-95" in pred_df.columns
    assert "LGBMRegressor-hi-95" in pred_df.columns


def test_mscp_evaluate_dataframe(mock_point_learner, sample_distribution_data):
    """Test full evaluation pipeline returning summary metric DataFrame."""
    cdr = MultiStepConformalTimeSeriesRegressor(
        learner=mock_point_learner, horizon=3, n_windows=2, alpha=0.05
    )
    cdr.fit(sample_distribution_data)

    test_dates = pd.date_range("2024-01-26", periods=3, freq="D")
    test_df = pd.DataFrame(
        {
            "unique_id": ["id_1"] * 3 + ["id_2"] * 3,
            "ds": list(test_dates) * 2,
            "y": [20.0] * 6,
        }
    )
    eval_df = cdr.evaluate(df_test=test_df, h=3)

    expected_cols = [
        "model",
        "level",
        "alpha",
        "coverage_rate",
        "interval_width_mean",
        "mwis",
    ]
    for col in expected_cols:
        assert col in eval_df.columns

    assert len(eval_df) == 1
    assert eval_df["model"].iloc[0] == "LGBMRegressor"
    assert eval_df["level"].iloc[0] == "95%"


# --- Additional Unit Tests for Internal Methods in Base / MSCP ---


def test_invoke_parameter_filtering(mock_point_learner):
    """Verify _invoke correctly filters keyword arguments based on method signature."""
    cdr = MultiStepConformalTimeSeriesRegressor(learner=mock_point_learner, horizon=3)

    def dummy_method(a, b=2):
        return a + b

    # 'c' and 'd' should be filtered out without raising TypeError
    res = cdr._invoke(dummy_method, a=5, b=10, c=100, d=None)
    assert res == 15


def test_invoke_with_var_keywords(mock_point_learner):
    """Verify _invoke passes all non-None kwargs when method accepts **kwargs."""
    cdr = MultiStepConformalTimeSeriesRegressor(learner=mock_point_learner, horizon=3)

    def dummy_kw_method(a, **kwargs):
        return a + kwargs.get("c", 0)

    res = cdr._invoke(dummy_kw_method, a=5, c=20, d=None)
    assert res == 25


def test_infer_model_cols(mock_point_learner):
    """Test model column inference logic from output DataFrames."""
    cdr = MultiStepConformalTimeSeriesRegressor(learner=mock_point_learner, horizon=3)
    cdr.exog_cols_ = ["exog_1"]

    df_fcst = pd.DataFrame(
        {
            "unique_id": ["id_1"],
            "ds": ["2024-01-01"],
            "exog_1": [1.0],
            "ModelA": [10.0],
            "ModelB": [12.0],
        }
    )

    # Auto-infer excluding id, time, and exog columns
    cols = cdr._infer_model_cols(df_fcst)
    assert cols == ["ModelA", "ModelB"]

    # Explicit override via model_col_ attribute
    cdr.model_col_ = "ModelA"
    assert cdr._infer_model_cols(df_fcst) == ["ModelA"]


def test_infer_model_cols_raises_value_error(mock_point_learner):
    """Raise ValueError when no model prediction columns remain after filtering."""
    cdr = MultiStepConformalTimeSeriesRegressor(learner=mock_point_learner, horizon=3)
    df_empty = pd.DataFrame({"unique_id": ["id_1"], "ds": ["2024-01-01"]})
    with pytest.raises(ValueError, match="Could not infer any prediction model column"):
        cdr._infer_model_cols(df_empty)


def test_extract_predictions_and_target_sorting(mock_point_learner):
    """Test 2D array conversion and strict row/column sorting in pivot extraction."""
    cdr = MultiStepConformalTimeSeriesRegressor(learner=mock_point_learner, horizon=2)

    fcst_df = pd.DataFrame(
        {
            "unique_id": ["id_2", "id_2", "id_1", "id_1"],
            "ds": ["2024-01-02", "2024-01-01", "2024-01-02", "2024-01-01"],
            "LGBM": [20.0, 10.0, 40.0, 30.0],
        }
    )

    target_df = pd.DataFrame(
        {
            "unique_id": ["id_2", "id_2", "id_1", "id_1"],
            "ds": ["2024-01-02", "2024-01-01", "2024-01-02", "2024-01-01"],
            "y": [2.0, 1.0, 4.0, 3.0],
        }
    )

    preds_arr = cdr._extract_predictions(fcst_df)
    target_arr = cdr._extract_target(target_df)

    # Output array must be sorted by unique_id (id_1, id_2) and ds (01, 02)
    expected_preds = np.array([[30.0, 40.0], [10.0, 20.0]])  # id_1  # id_2
    expected_targets = np.array([[3.0, 4.0], [1.0, 2.0]])  # id_1  # id_2

    np.testing.assert_array_equal(preds_arr, expected_preds)
    np.testing.assert_array_equal(target_arr, expected_targets)


def test_compute_qhat(mock_point_learner):
    """Verify _compute_qhat correctly calls np.quantile with method='higher'."""
    cdr = MultiStepConformalTimeSeriesRegressor(learner=mock_point_learner, horizon=3)
    ncscore = np.array([1.0, 2.0, 5.0, 10.0])

    # 50th percentile (higher method) -> pick value >= 50th quantile
    q_val = cdr._compute_qhat(ncscore, q_level=0.5)
    assert q_val == 5.0


def test_mscp_compute_bounds_direct(mock_point_learner):
    """Directly test _compute_bounds transformation for MSCP signed residuals."""
    cdr = MultiStepConformalTimeSeriesRegressor(learner=mock_point_learner, horizon=2)
    cdr.ncscores_ = {"LGBM": {"id_1": np.array([[-2.0, -1.0], [2.0, 3.0]])}}
    cdr.n = 2

    y_hat = np.array([10.0, 20.0])  # 1 series, horizon 2

    # low_q = 0.025 -> q_low_h = min residuals [-2.0, -1.0]
    # high_q = 0.975 -> q_high_h = max residuals [2.0, 3.0]
    # lower_bound = y_hat - q_high = [10 - 2, 20 - 3] = [8.0, 17.0]
    # upper_bound = y_hat - q_low = [10 - (-2), 20 - (-1)] = [12.0, 21.0]
    lower, upper = cdr._compute_bounds(
        y_hat=y_hat,
        model_name="LGBM",
        h=2,
        prediction_ids=np.array(["id_1", "id_1"]),
        alpha=0.10,
    )

    np.testing.assert_array_equal(lower, np.array([8.0, 17.0]))
    np.testing.assert_array_equal(upper, np.array([12.0, 21.0]))


def test_window_residuals_align_shuffled_forecasts_by_keys(mock_point_learner):
    """Forecast row order must not change residual-to-horizon alignment."""
    cdr = MultiStepConformalTimeSeriesRegressor(learner=mock_point_learner, horizon=2)
    val_df = pd.DataFrame(
        {
            "unique_id": ["id_1", "id_1", "id_2", "id_2"],
            "ds": [1, 2, 1, 2],
            "y": [10.0, 20.0, 30.0, 40.0],
        }
    )
    fcst = pd.DataFrame(
        {
            "unique_id": ["id_2", "id_1", "id_2", "id_1"],
            "ds": [2, 1, 1, 2],
            "model": [44.0, 11.0, 33.0, 22.0],
        }
    )
    residuals = {}

    cdr._compute_window_residuals(fcst, val_df, 2, residuals)

    np.testing.assert_array_equal(
        residuals["model"][0], np.array([[1.0, 2.0], [3.0, 4.0]])
    )


def test_predict_before_fit_raises_clear_error(mock_point_learner):
    cdr = MultiStepConformalTimeSeriesRegressor(learner=mock_point_learner, horizon=2)

    with pytest.raises(RuntimeError, match="must be fitted before prediction"):
        cdr.predict_interval(h=2)


@pytest.mark.parametrize(
    ("parameter", "value", "message"),
    [
        ("alpha", 0.0, "alpha must be"),
        ("alpha", 1.0, "alpha must be"),
        ("horizon", 0, "positive integer"),
        ("n_windows", 0, "n_windows must be"),
    ],
)
def test_fit_rejects_invalid_calibration_parameters(
    mock_point_learner, sample_distribution_data, parameter, value, message
):
    kwargs = {"horizon": 2, "n_windows": 2, "alpha": 0.05, parameter: value}
    cdr = MultiStepConformalTimeSeriesRegressor(learner=mock_point_learner, **kwargs)

    with pytest.raises(ValueError, match=message):
        cdr.fit(sample_distribution_data)


def test_fit_rejects_non_positive_step_size(
    mock_point_learner, sample_distribution_data
):
    cdr = MultiStepConformalTimeSeriesRegressor(
        learner=mock_point_learner, horizon=2, n_windows=2
    )

    with pytest.raises(ValueError, match="step_size must be a positive integer"):
        cdr.fit(sample_distribution_data, step_size=0)


def test_sequential_backtesting_short_series_raises_value_error(mock_point_learner):
    """Raise ValueError in MSCP sequential_backtesting when validation start index <= 0."""
    cdr = MultiStepConformalTimeSeriesRegressor(
        learner=mock_point_learner, horizon=10, n_windows=5
    )
    short_df = pd.DataFrame(
        {
            "unique_id": ["s1"] * 5,
            "ds": pd.date_range("2024-01-01", periods=5),
            "y": np.arange(5),
        }
    )
    with pytest.raises(ValueError, match="Time series length is too short"):
        cdr._sequential_backtesting(short_df)


def test_get_alpha_and_get_horizon_defaults(mock_point_learner):
    """Verify resolution of default vs overridden alpha and horizon values."""
    cdr = MultiStepConformalTimeSeriesRegressor(
        learner=mock_point_learner, horizon=7, alpha=0.05
    )

    assert cdr._get_alpha(None) == 0.05
    assert cdr._get_alpha(0.10) == 0.10

    assert cdr._get_horizon(None) == 7
    assert cdr._get_horizon(5) == 5


def test_predict_raw_direct(mock_point_learner, sample_distribution_data):
    """Test direct execution of _predict_raw returning 2D numpy array of predictions."""
    cdr = MultiStepConformalTimeSeriesRegressor(
        learner=mock_point_learner, horizon=3, n_windows=2
    )
    cdr.fit(sample_distribution_data)
    preds_raw = cdr._predict_raw(h=3)

    assert isinstance(preds_raw, np.ndarray)
    assert preds_raw.shape == (2, 3)  # 2 series, horizon 3


def test_fit_and_predict_with_exogenous_features(
    mock_point_learner, sample_distribution_data
):
    """Test automatic exogenous columns extraction during fit and usage in predict_interval with X_df."""
    df = sample_distribution_data.copy()
    df["exog_var"] = 1.0

    cdr = MultiStepConformalTimeSeriesRegressor(
        learner=mock_point_learner, horizon=3, n_windows=2
    )
    cdr.fit(df)
    assert cdr.exog_cols_ == ["exog_var"]

    X_future = pd.DataFrame(
        {
            "unique_id": ["id_1"] * 3 + ["id_2"] * 3,
            "ds": list(pd.date_range("2024-01-26", periods=3)) * 2,
            "exog_var": [1.0] * 6,
        }
    )
    pred_df = cdr.predict_interval(h=3, X_df=X_future)
    assert "LGBMRegressor-lo-95" in pred_df.columns


def test_fit_empty_ncscores_raises_runtime_error(
    mock_point_learner, sample_distribution_data, monkeypatch
):
    """Ensure RuntimeError is raised when no nonconformity scores are extracted during backtesting."""
    cdr = MultiStepConformalTimeSeriesRegressor(
        learner=mock_point_learner, horizon=3, n_windows=2
    )
    # Force _sequential_backtesting to return empty residual dict
    monkeypatch.setattr(cdr, "_sequential_backtesting", lambda *args, **kwargs: {})

    with pytest.raises(RuntimeError, match="No nonconformity scores were extracted"):
        cdr.fit(sample_distribution_data)


def test_extract_predictions_no_model_col_raises_error(mock_point_learner):
    """Ensure ValueError is raised if forecast DataFrame contains only structural columns."""
    cdr = MultiStepConformalTimeSeriesRegressor(learner=mock_point_learner, horizon=2)
    invalid_fcst = pd.DataFrame({"unique_id": ["id_1"], "ds": ["2024-01-01"]})
    with pytest.raises(ValueError, match="No prediction model column was detected"):
        cdr._extract_predictions(invalid_fcst)


# --- Parameterization & Custom Columns Tests for MSCP ---


@pytest.mark.parametrize("h_val", [1, 2, 3])
def test_mscp_predict_interval_different_horizons(
    mock_point_learner, sample_distribution_data, h_val
):
    """Test predict_interval across different valid forecast horizons h <= horizon."""
    cdr = MultiStepConformalTimeSeriesRegressor(
        learner=mock_point_learner, horizon=3, n_windows=2
    )
    cdr.fit(sample_distribution_data)
    pred_df = cdr.predict_interval(h=h_val)

    assert len(pred_df) == 2 * h_val  # 2 series * h_val steps


def test_mscp_custom_column_names(mock_point_learner):
    """Verify MSCP pipeline using custom column names for id, time, and target."""
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    custom_df = pd.DataFrame(
        {
            "series_id": ["S1"] * 20 + ["S2"] * 20,
            "date_time": list(dates) * 2,
            "target_val": np.random.randn(40),
        }
    )

    learner = MagicMock()
    learner.fit.return_value = learner
    learner.predict.return_value = pd.DataFrame(
        {
            "series_id": ["S1"] * 2 + ["S2"] * 2,
            "date_time": list(pd.date_range("2024-01-21", periods=2)) * 2,
            "LGBMRegressor": [10.0] * 4,
        }
    )

    cdr = MultiStepConformalTimeSeriesRegressor(
        learner=learner,
        horizon=2,
        n_windows=2,
        id_col="series_id",
        time_col="date_time",
        target_col="target_val",
    )

    cdr.fit(custom_df)
    pred_df = cdr.predict_interval(h=2)

    assert "series_id" in pred_df.columns
    assert "date_time" in pred_df.columns
    assert "LGBMRegressor-lo-95" in pred_df.columns


def test_fit_default_step_size_fallback(mock_point_learner, sample_distribution_data):
    """Verify that fit() correctly falls back step_size to self.horizon when step_size=None."""
    cdr = MultiStepConformalTimeSeriesRegressor(
        learner=mock_point_learner, horizon=3, n_windows=2
    )
    # Executing fit without step_size parameter
    cdr.fit(sample_distribution_data, step_size=None)
    assert all(
        scores.shape == (2, 3) for scores in cdr.ncscores_["LGBMRegressor"].values()
    )


def test_mscp_bounds_are_calibrated_by_unique_id(mock_point_learner):
    """A volatile series must not widen another series' interval."""
    cdr = MultiStepConformalTimeSeriesRegressor(
        learner=mock_point_learner, horizon=2, n_windows=2, alpha=0.5
    )
    cdr.ncscores_ = {
        "LGBM": {
            "stable": np.array([[-1.0, -2.0], [1.0, 2.0]]),
            "volatile": np.array([[-10.0, -20.0], [10.0, 20.0]]),
        }
    }
    cdr.n = 2

    lower, upper = cdr._compute_bounds(
        y_hat=np.full(4, 100.0),
        model_name="LGBM",
        h=2,
        prediction_ids=np.array(["stable", "stable", "volatile", "volatile"]),
        alpha=0.5,
    )

    np.testing.assert_array_equal(lower, [99.0, 98.0, 90.0, 80.0])
    np.testing.assert_array_equal(upper, [101.0, 102.0, 110.0, 120.0])


def test_evaluate_inner_join_behavior(mock_point_learner, sample_distribution_data):
    """Ensure evaluate() properly inner joins predictions with test data across id and time columns."""
    cdr = MultiStepConformalTimeSeriesRegressor(
        learner=mock_point_learner, horizon=3, n_windows=2
    )
    cdr.fit(sample_distribution_data)

    # test_df with matching future keys but an additional unneeded column
    test_dates = pd.date_range("2024-01-26", periods=3, freq="D")
    test_df = pd.DataFrame(
        {
            "unique_id": ["id_1"] * 3 + ["id_2"] * 3,
            "ds": list(test_dates) * 2,
            "y": [20.0] * 6,
        }
    )
    test_df["extra_junk"] = 999

    eval_df = cdr.evaluate(df_test=test_df, h=3)
    assert not eval_df.empty
    assert "coverage_rate" in eval_df.columns
    assert "X_df" not in mock_point_learner.predict.call_args.kwargs


def test_predict_rejects_inconsistent_forecast_time_grids(
    mock_point_learner, sample_distribution_data
):
    cdr = MultiStepConformalTimeSeriesRegressor(
        learner=mock_point_learner, horizon=2, n_windows=2
    )
    cdr.fit(sample_distribution_data)
    mock_point_learner.predict.side_effect = None
    mock_point_learner.predict.return_value = pd.DataFrame(
        {
            "unique_id": ["id_1", "id_1", "id_2", "id_2"],
            "ds": pd.to_datetime(
                ["2024-01-26", "2024-01-27", "2024-01-26", "2024-01-28"]
            ),
            "LGBMRegressor": [20.0] * 4,
        }
    )

    with pytest.raises(ValueError, match="same horizon timestamps"):
        cdr.predict_interval(h=2)


def test_predict_rejects_model_not_seen_during_calibration(
    mock_point_learner, sample_distribution_data
):
    cdr = MultiStepConformalTimeSeriesRegressor(
        learner=mock_point_learner, horizon=2, n_windows=2
    ).fit(sample_distribution_data)
    mock_point_learner.predict.side_effect = None
    mock_point_learner.predict.return_value = pd.DataFrame(
        {
            "unique_id": ["id_1"] * 2 + ["id_2"] * 2,
            "ds": list(pd.date_range("2024-01-26", periods=2)) * 2,
            "OtherModel": [20.0] * 4,
        }
    )

    with pytest.raises(ValueError, match="was not present during calibration"):
        cdr.predict_interval(h=2)


def test_evaluate_rejects_duplicate_targets(
    mock_point_learner, sample_distribution_data
):
    cdr = MultiStepConformalTimeSeriesRegressor(
        learner=mock_point_learner, horizon=2, n_windows=2
    ).fit(sample_distribution_data)
    test_df = pd.DataFrame(
        {
            "unique_id": ["id_1", "id_1", "id_1", "id_2", "id_2"],
            "ds": pd.to_datetime(
                ["2024-01-26", "2024-01-26", "2024-01-27", "2024-01-26", "2024-01-27"]
            ),
            "y": [20.0] * 5,
        }
    )

    with pytest.raises(ValueError, match="at most one target"):
        cdr.evaluate(test_df, h=2)


def test_evaluate_requires_target_for_every_prediction(
    mock_point_learner, sample_distribution_data
):
    cdr = MultiStepConformalTimeSeriesRegressor(
        learner=mock_point_learner, horizon=2, n_windows=2
    ).fit(sample_distribution_data)
    test_df = pd.DataFrame(
        {
            "unique_id": ["id_1", "id_1", "id_2"],
            "ds": pd.to_datetime(["2024-01-26", "2024-01-27", "2024-01-26"]),
            "y": [20.0] * 3,
        }
    )

    with pytest.raises(ValueError, match="target for every prediction row"):
        cdr.evaluate(test_df, h=2)


def test_fit_separates_static_and_dynamic_features(
    mock_point_learner, sample_distribution_data
):
    df = sample_distribution_data.assign(
        region=lambda frame: frame["unique_id"].map({"id_1": "north", "id_2": "south"}),
        temperature=np.arange(len(sample_distribution_data)),
    )
    cdr = MultiStepConformalTimeSeriesRegressor(
        learner=mock_point_learner, horizon=2, n_windows=2
    ).fit(df, static_features=["region"], n_jobs=1)

    assert cdr.static_features_ == ["region"]
    assert cdr.exog_cols_ == ["temperature"]
    for call in mock_point_learner.fit.call_args_list:
        assert call.kwargs["static_features"] == ["region"]
    for call in mock_point_learner.predict.call_args_list[:-1]:
        assert "region" not in call.kwargs["X_df"].columns
        assert "temperature" in call.kwargs["X_df"].columns


def test_predict_validates_explicit_future_features(
    mock_point_learner, sample_distribution_data
):
    df = sample_distribution_data.assign(temperature=1.0)
    cdr = MultiStepConformalTimeSeriesRegressor(
        learner=mock_point_learner, horizon=2, n_windows=2
    ).fit(df)
    invalid_future = pd.DataFrame(
        {
            "unique_id": ["id_1"] * 2 + ["id_2"] * 2,
            "ds": list(pd.date_range("2024-01-26", periods=2)) * 2,
        }
    )

    with pytest.raises(ValueError, match="temperature"):
        cdr.predict_interval(h=2, X_df=invalid_future)


def test_mscp_preserves_fractional_coverage_in_column_names(
    mock_point_learner, sample_distribution_data
):
    cdr = MultiStepConformalTimeSeriesRegressor(
        learner=mock_point_learner, horizon=2, n_windows=2, alpha=0.055
    ).fit(sample_distribution_data)

    pred_df = cdr.predict_interval(h=2)

    assert "LGBMRegressor-lo-94.5" in pred_df
    assert "LGBMRegressor-hi-94.5" in pred_df

    test_dates = pd.date_range("2024-01-26", periods=2)
    test_df = pd.DataFrame(
        {
            "unique_id": ["id_1"] * 2 + ["id_2"] * 2,
            "ds": list(test_dates) * 2,
            "y": [20.0] * 4,
        }
    )
    eval_df = cdr.evaluate(test_df, h=2)
    assert eval_df.loc[0, "level"] == "94.5%"
    assert eval_df.loc[0, "alpha"] == pytest.approx(0.055)


def test_nexcp_weighted_refit_passes_internal_weight_column(
    mock_point_learner, sample_distribution_data
):
    MultiStepConformalTimeSeriesRegressor(
        learner=mock_point_learner,
        horizon=2,
        n_windows=2,
        nexcp=True,
        decay=0.5,
        weighted_refit=True,
    ).fit(sample_distribution_data, n_jobs=1)

    fit_kwargs = mock_point_learner.fit.call_args.kwargs
    assert fit_kwargs["weight_col"] == "_tinyconformal_weight"
    weights = fit_kwargs["df"].groupby("ds")["_tinyconformal_weight"].first()
    assert weights.is_monotonic_increasing
    assert weights.iloc[-1] > weights.iloc[0]


def test_nexcp_weighted_refit_requires_weight_col_support(
    sample_distribution_data,
):
    class LearnerWithoutWeights:
        def fit(self, df):
            return self

        def predict(self, h):
            raise AssertionError("predict should not be reached")

    cdr = MultiStepConformalTimeSeriesRegressor(
        learner=LearnerWithoutWeights(),
        horizon=2,
        n_windows=2,
        nexcp=True,
        weighted_refit=True,
    )
    with pytest.raises(TypeError, match="weight_col"):
        cdr.fit(sample_distribution_data, n_jobs=1)
