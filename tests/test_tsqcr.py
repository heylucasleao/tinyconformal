import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from tinyconformal.series import ConformalQuantileTimeSeriesRegressor


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
    """Mock estimator returning a single quantile pair ('LGBM-lo-90', 'LGBM-hi-90')."""
    learner = MagicMock()
    learner.fit.return_value = learner

    def mock_predict(h, X_df=None):
        dates = pd.date_range("2024-01-31", periods=h, freq="D")
        records = []
        for uid in ["series_1", "series_2"]:
            for d in dates:
                records.append(
                    {"unique_id": uid, "ds": d, "LGBM-lo-90": 10.0, "LGBM-hi-90": 20.0}
                )
        return pd.DataFrame(records)

    learner.predict.side_effect = mock_predict
    return learner


@pytest.fixture
def mock_quantile_learner_multi():
    """Mock estimator returning multiple quantile pairs."""
    learner = MagicMock()
    learner.fit.return_value = learner

    def mock_predict(h, X_df=None):
        dates = pd.date_range("2024-01-31", periods=h, freq="D")
        records = []
        for uid in ["series_1", "series_2"]:
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


# --- Column Normalization Tests ---


def test_normalize_quantile_cols_single_tuple(mock_quantile_learner_single):
    """Verify single tuple input converts to a list of tuples."""
    cqr = ConformalQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_single,
        horizon=5,
        quantile_cols=("LGBM-lo-90", "LGBM-hi-90"),
    )
    assert cqr.quantile_pairs_ == [("LGBM-lo-90", "LGBM-hi-90")]


def test_normalize_quantile_cols_list_of_tuples(mock_quantile_learner_multi):
    """Verify normalization when given a list of tuples/lists."""
    pairs = [("LGBM-lo-90", "LGBM-hi-90"), ["LGBM-lo-50", "LGBM-hi-50"]]
    cqr = ConformalQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_multi, horizon=5, quantile_cols=pairs
    )
    assert cqr.quantile_pairs_ == [
        ("LGBM-lo-90", "LGBM-hi-90"),
        ("LGBM-lo-50", "LGBM-hi-50"),
    ]


@pytest.mark.parametrize(
    "invalid_cols",
    ["invalid_string", ("only_one_col",), [("low", "high", "extra")], 12345],
)
def test_normalize_quantile_cols_invalid_raises_error(
    mock_quantile_learner_single, invalid_cols
):
    """Ensure ValueError is raised when invalid quantile_cols formats are passed."""
    with pytest.raises(
        ValueError, match="quantile_cols must be a tuple of 2 column names"
    ):
        ConformalQuantileTimeSeriesRegressor(
            learner=mock_quantile_learner_single, horizon=5, quantile_cols=invalid_cols
        )


# --- Nonconformity Scores and Residual Calculations ---


def test_generate_residuals(mock_quantile_learner_single):
    """Test correctness of CQR nonconformity score computation: max(q_low - y, y - q_high)."""
    cqr = ConformalQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_single,
        horizon=3,
        quantile_cols=("LGBM-lo-90", "LGBM-hi-90"),
    )
    q_low = np.array([10.0, 10.0, 10.0])
    q_high = np.array([20.0, 20.0, 20.0])

    # Obs 1: inside interval (15) -> max(-5, -5) = -5
    # Obs 2: above upper bound (25) -> max(-15, 5) = 5
    # Obs 3: below lower bound (5) -> max(5, -15) = 5
    y_true = np.array([15.0, 25.0, 5.0])

    residuals = cqr._generate_residuals(q_low, q_high, y_true)
    np.testing.assert_array_equal(residuals, np.array([-5.0, 5.0, 5.0]))


def test_sample_correction(mock_quantile_learner_single):
    """Test finite-sample quantile adjustment computation."""
    cqr = ConformalQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_single,
        horizon=5,
        quantile_cols=("LGBM-lo-90", "LGBM-hi-90"),
    )
    cqr.n = 100
    # ceil((101) * 0.95) / 100 = ceil(95.95) / 100 = 96 / 100 = 0.96
    q_level = cqr._sample_correction(alpha=0.05)
    assert pytest.approx(q_level, abs=1e-4) == 0.96


# --- Validation and Backtesting Tests ---


def test_sequential_backtesting_insufficient_time_steps(mock_quantile_learner_single):
    """Raise ValueError if time series lacks sufficient time steps for backtesting."""
    cqr = ConformalQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_single,
        horizon=10,
        n_windows=5,
        quantile_cols=("LGBM-lo-90", "LGBM-hi-90"),
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
    """Ensure KeyError is raised when configured quantile columns are missing from predictions."""
    cqr = ConformalQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_single,
        horizon=3,
        n_windows=2,
        quantile_cols=("NON_EXISTENT_LO", "NON_EXISTENT_HI"),
    )
    with pytest.raises(KeyError, match="were not found in forecast output"):
        cqr.fit(sample_time_series_data)


# --- Fitting and Interval Prediction Tests ---


def test_fit_and_ncscores_structure(
    mock_quantile_learner_single, sample_time_series_data
):
    """Verify that fitting populates ncscores_ correctly and updates calibration sample size n."""
    cqr = ConformalQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_single,
        horizon=3,
        n_windows=2,
        quantile_cols=("LGBM-lo-90", "LGBM-hi-90"),
    )
    cqr.fit(sample_time_series_data)

    pair_key = "LGBM-lo-90:LGBM-hi-90"
    assert pair_key in cqr.ncscores_
    # 2 windows * 2 series = 4 calibration trajectories
    assert cqr.ncscores_[pair_key].shape == (4, 3)
    assert cqr.n == 4


def test_predict_interval_single_pair_formatting(
    mock_quantile_learner_single, sample_time_series_data
):
    """Validate output column formatting for a single quantile pair."""
    cqr = ConformalQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_single,
        horizon=3,
        n_windows=2,
        alpha=0.10,
        quantile_cols=("LGBM-lo-90", "LGBM-hi-90"),
    )
    cqr.fit(sample_time_series_data)
    pred_df = cqr.predict_interval(h=3)

    assert "LGBM-lo-90" in pred_df.columns
    assert "LGBM-hi-90" in pred_df.columns


def test_predict_interval_multi_pair_formatting(
    mock_quantile_learner_multi, sample_time_series_data
):
    """Validate output column formatting with -CQR suffix for multiple quantile pairs."""
    pairs = [("LGBM-lo-90", "LGBM-hi-90"), ("LGBM-lo-50", "LGBM-hi-50")]
    cqr = ConformalQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_multi, horizon=3, n_windows=2, quantile_cols=pairs
    )
    cqr.fit(sample_time_series_data)
    pred_df = cqr.predict_interval(h=3)

    assert "LGBM-lo-90-CQR" in pred_df.columns
    assert "LGBM-hi-90-CQR" in pred_df.columns
    assert "LGBM-lo-50-CQR" in pred_df.columns
    assert "LGBM-hi-50-CQR" in pred_df.columns


def test_tscqr_compute_bounds_direct(mock_quantile_learner_single):
    """Directly unit test _compute_bounds logic using mocked ncscores_."""
    cqr = ConformalQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_single,
        horizon=2,
        quantile_cols=("LGBM-lo-90", "LGBM-hi-90"),
    )
    pair_key = "LGBM-lo-90:LGBM-hi-90"
    # Pre-populate nonconformity scores: 2 windows, horizon 2
    cqr.ncscores_ = {pair_key: np.array([[1.0, 2.0], [3.0, 4.0]])}
    cqr.n = 2

    q_low = np.array([10.0, 10.0, 10.0, 10.0])
    q_high = np.array([20.0, 20.0, 20.0, 20.0])

    # For n=2 and alpha=0.05, _sample_correction gives q_level=1.0 (max)
    # qhat across axis=0 with method="higher" for q_level=1.0 yields max per step: [3.0, 4.0]
    # lower_bound = q_low - 3.0 = 7.0 (step 0), 10.0 - 4.0 = 6.0 (step 1)
    lower, upper = cqr._compute_bounds(
        q_low=q_low, q_high=q_high, pair_key=pair_key, h=2, n_series=2, alpha=0.05
    )

    expected_lower = np.array([7.0, 6.0, 7.0, 6.0])
    expected_upper = np.array([23.0, 24.0, 23.0, 24.0])

    np.testing.assert_array_equal(lower, expected_lower)
    np.testing.assert_array_equal(upper, expected_upper)


def test_tscqr_sequential_backtesting_custom_step_size_and_static_features(
    mock_quantile_learner_single, sample_time_series_data
):
    """Verify _sequential_backtesting executes with explicit step_size and static_features."""
    cqr = ConformalQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_single,
        horizon=2,
        n_windows=2,
        quantile_cols=("LGBM-lo-90", "LGBM-hi-90"),
    )
    cqr.exog_cols_ = ["exog_feat"]

    residuals = cqr._sequential_backtesting(
        df=sample_time_series_data, step_size=3, static_features=["static_col"]
    )

    pair_key = "LGBM-lo-90:LGBM-hi-90"
    assert pair_key in residuals
    assert len(residuals[pair_key]) == 2


# --- Additional Edge Case Tests for TSCQR ---


def test_tscqr_evaluate_multi_quantile_cqr_suffix(
    mock_quantile_learner_multi, sample_time_series_data
):
    """Test evaluate() method on TSCQR with multi-pair quantiles parsing the -CQR column suffix."""
    pairs = [("LGBM-lo-90", "LGBM-hi-90"), ("LGBM-lo-50", "LGBM-hi-50")]
    cqr = ConformalQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_multi, horizon=3, n_windows=2, quantile_cols=pairs
    )
    cqr.fit(sample_time_series_data)
    test_df = sample_time_series_data.tail(6).copy()
    eval_df = cqr.evaluate(df_test=test_df, h=3)

    assert not eval_df.empty
    assert any("-CQR" in model for model in eval_df["model"])


def test_tscqr_predict_raw(mock_quantile_learner_single, sample_time_series_data):
    """Verify _predict_raw returns raw predictions in 2D numpy array format for TSCQR."""
    cqr = ConformalQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_single,
        horizon=3,
        n_windows=2,
        quantile_cols=("LGBM-lo-90", "LGBM-hi-90"),
    )
    cqr.fit(sample_time_series_data)
    preds_raw = cqr._predict_raw(h=3)

    assert isinstance(preds_raw, np.ndarray)
    assert preds_raw.shape == (2, 3)  # 2 series, horizon 3


# --- Parameterization & Custom Columns Tests for TSCQR ---


@pytest.mark.parametrize(
    "alpha_val, expected_level",
    [
        (0.01, "99"),
        (0.05, "95"),
        (0.10, "90"),
        (0.20, "80"),
    ],
)
def test_tscqr_alpha_override_predict_interval(
    mock_quantile_learner_single, sample_time_series_data, alpha_val, expected_level
):
    """Verify that overriding alpha in predict_interval correctly updates column names and levels."""
    cqr = ConformalQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_single,
        horizon=3,
        n_windows=2,
        alpha=0.05,
        quantile_cols=("LGBM-lo-90", "LGBM-hi-90"),
    )
    cqr.fit(sample_time_series_data)
    pred_df = cqr.predict_interval(h=3, alpha=alpha_val)

    assert f"LGBM-lo-{expected_level}" in pred_df.columns
    assert f"LGBM-hi-{expected_level}" in pred_df.columns


def test_tscqr_custom_column_names(mock_quantile_learner_single):
    """Verify TSCQR pipeline with custom structural column names (id_col, time_col, target_col)."""
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    custom_df = pd.DataFrame(
        {
            "item_id": ["A"] * 20 + ["B"] * 20,
            "timestamp": list(dates) * 2,
            "value": np.random.randn(40),
        }
    )

    learner = MagicMock()
    learner.fit.return_value = learner
    learner.predict.return_value = pd.DataFrame(
        {
            "item_id": ["A"] * 2 + ["B"] * 2,
            "timestamp": list(pd.date_range("2024-01-21", periods=2)) * 2,
            "LGBM-lo-90": [1.0] * 4,
            "LGBM-hi-90": [5.0] * 4,
        }
    )

    cqr = ConformalQuantileTimeSeriesRegressor(
        learner=learner,
        horizon=2,
        n_windows=2,
        quantile_cols=("LGBM-lo-90", "LGBM-hi-90"),
        id_col="item_id",
        time_col="timestamp",
        target_col="value",
    )

    cqr.fit(custom_df)
    pred_df = cqr.predict_interval(h=2)

    assert cqr.id_col == "item_id"
    assert "item_id" in pred_df.columns
    assert "timestamp" in pred_df.columns
