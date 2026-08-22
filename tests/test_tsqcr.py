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
    """Mock estimator returning a single interval pair ('LGBM-lo-90', 'LGBM-hi-90')."""
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
    """Mock estimator returning multiple interval pairs."""
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


# --- Column Normalization & Renaming Tests ---


def test_normalize_interval_cols_single_tuple(mock_quantile_learner_single):
    """Verify single tuple input converts to a list of tuples using interval_cols."""
    cqr = ConformalQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_single,
        horizon=5,
        interval_cols=("LGBM-lo-90", "LGBM-hi-90"),
    )
    assert cqr.interval_pairs_ == [("LGBM-lo-90", "LGBM-hi-90")]


def test_normalize_interval_cols_list_of_tuples(mock_quantile_learner_multi):
    """Verify normalization when given a list of tuples/lists via interval_cols."""
    pairs = [("LGBM-lo-90", "LGBM-hi-90"), ["LGBM-lo-50", "LGBM-hi-50"]]
    cqr = ConformalQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_multi, horizon=5, interval_cols=pairs
    )
    assert cqr.interval_pairs_ == [
        ("LGBM-lo-90", "LGBM-hi-90"),
        ("LGBM-lo-50", "LGBM-hi-50"),
    ]


@pytest.mark.parametrize(
    "invalid_cols",
    ["invalid_string", ("only_one_col",), [("low", "high", "extra")], 12345],
)
def test_normalize_interval_cols_invalid_raises_error(
    mock_quantile_learner_single, invalid_cols
):
    """Ensure ValueError is raised when invalid interval_cols formats are passed."""
    with pytest.raises(
        ValueError, match="interval_cols must be a tuple of 2 column names"
    ):
        ConformalQuantileTimeSeriesRegressor(
            learner=mock_quantile_learner_single, horizon=5, interval_cols=invalid_cols
        )


def test_invalid_interval_col_pattern_raises_error(mock_quantile_learner_single):
    """Ensure ValueError is raised when column names don't match <model>-(lo|hi)-<level>."""
    with pytest.raises(ValueError, match="Invalid lower quantile column name"):
        ConformalQuantileTimeSeriesRegressor(
            learner=mock_quantile_learner_single,
            horizon=5,
            interval_cols=("invalid_lo_format", "LGBM-hi-90"),
        )


# --- Nonconformity Scores and Residual Calculations ---


def test_generate_residuals(mock_quantile_learner_single):
    """Test correctness of CQR nonconformity score computation: max(q_low - y, y - q_high)."""
    cqr = ConformalQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_single,
        horizon=3,
        interval_cols=("LGBM-lo-90", "LGBM-hi-90"),
    )
    q_low = np.array([10.0, 10.0, 10.0])
    q_high = np.array([20.0, 20.0, 20.0])

    y_true = np.array([15.0, 25.0, 5.0])

    residuals = cqr._generate_residuals(q_low, q_high, y_true)
    np.testing.assert_array_equal(residuals, np.array([-5.0, 5.0, 5.0]))


def test_sample_correction(mock_quantile_learner_single):
    """Test finite-sample quantile adjustment computation."""
    cqr = ConformalQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_single,
        horizon=5,
        interval_cols=("LGBM-lo-90", "LGBM-hi-90"),
    )
    cqr.n = 100
    q_level = cqr._sample_correction(alpha=0.05)
    assert pytest.approx(q_level, abs=1e-4) == 0.96


# --- Validation and Backtesting Tests ---


def test_sequential_backtesting_insufficient_time_steps(mock_quantile_learner_single):
    """Raise ValueError if time series lacks sufficient time steps for backtesting."""
    cqr = ConformalQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_single,
        horizon=10,
        n_windows=5,
        interval_cols=("LGBM-lo-90", "LGBM-hi-90"),
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
    cqr = ConformalQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_single,
        horizon=3,
        n_windows=2,
        interval_cols=("LGBM-lo-90", "LGBM-hi-90"),
    )
    # Return df missing expected columns
    mock_quantile_learner_single.predict.side_effect = (
        lambda h, X_df=None: pd.DataFrame(
            {
                "unique_id": ["series_1"] * h,
                "ds": pd.date_range("2024-01-31", periods=h),
            }
        )
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
        interval_cols=("LGBM-lo-90", "LGBM-hi-90"),
    )
    cqr.fit(sample_time_series_data)

    pair_key = "LGBM-lo-90:LGBM-hi-90"
    assert pair_key in cqr.ncscores_
    assert cqr.ncscores_[pair_key].shape == (4, 3)
    assert cqr.n == 4


def test_predict_interval_single_pair_formatting(
    mock_quantile_learner_single, sample_time_series_data
):
    """Validate output column formatting with -cqr suffix for a single interval pair."""
    cqr = ConformalQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_single,
        horizon=3,
        n_windows=2,
        alpha=0.10,
        interval_cols=("LGBM-lo-90", "LGBM-hi-90"),
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
    cqr = ConformalQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_multi, horizon=3, n_windows=2, interval_cols=pairs
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
    cqr = ConformalQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_multi, horizon=3, n_windows=2, interval_cols=pairs
    )
    cqr.fit(sample_time_series_data)

    test_dates = pd.date_range("2024-01-31", periods=3, freq="D")
    df_test = pd.DataFrame(
        {
            "unique_id": ["series_1"] * 3 + ["series_2"] * 3,
            "ds": list(test_dates) * 2,
            "y": [15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
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


def test_evaluate_metric_values_correctness(mock_quantile_learner_single):
    """Test exact mathematical outputs of evaluate() on deterministic bounds."""
    cqr = ConformalQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_single,
        horizon=2,
        interval_cols=("LGBM-lo-90", "LGBM-hi-90"),
    )
    # Mock do predict_interval para retornar limites fixos e controlados
    cqr.predict_interval = MagicMock(
        return_value=pd.DataFrame(
            {
                "unique_id": ["s1", "s1"],
                "ds": pd.date_range("2024-01-01", periods=2),
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

    eval_df = cqr.evaluate(df_test=df_test, h=2, alpha=0.10)
    row = eval_df.iloc[0]

    # Coverage: 1 de 2 cobertos -> 0.5
    assert row["coverage_rate"] == 0.5
    # Width: (20 - 10) = 10.0
    assert row["interval_width_mean"] == 10.0
    # MWIS:
    # Obs 1: 10 + 0 + 0 = 10
    # Obs 2: 10 + (2/0.1)*(25 - 20) = 10 + 20*5 = 110
    # Mean MWIS: (10 + 110) / 2 = 60.0
    assert row["mwis"] == 60.0


def test_tscqr_predict_raw(mock_quantile_learner_single, sample_time_series_data):
    """Verify _predict_raw returns raw predictions in 2D numpy array format for TSCQR."""
    cqr = ConformalQuantileTimeSeriesRegressor(
        learner=mock_quantile_learner_single,
        horizon=3,
        n_windows=2,
        interval_cols=("LGBM-lo-90", "LGBM-hi-90"),
    )
    cqr.fit(sample_time_series_data)
    preds_raw = cqr._predict_raw(h=3)

    assert isinstance(preds_raw, np.ndarray)
    assert preds_raw.shape == (2, 3)
