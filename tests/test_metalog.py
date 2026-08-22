# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

import numpy as np
import pandas as pd
import pytest

from tinyconformal.series.metalog import ConformalNewsvendor


@pytest.fixture
def sample_dataframe():
    """Fixture providing a standard test DataFrame with conformal prediction bounds."""
    return pd.DataFrame(
        {
            "cu": [10.0, 15.0, 5.0],
            "co": [2.0, 5.0, 15.0],
            "RF-lo-90": [100.0, 200.0, 50.0],
            "RF-hi-90": [150.0, 280.0, 100.0],
            "RF-median": [120.0, 230.0, 70.0],
        }
    )


class TestConformalNewsvendor:
    """Test suite for ConformalNewsvendor optimization utilities."""

    def test_p_star_computation(self, sample_dataframe):
        """Verify that p_star = cu / (cu + co) is calculated correctly."""
        df_res = ConformalNewsvendor.optimize_order_quantity(
            df=sample_dataframe,
            interval_pair=("RF-lo-90", "RF-hi-90"),
        )

        expected_p_star = np.array([10 / 12, 15 / 20, 5 / 20])  # ~0.8333, 0.75, 0.25
        np.testing.assert_allclose(
            df_res["p_star"].to_numpy(), expected_p_star, rtol=1e-5
        )

    def test_2term_optimization_output(self, sample_dataframe):
        """Test symmetric 2-term Metalog optimization without median_col."""
        df_res = ConformalNewsvendor.optimize_order_quantity(
            df=sample_dataframe,
            interval_pair=("RF-lo-90", "RF-hi-90"),
            suffix="_2term",
        )

        assert "y_optimal_2term" in df_res.columns
        assert len(df_res) == len(sample_dataframe)
        assert (df_res["y_optimal_2term"] >= 0).all()

        # Row 1: cu=10, co=2 -> p_star=0.8333 (> 0.5) -> y_optimal should be above midpoint (125.0)
        assert df_res.loc[0, "y_optimal_2term"] > 125.0
        # Row 3: cu=5, co=15 -> p_star=0.25 (< 0.5) -> y_optimal should be below midpoint (75.0)
        assert df_res.loc[2, "y_optimal_2term"] < 75.0

    def test_3term_optimization_output(self, sample_dataframe):
        """Test asymmetric 3-term Metalog optimization with median_col."""
        df_res = ConformalNewsvendor.optimize_order_quantity(
            df=sample_dataframe,
            interval_pair=("RF-lo-90", "RF-hi-90"),
            median_col="RF-median",
            suffix="_3term",
        )

        assert "y_optimal_3term" in df_res.columns
        assert len(df_res) == len(sample_dataframe)
        assert (df_res["y_optimal_3term"] >= 0).all()

    def test_non_negativity_clipping(self):
        """Verify that negative demand projections are properly clipped at zero."""
        df_negative = pd.DataFrame(
            {
                "cu": [1.0],
                "co": [99.0],  # Very low p_star (0.01)
                "RF-lo-90": [-50.0],
                "RF-hi-90": [-10.0],
            }
        )

        df_res = ConformalNewsvendor.optimize_order_quantity(
            df=df_negative,
            interval_pair=("RF-lo-90", "RF-hi-90"),
        )

        assert df_res.loc[0, "y_optimal"] == 0.0

    def test_invalid_interval_pair_format(self, sample_dataframe):
        """Ensure exception is raised when interval columns don't match pattern."""
        with pytest.raises(ValueError, match="must follow the pattern"):
            ConformalNewsvendor.optimize_order_quantity(
                df=sample_dataframe,
                interval_pair=("invalid_lower", "invalid_upper"),
            )

    def test_missing_columns_raises_key_error(self, sample_dataframe):
        """Ensure KeyError is raised when specified columns are missing from DataFrame."""
        with pytest.raises(KeyError, match="not found in DataFrame"):
            ConformalNewsvendor.optimize_order_quantity(
                df=sample_dataframe,
                interval_pair=("RF-lo-90", "RF-hi-90"),
                cu_col="non_existing_cu",
            )

    @pytest.mark.parametrize("invalid_level", [0.0, 100.0, -10.0, 105.0])
    def test_invalid_level_raises_value_error(self, sample_dataframe, invalid_level):
        """Ensure ValueError is raised for levels outside (0, 100)."""
        with pytest.raises(ValueError, match="must be strictly between 0 and 100"):
            ConformalNewsvendor.optimize_order_quantity(
                df=sample_dataframe,
                interval_pair=("RF-lo-90", "RF-hi-90"),
                level=invalid_level,
            )

    def test_non_positive_costs_raise_value_error(self, sample_dataframe):
        """Ensure ValueError is raised when unit costs cu or co are <= 0."""
        df_bad_cost = sample_dataframe.copy()
        df_bad_cost.loc[0, "cu"] = 0.0

        with pytest.raises(ValueError, match="must be strictly positive"):
            ConformalNewsvendor.optimize_order_quantity(
                df=df_bad_cost,
                interval_pair=("RF-lo-90", "RF-hi-90"),
            )
