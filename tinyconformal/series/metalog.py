# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd


class ConformalMetalogNewsvendor:
    """
    Utility suite for Newsvendor inventory optimization using Metalog distributions
    fitted over pre-computed conformal prediction intervals.
    """

    @staticmethod
    def _validate_interval_pair(interval_pair: Tuple[str, str]) -> Tuple[str, str]:
        pattern = re.compile(r"^.+-(lo|hi)-\d+.*$")

        if (
            not isinstance(interval_pair, (tuple, list))
            or len(interval_pair) != 2
            or not isinstance(interval_pair[0], str)
            or not isinstance(interval_pair[1], str)
        ):
            raise ValueError(
                "interval_pair must be a 2-tuple of strings: (lower_col, upper_col)."
            )

        low, high = interval_pair
        if not pattern.match(low) or not pattern.match(high):
            raise ValueError(
                f"Interval columns ('{low}', '{high}') must follow the pattern '<model>-(lo|hi)-<level>'."
            )

        return str(low), str(high)

    @staticmethod
    def _compute_2term_coefficients(
        p_low_cqr: np.ndarray,
        p_high_cqr: np.ndarray,
        logit_level: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Metalog coefficients a1 (median) and a2 (scale) for 2-term symmetric fit."""
        a1 = 0.5 * (p_low_cqr + p_high_cqr)
        a2 = (p_high_cqr - p_low_cqr) / (2.0 * logit_level)
        return a1, a2

    @staticmethod
    def _compute_3term_coefficients(
        p_low_cqr: np.ndarray,
        p_high_cqr: np.ndarray,
        p_low_base: np.ndarray,
        p_high_base: np.ndarray,
        p50_base: np.ndarray,
        p_high_: float,
        logit_level: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Metalog coefficients a1, a2, and a3 (skewness) with CQR-adjusted median."""
        base_spread = np.maximum(p_high_base - p_low_base, 1e-6)
        skew_ratio = np.clip((p50_base - p_low_base) / base_spread, 0.01, 0.99)
        p50_adj = p_low_cqr + skew_ratio * (p_high_cqr - p_low_cqr)

        a1 = p50_adj
        a2 = (p_high_cqr - p_low_cqr) / (2.0 * logit_level)

        a3_denom = (p_high_ - 0.5) * logit_level
        a3 = (p_high_cqr + p_low_cqr - 2.0 * p50_adj) / a3_denom

        return a1, a2, a3

    @staticmethod
    def _eval_quantile_function(
        p_clipped: np.ndarray,
        logit_p: np.ndarray,
        a1: np.ndarray,
        a2: np.ndarray,
        a3: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Evaluate Metalog quantile function (SPT) for 2-term or 3-term formulation."""
        y_star = a1 + a2 * logit_p
        if a3 is not None:
            y_star += a3 * (p_clipped - 0.5) * logit_p
        return y_star

    @staticmethod
    def _eval_metalog_spt(
        p_star: np.ndarray,
        p_low_cqr: np.ndarray,
        p_high_cqr: np.ndarray,
        p_low_base: Optional[np.ndarray] = None,
        p_high_base: Optional[np.ndarray] = None,
        p50_base: Optional[np.ndarray] = None,
        p_high_: float = 0.95,
        logit_level: float = 2.9444389791664403,
    ) -> np.ndarray:
        p_clipped = np.clip(p_star, 1e-5, 1.0 - 1e-5)
        logit_p = np.log(p_clipped / (1.0 - p_clipped))

        has_3_terms = (
            p50_base is not None and p_low_base is not None and p_high_base is not None
        )

        if has_3_terms:
            a1, a2, a3 = ConformalMetalogNewsvendor._compute_3term_coefficients(
                p_low_cqr=p_low_cqr,
                p_high_cqr=p_high_cqr,
                p_low_base=p_low_base,
                p_high_base=p_high_base,
                p50_base=p50_base,
                p_high_=p_high_,
                logit_level=logit_level,
            )
        else:
            a1, a2 = ConformalMetalogNewsvendor._compute_2term_coefficients(
                p_low_cqr=p_low_cqr,
                p_high_cqr=p_high_cqr,
                logit_level=logit_level,
            )
            a3 = None

        return ConformalMetalogNewsvendor._eval_quantile_function(
            p_clipped=p_clipped,
            logit_p=logit_p,
            a1=a1,
            a2=a2,
            a3=a3,
        )

    @staticmethod
    def optimize_order_quantity(
        df: pd.DataFrame,
        interval_pair: Tuple[str, str],
        cu_col: str = "cu",
        co_col: str = "co",
        median_col: Optional[str] = None,
        level: float = 90.0,
        suffix: str = "",
    ) -> pd.DataFrame:
        """
        Compute expected cost-minimizing order quantity (y_optimal) on a DataFrame
        containing pre-computed conformal prediction intervals.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing forecast outputs along with cost columns.
        interval_pair : Tuple[str, str]
            Tuple with lower and upper quantile column names present in `df`
            (e.g., ("RF-lo-90", "RF-hi-90")).
        cu_col : str, default="cu"
            Column name in `df` containing the shortage cost per unit (cost of underage, cu).
        co_col : str, default="co"
            Column name in `df` containing the excess cost per unit (cost of overage, co).
        median_col : str, optional
            Column name for central predictions/median. If provided, enables 3-term
            Metalog reconstruction (skewness adjustment).
        level : float, default=90.0
            Nominal coverage level in percentage corresponding to the input interval bounds
            (e.g., 90.0 for P5 and P95 quantiles; 80.0 for P10 and P90 quantiles).
        suffix : str, default=""
            Optional suffix to append to the target decision column (e.g., suffix="_rf"
            results in `y_optimal_rf`).

        Returns
        -------
        pd.DataFrame
            Copy of input DataFrame augmented with decision outputs:

            - `p_star`: The critical service level computed as cu / (cu + co).
            - `y_optimal{suffix}`: The final decision variable representing the optimal
              inventory order quantity clipped at zero (max(0, y_star)).
        """
        if not (0.0 < level < 100.0):
            raise ValueError("Parameter 'level' must be strictly between 0 and 100.")

        low_col, high_col = ConformalMetalogNewsvendor._validate_interval_pair(
            interval_pair
        )

        if low_col not in df.columns or high_col not in df.columns:
            raise KeyError(
                f"Interval pair columns ('{low_col}', '{high_col}') not found in DataFrame."
            )

        if cu_col not in df.columns or co_col not in df.columns:
            raise KeyError(
                f"Cost columns ('{cu_col}', '{co_col}') not found in DataFrame."
            )

        p_low = (100.0 - level) / 200.0
        p_high = 1.0 - p_low
        logit_level = float(np.log(p_high / p_low))

        df_out = df.copy()

        c_u = df_out[cu_col].to_numpy(dtype=float)
        c_o = df_out[co_col].to_numpy(dtype=float)

        if np.any(c_u <= 0) or np.any(c_o <= 0):
            raise ValueError(
                "Unit costs 'cu' and 'co' in df must be strictly positive."
            )

        p_star = c_u / (c_u + c_o)
        df_out["p_star"] = p_star

        p_low_cqr = df_out[low_col].to_numpy(dtype=float)
        p_high_cqr = df_out[high_col].to_numpy(dtype=float)

        if median_col is not None:
            if median_col not in df_out.columns:
                raise KeyError(f"Median column '{median_col}' not found in DataFrame.")
            p_low_base = p_low_cqr
            p_high_base = p_high_cqr
            p50_base = df_out[median_col].to_numpy(dtype=float)
        else:
            p_low_base = p_high_base = p50_base = None

        y_star = ConformalMetalogNewsvendor._eval_metalog_spt(
            p_star=p_star,
            p_low_cqr=p_low_cqr,
            p_high_cqr=p_high_cqr,
            p_low_base=p_low_base,
            p_high_base=p_high_base,
            p50_base=p50_base,
            p_high_=p_high,
            logit_level=logit_level,
        )

        col_name = f"y_optimal{suffix}"
        df_out[col_name] = np.maximum(0.0, y_star)

        return df_out
