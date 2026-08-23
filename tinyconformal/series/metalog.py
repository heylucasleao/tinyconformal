# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd


class ConformalNewsvendor:
    """
    Utility suite for Newsvendor inventory optimization using Metalog distributions
    fitted over pre-computed conformal prediction intervals.

    Notes
    -----
    The decision core relies on evaluating the continuous Metalog Quantile Function
    at the Critical Service Level (p_star):

    .. math::
        p_star = c_u / c_u + c_o

    where :math: p_star defines the optimal probability threshold of non-stockout
    given unit costs of underage `c_u` and overage `c_o`.
    """

    @staticmethod
    def _validate_interval_pair(interval_pair: Tuple[str, str]) -> Tuple[str, str]:
        """
        Validate that input interval_pair follows naming conventions and structure.

        Parameters
        ----------
        interval_pair : Tuple[str, str]
            Tuple containing column names for lower and upper conformal bounds.

        Returns
        -------
        Tuple[str, str]
            Validated tuple of string column names.
        """
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
        """
        Compute 2-term symmetric Metalog distribution coefficients (a1, a2).

        Objective
        ---------
        Fit a symmetric 2-term Metalog logistic-quantile curve passing strictly through
        the calibrated lower and upper conformal prediction bounds.

        Mathematical Equations
        ----------------------
        a1 (Median/Location) = 0.5 * (q_low + q_high)
        a2 (Scale/Spread)    = (q_high - q_low) / (2 * logit(p_high))

        Parameters
        ----------
        p_low_cqr : np.ndarray
            1D array of lower conformal quantile predictions (e.g., P5).
        p_high_cqr : np.ndarray
            1D array of upper conformal quantile predictions (e.g., P95).
        logit_level : float
            Pre-computed logit constant at nominal upper percentile:
            logit(p_high) = log(p_high / (1 - p_high)).

        Returns
        -------
        a1 : np.ndarray
            Metalog location parameter (central median under symmetry).
        a2 : np.ndarray
            Metalog scale parameter representing distribution spread.

        Examples
        --------
        >>> a1, a2 = ConformalNewsvendor._compute_2term_coefficients(
        ...     p_low_cqr=np.array([10.0]),
        ...     p_high_cqr=np.array([20.0]),
        ...     logit_level=2.9444
        ... )
        >>> # a1 = array([15.0]), a2 = array([1.6981])
        """
        cqr_spread = np.maximum(p_high_cqr - p_low_cqr, 1e-6)

        a1 = 0.5 * (p_low_cqr + p_high_cqr)
        a2 = cqr_spread / (2.0 * logit_level)
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
        """
        Compute 3-term asymmetric Metalog distribution coefficients (a1, a2, a3).

        Objective
        ---------
        Fit an asymmetric 3-term Metalog curve preserving underlying point-forecast
        skewness while scaling strictly to conformal prediction bounds.

        Mathematical Equations
        ----------------------
        skew_ratio = clip((p50_base - p_low_base) / max(p_high_base - p_low_base, 1e-6), 0.01, 0.99)
        p50_adj    = p_low_cqr + skew_ratio * (p_high_cqr - p_low_cqr)

        a1 (Adjusted Median) = p50_adj
        a2 (Scale/Spread)    = (p_high_cqr - p_low_cqr) / (2 * logit_level)
        a3 (Skewness Shape)  = (p_high_cqr + p_low_cqr - 2 * p50_adj) / ((p_high_ - 0.5) * logit_level)

        Parameters
        ----------
        p_low_cqr : np.ndarray
            1D array of lower conformal quantile predictions (CQR).
        p_high_cqr : np.ndarray
            1D array of upper conformal quantile predictions (CQR).
        p_low_base : np.ndarray
            1D array of base model lower bounds used to isolate baseline skewness ratio.
        p_high_base : np.ndarray
            1D array of base model upper bounds.
        p50_base : np.ndarray
            1D array of base central predictions/medians.
        p_high_ : float
            Nominal probability corresponding to upper bound (e.g., 0.95 for level=90.0).
        logit_level : float
            Pre-computed logit constant: log(p_high_ / (1 - p_high_)).

        Returns
        -------
        a1 : np.ndarray
            Adjusted central median parameter preserving baseline relative skew.
        a2 : np.ndarray
            Metalog scale/spread parameter.
        a3 : np.ndarray
            Asymmetry (skewness) coefficient controlling tail shape.

        Notes
        -----
        - Preserving base quantiles prevents the "asymmetry dilution effect" caused by
          symmetric CQR delta expansion (eta) pushing the implied skew_ratio towards 0.5.
        - The validity of the 3-term Metalog relies on Keelin (2016) monotonicity conditions:
          |a3 / a2| <= 1.6671. If violated due to severe asymmetry, callers should fall
          back to symmetric 2-term formulation (a3 = 0, a1 = (p_low_cqr + p_high_cqr) / 2)
          to avoid PDF invalidity or non-monotonic CDFs.

        Examples
        --------
        >>> import numpy as np
        >>> p_low_cqr = np.array([10.0])
        >>> p_high_cqr = np.array([30.0])
        >>> p_low_base = np.array([10.0])
        >>> p_high_base = np.array([30.0])
        >>> p50_base = np.array([12.0])
        >>> p_high_ = 0.95
        >>> logit_level = 2.9444389791664403
        >>> a1, a2, a3 = ConformalNewsvendor._compute_3term_coefficients(
        ...     p_low_cqr, p_high_cqr, p_low_base, p_high_base, p50_base, p_high_, logit_level
        ... )
        >>> a1, np.round(a2, 4), np.round(a3, 4)
        (array([12.]), array([3.3962]), array([4.5282]))
        """
        base_spread = np.maximum(
            p_high_base.astype(np.float64) - p_low_base.astype(np.float64), 1e-6
        )
        skew_ratio = np.clip(
            (p50_base.astype(np.float64) - p_low_base.astype(np.float64)) / base_spread,
            0.01,
            0.99,
        )

        cqr_spread = np.maximum(
            p_high_cqr.astype(np.float64) - p_low_cqr.astype(np.float64), 1e-6
        )
        p50_adj = p_low_cqr.astype(np.float64) + skew_ratio * cqr_spread

        a1_raw = p50_adj
        a2 = cqr_spread / (2.0 * logit_level)

        a3_denom = (p_high_ - 0.5) * logit_level
        a3_raw = (
            p_high_cqr.astype(np.float64) + p_low_cqr.astype(np.float64) - 2.0 * p50_adj
        ) / a3_denom

        # Keelin (2016) monotonicity check (|a3/a2| <= 1.6671)
        a3_ratio = np.abs(a3_raw) / np.maximum(a2, 1e-8)
        invalid_mask = a3_ratio > 1.6671

        a1_midpoint = 0.5 * (
            p_low_cqr.astype(np.float64) + p_high_cqr.astype(np.float64)
        )

        a1 = np.where(invalid_mask, a1_midpoint, a1_raw)
        a3 = np.where(invalid_mask, 0.0, a3_raw)

        return a1, a2, a3

    @staticmethod
    def _eval_quantile_function(
        p_clipped: np.ndarray,
        logit_p: np.ndarray,
        a1: np.ndarray,
        a2: np.ndarray,
        a3: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Evaluate Metalog Quantile Function M(y) for target critical service levels (p_star).

        Objective
        ---------
        Calculate target order quantity y_star = M(p_star) corresponding to critical ratio p_star.

        Mathematical Equations
        ----------------------
        M(p) = a1 + a2 * logit(p) + a3 * (p - 0.5) * logit(p)
        where logit(p) = log(p / (1 - p)).

        Parameters
        ----------
        p_clipped : np.ndarray
            1D array of critical service levels clipped to (1e-5, 1 - 1e-5).
        logit_p : np.ndarray
            1D array of logit values evaluated at clipped service levels: log(p / (1 - p)).
        a1 : np.ndarray
            Metalog location parameter array.
        a2 : np.ndarray
            Metalog scale parameter array.
        a3 : np.ndarray, optional
            Metalog skewness parameter array for 3-term formulation. Default is None.

        Returns
        -------
        y_star : np.ndarray
            Calculated target optimal quantities evaluated along continuous Metalog curve.

        Examples
        --------
        >>> y = ConformalNewsvendor._eval_quantile_function(
        ...     p_clipped=np.array([0.8]), logit_p=np.array([1.3862]),
        ...     a1=np.array([15.0]), a2=np.array([1.6981])
        ... )
        >>> # y = array([17.3539])
        """
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
        """
        Orchestrate Metalog distribution parameter fitting and quantile evaluation.

        Objective
        ---------
        Transform discrete conformal prediction intervals into continuous Metalog probability
        curves and evaluate optimal Newsvendor quantile.

        Parameters
        ----------
        p_star : np.ndarray
            Unclipped critical ratios cu / (cu + co).
        p_low_cqr : np.ndarray
            Lower conformal bound array.
        p_high_cqr : np.ndarray
            Upper conformal bound array.
        p_low_base : np.ndarray, optional
            Base model lower bound array.
        p_high_base : np.ndarray, optional
            Base model upper bound array.
        p50_base : np.ndarray, optional
            Base central prediction array.
        p_high_ : float, default=0.95
            Nominal upper probability bound percentile.
        logit_level : float, default=2.9444389791664403
            Pre-computed logit constant for given coverage level.

        Returns
        -------
        y_star : np.ndarray
            Target inventory decision array before non-negativity constraint.
        """
        p_clipped = np.clip(p_star, 1e-5, 1.0 - 1e-5)
        logit_p = np.log(p_clipped / (1.0 - p_clipped))

        has_3_terms = (
            p50_base is not None and p_low_base is not None and p_high_base is not None
        )

        if has_3_terms:
            a1, a2, a3 = ConformalNewsvendor._compute_3term_coefficients(
                p_low_cqr=p_low_cqr,
                p_high_cqr=p_high_cqr,
                p_low_base=p_low_base,
                p_high_base=p_high_base,
                p50_base=p50_base,
                p_high_=p_high_,
                logit_level=logit_level,
            )
        else:
            a1, a2 = ConformalNewsvendor._compute_2term_coefficients(
                p_low_cqr=p_low_cqr,
                p_high_cqr=p_high_cqr,
                logit_level=logit_level,
            )
            a3 = None

        return ConformalNewsvendor._eval_quantile_function(
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
        base_interval_pair: Optional[Tuple[str, str]] = None,
        level: float = 90.0,
        suffix: str = "",
    ) -> pd.DataFrame:
        """
        Compute expected cost-minimizing order quantity (y_optimal) on a DataFrame
        containing pre-computed conformal prediction intervals.

        Objective
        ---------
        Execute end-to-end Newsvendor decision optimization using continuous 2-term
        or 3-term Metalog distributions fitted over calibrated conformal bounds.

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
            Nominal coverage level in percentage corresponding to input bounds
            (e.g., 90.0 for P5 and P95 quantiles).
        suffix : str, default=""
            Optional suffix to append to output decision column (e.g., suffix="_rf"
            results in `y_optimal_rf`).

        Returns
        -------
        pd.DataFrame
            Copy of input DataFrame augmented with decision outputs:

            - `p_star`: The critical service level computed as cu / (cu + co).
            - `y_optimal{suffix}`: The final decision variable representing the optimal
              inventory order quantity clipped at zero (max(0, y_star)).

        Examples
        --------
        >>> df_input = pd.DataFrame({
        ...     "cu": [10.0, 15.0], "co": [2.0, 5.0],
        ...     "RF-lo-90": [100.0, 200.0], "RF-hi-90": [150.0, 280.0], "RF-median": [120.0, 230.0]
        ... })
        >>> df_result = ConformalNewsvendor.optimize_order_quantity(
        ...     df=df_input, interval_pair=("RF-lo-90", "RF-hi-90"), median_col="RF-median", suffix="_rf"
        ... )
        >>> # Returns DataFrame with columns 'p_star' and 'y_optimal_rf'
        """
        if not (0.0 < level < 100.0):
            raise ValueError("Parameter 'level' must be strictly between 0 and 100.")

        low_col, high_col = ConformalNewsvendor._validate_interval_pair(interval_pair)

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

            p50_base = df_out[median_col].to_numpy(dtype=float)

            if base_interval_pair is not None:
                b_low, b_high = base_interval_pair
                if b_low not in df_out.columns or b_high not in df.columns:
                    raise KeyError(
                        f"Base interval pair columns ('{b_low}', '{b_high}') not found in DataFrame."
                    )
                p_low_base = df_out[b_low].to_numpy(dtype=float)
                p_high_base = df_out[b_high].to_numpy(dtype=float)
            else:
                p_low_base = p_low_cqr
                p_high_base = p_high_cqr
        else:
            p_low_base = p_high_base = p50_base = None

        y_star = ConformalNewsvendor._eval_metalog_spt(
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
