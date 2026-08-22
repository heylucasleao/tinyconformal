# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class ConformalMetalogNewsvendor(BaseEstimator):
    """
    Newsvendor Decision Maker via Metalog Distribution over a Fitted Conformal Regressor.

    Fits continuous probability curves (2-term or 3-term Metalog) using calibrated
    prediction bounds output by a conformal regressor. It calculates the optimal
    order quantity (y_optimal) corresponding to the critical cost ratio
    p_star = cu / (cu + co).

    Parameters
    ----------
    conformal_regressor : BaseEstimator
        An instance of a fitted conformal regressor capable of executing `predict_interval`.
    cu_col : str
        Column name in `X_df` containing the shortage cost per unit (cost of underage, cu).
    co_col : str
        Column name in `X_df` containing the excess cost per unit (cost of overage, co).
    level : float, default=90.0
        Nominal coverage level in percentage corresponding to the evaluated interval bounds
        (e.g., 90.0 for P5 and P95 quantiles; 80.0 for P10 and P90 quantiles).
    id_col : str, default="unique_id"
        Identifier column for time series or SKUs.

    Attributes
    ----------
    p_low_ : float
        Lower probability percentile derived from nominal coverage level (e.g., 0.05 for level=90.0).
    p_high_ : float
        Upper probability percentile derived from nominal coverage level (e.g., 0.95 for level=90.0).

    Examples
    --------
    >>> from tinyconformal.decision import ConformalMetalogNewsvendor
    >>> newsvendor = ConformalMetalogNewsvendor(
    ...     conformal_regressor=cqr_model,
    ...     cu_col="unit_margin",
    ...     co_col="holding_cost",
    ...     level=90.0
    ... )
    >>> # Evaluate optimal stock using a 90% interval and median
    >>> df_opt = newsvendor.predict_optimal_quantity(
    ...     X_df=df_test,
    ...     interval_pair=("RF-lo-90", "RF-hi-90"),
    ...     median_col="RF-median"
    ... )
    """

    def __init__(
        self,
        conformal_regressor: BaseEstimator,
        cu_col: str,
        co_col: str,
        level: float = 90.0,
        id_col: str = "unique_id",
    ):
        self.conformal_regressor = conformal_regressor
        self.cu_col = cu_col
        self.co_col = co_col
        self.level = level
        self.id_col = id_col

        if not (0.0 < self.level < 100.0):
            raise ValueError("Parameter 'level' must be strictly between 0 and 100.")

        self.p_low_ = (100.0 - self.level) / 200.0
        self.p_high_ = 1.0 - self.p_low_
        self._logit_level = float(np.log(self.p_high_ / self.p_low_))

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
                "interval_pair must be a tuple of two strings: (lower_col, upper_col)."
            )

        low, high = interval_pair
        if not pattern.match(low) or not pattern.match(high):
            raise ValueError(
                f"Interval columns ('{low}', '{high}') must follow the pattern '<model>-(lo|hi)-<level>'."
            )

        return str(low), str(high)

    def _eval_metalog_spt(
        self,
        p_star: np.ndarray,
        p_low_cqr: np.ndarray,
        p_high_cqr: np.ndarray,
        p_low_base: Optional[np.ndarray] = None,
        p_high_base: Optional[np.ndarray] = None,
        p50_base: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        p_clipped = np.clip(p_star, 1e-5, 1.0 - 1e-5)
        logit_p = np.log(p_clipped / (1.0 - p_clipped))

        has_3_terms = (
            p50_base is not None and p_low_base is not None and p_high_base is not None
        )

        if has_3_terms:
            base_spread = np.maximum(p_high_base - p_low_base, 1e-6)
            skew_ratio = np.clip((p50_base - p_low_base) / base_spread, 0.01, 0.99)
            p50_adj = p_low_cqr + skew_ratio * (p_high_cqr - p_low_cqr)

            a1 = p50_adj
            a2 = (p_high_cqr - p_low_cqr) / (2.0 * self._logit_level)

            a3_denom = (self.p_high_ - 0.5) * self._logit_level
            a3 = (p_high_cqr + p_low_cqr - 2.0 * p50_adj) / a3_denom

            y_star = a1 + a2 * logit_p + a3 * (p_clipped - 0.5) * logit_p
            return y_star, p50_adj
        else:
            a1 = 0.5 * (p_low_cqr + p_high_cqr)
            a2 = (p_high_cqr - p_low_cqr) / (2.0 * self._logit_level)

            y_star = a1 + a2 * logit_p
            return y_star, None

    def predict_optimal_quantity(
        self,
        X_df: pd.DataFrame,
        interval_pair: Tuple[str, str],
        median_col: Optional[str] = None,
        h: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Compute the expected cost-minimizing order quantity (y_optimal) for each observation.

        Parameters
        ----------
        X_df : pd.DataFrame
            Input features DataFrame containing unit cost columns (`cu_col` and `co_col`).
        interval_pair : Tuple[str, str]
            Tuple with lower and upper quantile column names generated by `predict_interval`
            (e.g., ("RF-lo-90", "RF-hi-90")).
        median_col : str, optional
            Column name of the central prediction/median. If provided, activates 3-term
            Metalog reconstruction with skewness preservation.
        h : int, optional
            Forecast horizon passed directly to `conformal_regressor.predict_interval`.
        alpha : float, optional
            Custom significance level (1 - coverage) overriding default settings.

        Returns
        -------
        pd.DataFrame
            DataFrame containing predicted conformal bounds alongside added decision columns:

            - `p_star`: The critical service level computed as cu / (cu + co).
            - `y_optimal`: The final decision variable representing the optimal inventory order
              quantity clipped at zero (max(0, y_star)).
            - `p50_conformal_adj` (Optional): The conformalized 50th percentile adjusted for
              baseline skewness. Only generated when 3-term Metalog is active (`median_col` provided).
        """
        low_col, high_col = self._validate_interval_pair(interval_pair)

        df_pred = self.conformal_regressor.predict_interval(h=h, alpha=alpha, X_df=X_df)

        c_u = X_df[self.cu_col].to_numpy(dtype=float)
        c_o = X_df[self.co_col].to_numpy(dtype=float)

        if np.any(c_u <= 0) or np.any(c_o <= 0):
            raise ValueError(
                "Unit costs 'cu' and 'co' in X_df must be strictly positive."
            )

        p_star = c_u / (c_u + c_o)
        df_pred["p_star"] = p_star

        p_low_cqr = df_pred[low_col].to_numpy(dtype=float)
        p_high_cqr = df_pred[high_col].to_numpy(dtype=float)

        if median_col is not None:
            if median_col not in df_pred.columns:
                raise KeyError(
                    f"Median column '{median_col}' not found in predictions."
                )
            p_low_base = p_low_cqr
            p_high_base = p_high_cqr
            p50_base = df_pred[median_col].to_numpy(dtype=float)
        else:
            p_low_base = p_high_base = p50_base = None

        y_star, p50_adj = self._eval_metalog_spt(
            p_star=p_star,
            p_low_cqr=p_low_cqr,
            p_high_cqr=p_high_cqr,
            p_low_base=p_low_base,
            p_high_base=p_high_base,
            p50_base=p50_base,
        )

        if p50_adj is not None:
            df_pred["p50_conformal_adj"] = p50_adj

        df_pred["y_optimal"] = np.maximum(0.0, y_star)

        return df_pred
