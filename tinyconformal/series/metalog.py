# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class ConformalMetalogNewsvendor(BaseEstimator):
    """
    Newsvendor Decision Maker via Metalog Distribution over a Fitted Conformal Regressor.

    This class fits continuous probability curves (2-term or 3-term Metalog)
    using calibrated prediction bounds from a conformal regressor. It then computes
    the optimal order quantity (y_optimal) corresponding to the critical cost ratio
    p_star = cu / (cu + co).

    Parameters
    ----------
    conformal_regressor : BaseEstimator
        An instance of a fitted conformal regressor capable of calling `predict_interval`.
    interval_pairs : Tuple[str, str] or List[Tuple[str, str]]
        Tuple or list of tuples containing the column names for lower and upper quantiles.
        Must follow the pattern '(model)-(lo|hi)-(level)' (e.g., ("RF-lo-90", "RF-hi-90")).
    cu_col : str
        Column name in `X_df` containing the shortage cost per unit (cost of underage, cu).
    co_col : str
        Column name in `X_df` containing the excess cost per unit (cost of overage, co).
    median_cols : str, List[str], or None, default=None
        Column name or list of column names for the central prediction/median.
        If provided, enables 3-term Metalog reconstruction (skewness adjustment).
        Must match the length of `interval_pairs`.
    level : float, default=90.0
        Nominal coverage level in percentage corresponding to the input interval bounds
        (e.g., 90.0 for P5 and P95 quantiles; 80.0 for P10 and P90 quantiles).
    id_col : str, default="unique_id"
        Identifier column for time series / SKUs.

    Attributes
    ----------
    interval_pairs_ : List[Tuple[str, str]]
        Validated and normalized list of interval column pairs.
    median_cols_ : List[Optional[str]]
        Validated and normalized list of median column names.
    pair_mappings_ : Dict[int, Dict[str, Optional[str]]]
        Indexed mapping dictionary linking low, high, and median columns per pair.
    p_low_ : float
        Lower probability percentile derived from nominal coverage level (e.g., 0.05 for level=90.0).
    p_high_ : float
        Upper probability percentile derived from nominal coverage level (e.g., 0.95 for level=90.0).

    Notes
    -----
    Quantile parameterization follows the SPT (Symmetric Percentile Triplet) Metalog formulation:

    - 2-term Metalog (no median provided):
      Estimates location (a1) and scale (a2) directly from the conformal interval spread [q_lo, q_hi].

    - 3-term Metalog (median provided):
      Preserves the baseline skewness ratio (p50_base - p_low_base) / (p_high_base - p_low_base)
      to project an adjusted conformal median (p50_conformal_adj), estimating a shape parameter (a3).

    Examples
    --------
    >>> from tinyconformal.decision import ConformalMetalogNewsvendor
    >>> newsvendor = ConformalMetalogNewsvendor(
    ...     conformal_regressor=cqr_model,
    ...     interval_pairs=("RF-lo-90", "RF-hi-90"),
    ...     cu_col="unit_margin",
    ...     co_col="holding_cost",
    ...     median_cols="RF-median",
    ...     level=90.0
    ... )
    >>> df_optimal = newsvendor.predict_optimal_quantity(X_df=df_test)
    """

    def __init__(
        self,
        conformal_regressor: BaseEstimator,
        interval_pairs: Union[Tuple[str, str], List[Tuple[str, str]]],
        cu_col: str,
        co_col: str,
        median_cols: Union[str, List[str], None] = None,
        level: float = 90.0,
        id_col: str = "unique_id",
    ):
        self.conformal_regressor = conformal_regressor
        self.cu_col = cu_col
        self.co_col = co_col
        self.level = level
        self.id_col = id_col
        self.interval_pairs = interval_pairs
        self.median_cols = median_cols

        self.interval_pairs_ = self._validate_interval_pairs(interval_pairs)
        self.median_cols_ = self._validate_median_cols(
            median_cols, num_pairs=len(self.interval_pairs_)
        )
        self.pair_mappings_ = self._build_pair_mappings(
            self.interval_pairs_, self.median_cols_
        )

        if not (0.0 < self.level < 100.0):
            raise ValueError("Parameter 'level' must be strictly between 0 and 100.")

        self.p_low_ = (100.0 - self.level) / 200.0
        self.p_high_ = 1.0 - self.p_low_
        self._logit_level = float(np.log(self.p_high_ / self.p_low_))

    def _validate_interval_pairs(
        self,
        interval_pairs: Union[Tuple[str, str], List[Tuple[str, str]]],
    ) -> List[Tuple[str, str]]:
        pattern = re.compile(r"^.+-(lo|hi)-\d+.*$")

        if (
            isinstance(interval_pairs, tuple)
            and len(interval_pairs) == 2
            and isinstance(interval_pairs[0], str)
            and isinstance(interval_pairs[1], str)
        ):
            pairs = [interval_pairs]
        elif isinstance(interval_pairs, list) and all(
            isinstance(pair, (tuple, list))
            and len(pair) == 2
            and isinstance(pair[0], str)
            and isinstance(pair[1], str)
            for pair in interval_pairs
        ):
            pairs = [tuple(pair) for pair in interval_pairs]
        else:
            raise ValueError(
                "interval_pairs must be a 2-tuple of strings (low, high) or a list of such tuples."
            )

        for low, high in pairs:
            if not pattern.match(low) or not pattern.match(high):
                raise ValueError(
                    f"Interval columns ('{low}', '{high}') must follow the pattern '<model>-(lo|hi)-<level>'."
                )

        return pairs

    def _validate_median_cols(
        self, median_cols: Union[str, List[str], None], num_pairs: int
    ) -> List[Optional[str]]:
        if isinstance(median_cols, str):
            normalized_medians = [median_cols]
        elif isinstance(median_cols, list) and all(
            isinstance(col, str) for col in median_cols
        ):
            normalized_medians = median_cols
        elif median_cols is None:
            normalized_medians = [None] * num_pairs
        else:
            raise TypeError("median_cols must be a string, list of strings, or None.")

        if len(normalized_medians) != num_pairs:
            raise ValueError(
                f"Number of median_cols ({len(normalized_medians)}) must match "
                f"the number of interval_pairs ({num_pairs})."
            )

        return normalized_medians

    @staticmethod
    def _build_pair_mappings(
        interval_pairs: List[Tuple[str, str]], median_cols: List[Optional[str]]
    ) -> Dict[int, Dict[str, Optional[str]]]:
        return {
            idx: {
                "low": q_pair[0],
                "high": q_pair[1],
                "median": med,
            }
            for idx, (q_pair, med) in enumerate(zip(interval_pairs, median_cols))
        }

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
        h: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Compute the expected cost-minimizing order quantity (y_optimal) for each observation.

        Parameters
        ----------
        X_df : pd.DataFrame
            Input features DataFrame containing unit cost columns (`cu_col` and `co_col`).
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
              quantity clipped at zero (max(0, y_star)). If multiple interval pairs are provided,
              columns are suffixed with `_0`, `_1`, etc.
            - `p50_conformal_adj` (Optional): The conformalized 50th percentile adjusted for
              baseline skewness. Only generated when 3-term Metalog is active (`median_cols` provided).
        """
        df_pred = self.conformal_regressor.predict_interval(h=h, alpha=alpha, X_df=X_df)

        c_u = X_df[self.cu_col].to_numpy(dtype=float)
        c_o = X_df[self.co_col].to_numpy(dtype=float)

        if np.any(c_u <= 0) or np.any(c_o <= 0):
            raise ValueError(
                "Unit costs 'cu' and 'co' in X_df must be strictly positive."
            )

        p_star = c_u / (c_u + c_o)
        df_pred["p_star"] = p_star

        for idx, mapping in self.pair_mappings_.items():
            low_col = mapping["low"]
            high_col = mapping["high"]
            med_col = mapping["median"]

            p_low_cqr = df_pred[low_col].to_numpy(dtype=float)
            p_high_cqr = df_pred[high_col].to_numpy(dtype=float)

            if med_col is not None:
                p_low_base = p_low_cqr
                p_high_base = p_high_cqr
                p50_base = df_pred[med_col].to_numpy(dtype=float)
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

            suffix = f"_{idx}" if len(self.pair_mappings_) > 1 else ""

            if p50_adj is not None:
                df_pred[f"p50_conformal_adj{suffix}"] = p50_adj

            df_pred[f"y_optimal{suffix}"] = np.maximum(0.0, y_star)

        return df_pred
