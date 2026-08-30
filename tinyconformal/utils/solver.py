# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

import re
from typing import Any

import numpy as np
import pandas as pd

CostInput = str | float | int | dict[str | tuple[str, Any], float]

INTERVAL_COLUMN_PATTERN = re.compile(
    r"^(?P<model>.+)-(?P<bound>lo|hi)-(?P<level>\d+)(?P<suffix>-cqr)?$"
)


def _parse_interval_pair(interval_pair: tuple[str, str]) -> tuple[str, str, int]:
    """Validate a named central interval pair and infer its coverage level."""
    if not isinstance(interval_pair, tuple) or len(interval_pair) != 2:
        raise ValueError("interval_pair must be a tuple of two column names.")
    lo_col, hi_col = interval_pair
    if not isinstance(lo_col, str) or not isinstance(hi_col, str):
        raise TypeError("interval_pair must contain two string column names.")

    lo_match = INTERVAL_COLUMN_PATTERN.fullmatch(lo_col)
    hi_match = INTERVAL_COLUMN_PATTERN.fullmatch(hi_col)
    if lo_match is None or lo_match.group("bound") != "lo":
        raise ValueError(
            f"Invalid lower interval column '{lo_col}'. Expected "
            "'<model>-lo-<level>' with an optional '-cqr' suffix."
        )
    if hi_match is None or hi_match.group("bound") != "hi":
        raise ValueError(
            f"Invalid upper interval column '{hi_col}'. Expected "
            "'<model>-hi-<level>' with an optional '-cqr' suffix."
        )

    for field, label in (
        ("model", "model"),
        ("level", "coverage level"),
        ("suffix", "suffix"),
    ):
        if lo_match.group(field) != hi_match.group(field):
            raise ValueError(
                f"Interval {label} mismatch between '{lo_col}' and '{hi_col}'."
            )

    level = int(lo_match.group("level"))
    if not 0 < level < 100:
        raise ValueError("The inferred interval level must be between 1 and 99.")
    return lo_col, hi_col, level


def _compute_critical_quantile(cu: np.ndarray, co: np.ndarray) -> np.ndarray:
    """Computes the critical quantile q_star = c_u / (c_u + c_o) in-place safely."""
    denom = cu + co
    out = np.empty_like(denom)
    zero_mask = denom == 0
    out[zero_mask] = 0.5
    np.divide(cu, denom, out=out, where=~zero_mask)
    return out


def _extract_cost_array(
    df: pd.DataFrame,
    cost_input: CostInput,
    id_col: str,
    time_col: str,
    n_rows: int,
) -> np.ndarray:
    """Extracts cost structures into flat NumPy arrays with optimized dict lookup."""
    if isinstance(cost_input, (int, float, np.integer, np.floating)):
        return np.full(n_rows, float(cost_input), dtype=float)
    elif isinstance(cost_input, str):
        return df[cost_input].to_numpy(dtype=float)
    elif isinstance(cost_input, dict):
        if not cost_input:
            raise ValueError("Cost dictionary cannot be empty.")

        first_key = next(iter(cost_input.keys()))
        if isinstance(first_key, tuple):
            tuples = zip(df[id_col].to_numpy(), df[time_col].to_numpy())
            return np.fromiter(
                (cost_input.get(k, np.nan) for k in tuples),
                dtype=float,
                count=n_rows,
            )
        else:
            return df[id_col].map(cost_input).to_numpy(dtype=float)
    else:
        raise TypeError(
            f"Cost input must be a column name (str), numeric scalar (float/int), "
            f"or dict mapping IDs or (ID, Time) tuples to values. Received: {type(cost_input)}"
        )


def _enforce_monotonicity(q_lo: np.ndarray, q_hi: np.ndarray) -> None:
    """Enforce non-negativity and ordered interval bounds in-place.

    Applies sequential clipping using `np.maximum(..., out=...)` to prevent memory
    re-allocation while correcting crossed interval bounds.

    Args:
        q_lo (np.ndarray): Lower prediction interval bound array. Modified in-place.
        q_hi (np.ndarray): Upper prediction interval bound array. Modified in-place.
    """
    # 1. Non-negativity baseline (physical inventory floor)
    np.maximum(0.0, q_lo, out=q_lo)
    np.maximum(0.0, q_hi, out=q_hi)

    # 2. Quantile ordering (q_lo <= q_hi)
    np.maximum(q_lo, q_hi, out=q_hi)


def _interpolate_linear(
    q_star: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    p_lo: float,
    p_hi: float,
) -> np.ndarray:
    """Linearly interpolate a critical ratio between two interval bounds."""
    t = np.clip((q_star - p_lo) / (p_hi - p_lo), 0.0, 1.0)
    return q_lo + t * (q_hi - q_lo)


class NewsvendorSolver:
    """Vectorized Newsvendor Inventory Model Solver for Panel Data.

    This class solves the classic inventory/capacity decision problem under
    uncertainty (Newsvendor Problem) directly applied to probabilistic forecast
    DataFrames in standard Nixtla ecosystem formats (e.g., StatsForecast,
    NeuralForecast).

    General Workflow:
        1. **Critical Quantile Calculation (q_star):** Computes the optimal service
           level q_star = c_u / (c_u + c_o) for each time series and period.
        2. **Probabilistic Mapping:** Treats conformal interval bounds as two
           approximate cumulative-probability anchors.
        3. **Monotonicity Enforcement:** Enforces physical non-negativity and resolves
           quantile crossings in-place to ensure valid empirical CDF curves.
        4. **Optimal Point Interpolation (y_star):** Estimates the target inventory
           quantity using memory-optimized 2-point linear interpolation.
        5. **Adjustment and Physical Bounding:** Clips the final decision within
           bounds derived from the prediction interval.

    Global Attention Points:
        - **Nixtla Ecosystem Compatibility:** Designed for high-throughput panel
          DataFrames. Preserves sorting and index layout with zero-copy views where possible.
        - **Performance Optimization:** Uses vectorized interpolation and `np.fromiter`
          tuple mapping to eliminate Python loops and avoid redundant RAM allocation.
        - **CDF Monotonicity:** Crossed quantile errors from forecasting models
          (e.g., q_lo > q_hi) are resolved via in-place `_enforce_monotonicity`.

    Examples:
        >>> import pandas as pd
        >>> df_forecast = pd.DataFrame({
        ...     "unique_id": ["A", "A"],
        ...     "ds": ["2026-01-01", "2026-01-02"],
        ...     "model-lo-90-cqr": [10.0, 15.0],
        ...     "model-hi-90-cqr": [50.0, 60.0]
        ... })
        >>> solver = NewsvendorSolver()
        >>> res = solver.optimize(
        ...     df=df_forecast,
        ...     interval_pair=("model-lo-90-cqr", "model-hi-90-cqr"),
        ...     underage_cost=10.0,
        ...     overage_cost=2.0
        ... )
    """

    @staticmethod
    def optimize(
        df: pd.DataFrame,
        interval_pair: tuple[str, str],
        underage_cost: CostInput,
        overage_cost: CostInput,
        id_col: str = "unique_id",
        time_col: str = "ds",
        ratio_col: str = "critical_ratio",
        output_col: str = "y_optimal",
        assume_sorted: bool = True,
    ) -> pd.DataFrame:
        """Executes inventory optimization based on underage and overage costs.

        Calculates the optimal order quantity y_star by aligning the critical ratio
        q_star with the empirical demand distribution represented by the forecast
        quantiles in the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame containing probabilistic forecasts.
            interval_pair (Tuple[str, str]): Column name tuple `(lower_col, upper_col)`
                representing central prediction interval bounds. Names must follow
                `<model>-lo-<level>` and `<model>-hi-<level>`, optionally ending in
                `-cqr`. The coverage level is inferred from the names.
            underage_cost (Union[str, float, Dict]): Underage/shortage cost (c_u). Can be:
                - Scalar (`float`/`int`): Constant cost across the entire panel.
                - `str`: Column name in `df` containing variable costs per row.
                - `dict`: Cost mapping by ID `{id: cost}` or tuple `{(id, ds): cost}`.
            overage_cost (Union[str, float, Dict]): Overage/holding cost (c_o). Accepts the
                same input formats as `underage_cost`.
            id_col (str, optional): Identifier column for unique time series. Defaults to `"unique_id"`.
            time_col (str, optional): Timestamp/date column. Defaults to `"ds"`.
            output_col (str, optional): Name of the output column for optimized quantities
                in the returned DataFrame. Defaults to `"y_optimal"`.
            ratio_col : str, default="critical_ratio"
                Column name to store the computed critical ratio/fractile values.
            assume_sorted (bool, optional): If `True`, assumes the input DataFrame is already
                sorted by `[id_col, time_col]` (standard in Nixtla workflows), skipping
                redundant sorting operations to maximize performance. Defaults to `True`.

        Returns:
            pd.DataFrame: A copy of the DataFrame with calculated optimal inventory values in `output_col`.

        Raises:
            ValueError: If interval names are invalid or inconsistent, or if cost
                dictionaries are empty.
            TypeError: If the cost input type for `underage_cost` or `overage_cost` is unsupported.

        Intentional Clips and Truncations:
            1. **Physical Non-negativity (`q_lo >= 0`, `q_hi >= 0`):**
               Applies `np.maximum(0.0, ...)` via `_enforce_monotonicity` to ensure negative
               forecast predictions do not leak into physical inventory decisions.
            2. **Monotonicity Enforcement (`q_lo <= q_hi`):**
               Corrects inverted interval boundaries in-place with
               `q_hi = max(q_lo, q_hi)`.
            3. **Linear Boundary Clipping (`y_final` in range `[q_lo, q_hi]`):**
               Applies `np.clip(y_final, q_lo, q_hi)` to ensure that even for extreme critical
               quantiles (`q_star < p_lo` or `q_star > p_hi`), the final order quantity remains
               bounded within the prediction interval bounds.

        Attention Points:
            - **Zero-Division Safety:** If `underage_cost + overage_cost == 0` for any row, the critical
              quantile defaults to `0.5` (median) to prevent zero-division runtime errors.
            - **Approximation, not distributional calibration:** Conformal interval coverage
              does not imply that each endpoint is a calibrated CDF quantile. The linear
              interpolation is a bounded decision heuristic, not a conformal guarantee for
              the critical fractile.
            - **Out-of-Bounds Quantiles (`q_star` outside `[p_lo, p_hi]`):** Critical quantiles
              falling outside the prediction interval are capped at `p_lo` or `p_hi`
              (returning `q_lo` or `q_hi`), maintaining conservative inventory decisions.
            - **Missing Keys in Cost Dicts:** Unmatched IDs or timestamps in cost dictionaries
              will resolve to `NaN` in the optimal output column.
        """
        lo_col, hi_col, level = _parse_interval_pair(interval_pair)
        missing = [column for column in (lo_col, hi_col) if column not in df.columns]
        if missing:
            raise ValueError(
                f"Interval columns are missing from the DataFrame: {missing}"
            )

        if not assume_sorted and id_col in df.columns and time_col in df.columns:
            df_res = df.sort_values([id_col, time_col]).copy()
        else:
            df_res = df.copy()

        n_rows = len(df_res)
        p_lo = (100.0 - level) / 200.0
        p_hi = 1.0 - p_lo

        cu_arr = _extract_cost_array(df_res, underage_cost, id_col, time_col, n_rows)
        co_arr = _extract_cost_array(df_res, overage_cost, id_col, time_col, n_rows)

        for name, costs in (
            ("underage_cost", cu_arr),
            ("overage_cost", co_arr),
        ):
            invalid = (~np.isnan(costs)) & ((costs < 0) | ~np.isfinite(costs))
            if np.any(invalid):
                raise ValueError(
                    f"'{name}' must contain only non-negative finite values."
                )

        q_star = _compute_critical_quantile(cu=cu_arr, co=co_arr)

        q_lo = df_res[lo_col].to_numpy(dtype=float, copy=False)
        q_hi = df_res[hi_col].to_numpy(dtype=float, copy=False)
        _enforce_monotonicity(q_lo=q_lo, q_hi=q_hi)

        y_final = _interpolate_linear(
            q_star=q_star,
            q_lo=q_lo,
            q_hi=q_hi,
            p_lo=p_lo,
            p_hi=p_hi,
        )
        df_res[ratio_col] = q_star
        df_res[output_col] = np.clip(y_final, q_lo, q_hi)
        return df_res

    @staticmethod
    def optimize_distribution(
        df: pd.DataFrame,
        distribution,
        underage_cost: CostInput,
        overage_cost: CostInput,
        id_col: str = "unique_id",
        time_col: str = "ds",
        ratio_col: str = "critical_ratio",
        output_col: str = "y_optimal",
        nonnegative: bool = True,
    ) -> pd.DataFrame:
        """Optimize inventory directly from a predictive distribution's inverse CDF.

        ``distribution`` must implement ``len(distribution)`` and a row-wise
        ``ppf(quantiles)`` method, as objects returned by
        ``SplitConformalPredictiveSystem.predict_distribution`` do. Unlike
        :meth:`optimize`, this method evaluates the exact requested critical
        fractile instead of interpolating between two interval endpoints.
        """
        # Row order is significant: each distribution belongs to the DataFrame
        # row at the same position.
        df_res = df.copy()

        n_rows = len(df_res)
        try:
            n_distributions = len(distribution)
        except (TypeError, AttributeError) as exc:
            raise TypeError(
                "distribution must implement len() and a row-wise ppf() method."
            ) from exc
        if n_distributions != n_rows:
            raise ValueError(
                "The predictive distribution batch and DataFrame must have the same "
                f"number of rows; got {n_distributions} and {n_rows}."
            )
        if not callable(getattr(distribution, "ppf", None)):
            raise TypeError("distribution must implement a callable ppf() method.")

        cu_arr = _extract_cost_array(df_res, underage_cost, id_col, time_col, n_rows)
        co_arr = _extract_cost_array(df_res, overage_cost, id_col, time_col, n_rows)
        for name, costs in (("underage_cost", cu_arr), ("overage_cost", co_arr)):
            invalid = (~np.isnan(costs)) & ((costs < 0) | ~np.isfinite(costs))
            if np.any(invalid):
                raise ValueError(
                    f"'{name}' must contain only non-negative finite values."
                )

        q_star = _compute_critical_quantile(cu=cu_arr, co=co_arr)
        y_final = np.asarray(distribution.ppf(q_star), dtype=float)
        if y_final.shape != (n_rows,):
            raise ValueError(
                "distribution.ppf(row_wise_quantiles) must return one value per row."
            )
        if nonnegative:
            y_final = np.maximum(y_final, 0.0)

        df_res[ratio_col] = q_star
        df_res[output_col] = y_final
        return df_res
