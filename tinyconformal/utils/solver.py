# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from tinyconformal.distribution.base import DiscretePredictiveDistribution

CostInput = str | float | int | dict[str | tuple[str, Any], float]

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


def _validate_distribution_batch(distribution, n_rows: int, method: str) -> None:
    """Validate batch alignment and a required predictive-distribution method."""
    try:
        n_distributions = len(distribution)
    except (TypeError, AttributeError) as exc:
        raise TypeError(
            f"distribution must implement len() and a callable {method}() method."
        ) from exc

    if n_distributions != n_rows:
        raise ValueError(
            "The predictive distribution batch and DataFrame must have the same "
            f"number of rows; got {n_distributions} and {n_rows}."
        )
    if not callable(getattr(distribution, method, None)):
        raise TypeError(f"distribution must implement a callable {method}() method.")


def _resolve_distribution_inputs(df, distribution):
    """Resolve tabular CPS inputs or a self-contained time-series forecast."""
    if distribution is not None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        return df, distribution
    to_frame = getattr(df, "to_frame", None)
    underlying = getattr(df, "distribution", None)
    frame = to_frame() if callable(to_frame) else None
    if not isinstance(frame, pd.DataFrame) or underlying is None:
        raise TypeError(
            "Provide both a DataFrame and distribution, or a time-series "
            "predictive forecast returned by predict_distribution()."
        )
    return frame, underlying


def _resolve_unit_grid(max_k: int | None, units: Iterable[int] | None) -> np.ndarray:
    """Validate the unit selection and return its grid."""
    if max_k is not None and units is not None:
        raise ValueError("Provide either max_k or units, not both.")
    if units is None:
        max_k = 10 if max_k is None else max_k
        if (
            isinstance(max_k, (bool, np.bool_))
            or not isinstance(max_k, (int, np.integer))
            or max_k < 0
        ):
            raise ValueError("max_k must be a non-negative integer.")
        return np.arange(0, int(max_k) + 1)

    if isinstance(units, (str, bytes)):
        raise ValueError("units must be a non-empty iterable of integers.")
    try:
        unit_values = list(units)
    except TypeError as exc:
        raise ValueError("units must be a non-empty iterable of integers.") from exc
    if not unit_values or any(
        isinstance(unit, (bool, np.bool_))
        or not isinstance(unit, (int, np.integer))
        or unit < 0
        for unit in unit_values
    ):
        raise ValueError("units must be a non-empty iterable of non-negative integers.")
    if len(set(unit_values)) != len(unit_values):
        raise ValueError("units must not contain duplicates.")
    return np.asarray(unit_values, dtype=int)


def _validate_column_template(column_template: str) -> None:
    """Validate the output-column template used for inventory units."""
    if not isinstance(column_template, str) or "{k}" not in column_template:
        raise ValueError("column_template must be a string containing '{k}'.")


class NewsvendorSolver:
    """Apply Newsvendor decisions to row-aligned predictive distributions."""

    @staticmethod
    def optimize(
        df,
        distribution=None,
        underage_cost: CostInput = None,
        overage_cost: CostInput = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        ratio_col: str = "critical_ratio",
        output_col: str = "y_optimal",
    ) -> pd.DataFrame:
        """Optimize inventory directly from a predictive distribution's inverse CDF.

        For tabular CPS, provide the row-aligned ``df`` and ``distribution``.
        For TSCPS, pass its self-contained panel forecast as ``df`` and omit
        ``distribution``. The decision is the predictive quantile at the exact
        critical fractile; interval endpoints are not interpolated.
        """
        df, distribution = _resolve_distribution_inputs(df, distribution)
        df_res = df.copy()

        n_rows = len(df_res)
        _validate_distribution_batch(distribution, n_rows, method="ppf")

        cu_arr = _extract_cost_array(df_res, underage_cost, id_col, time_col, n_rows)
        co_arr = _extract_cost_array(df_res, overage_cost, id_col, time_col, n_rows)
        for name, costs in (("underage_cost", cu_arr), ("overage_cost", co_arr)):
            invalid = (~np.isnan(costs)) & ((costs < 0) | ~np.isfinite(costs))
            if np.any(invalid):
                raise ValueError(
                    f"'{name}' must contain only non-negative finite values."
                )

        q_star = _compute_critical_quantile(cu=cu_arr, co=co_arr)
        y_final = np.asarray(distribution.ppf(q_star[:, None]), dtype=float)
        if y_final.shape != (n_rows,):
            raise ValueError(
                "distribution.ppf(row_wise_quantiles) must return one value per row."
            )
        df_res[ratio_col] = q_star
        df_res[output_col] = y_final
        return df_res

    @staticmethod
    def pmf_distribution(
        df,
        distribution: DiscretePredictiveDistribution | None = None,
        max_k: int | None = None,
        units: Iterable[int] | None = None,
        column_template: str = "P(Y={k})",
    ) -> pd.DataFrame:
        """Evaluate exact probability masses for selected inventory units.

        Provide either ``max_k`` for the dense grid ``0, ..., max_k`` or
        ``units`` for an explicit sparse/stepped grid. If neither is provided,
        ``max_k`` defaults to 10. A TSCPS forecast can be passed directly with
        ``distribution=None``.
        """
        df, distribution = _resolve_distribution_inputs(df, distribution)
        if not isinstance(distribution, DiscretePredictiveDistribution):
            raise TypeError(
                "pmf is available only for discrete predictive distributions."
            )

        df_res = df.copy()
        n_rows = len(df_res)
        _validate_distribution_batch(distribution, n_rows, method="pmf")
        unit_grid = _resolve_unit_grid(max_k=max_k, units=units)
        _validate_column_template(column_template)

        pmf_matrix = np.asarray(distribution.pmf(unit_grid), dtype=float)
        expected_shape = (n_rows, unit_grid.size)
        if pmf_matrix.shape != expected_shape:
            raise ValueError(
                "distribution.pmf(unit_grid) must return a matrix with shape "
                f"{expected_shape}; got {pmf_matrix.shape}."
            )

        for index, unit in enumerate(unit_grid):
            df_res[column_template.format(k=int(unit))] = pmf_matrix[:, index]

        # Summing selected masses gives the upper-tail probability only for the
        # complete dense support prefix selected through max_k.
        if units is None:
            df_res[f"P(Y>{int(unit_grid[-1])})"] = 1.0 - pmf_matrix.sum(axis=1)
        return df_res

    @staticmethod
    def marginal_benefit_distribution(
        df,
        distribution: DiscretePredictiveDistribution | None = None,
        underage_cost: CostInput = None,
        overage_cost: CostInput = None,
        max_k: int | None = None,
        units: Iterable[int] | None = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        column_template: str = "MB(k={k})",
    ) -> pd.DataFrame:
        """Evaluate the marginal benefit of discrete inventory units.

        For every forecast row and candidate unit ``k``, this method computes

        ``MB(k) = c_u * P(Y >= k) - c_o * P(Y < k)``

        directly from the predictive CDF as

        ``MB(k) = c_u * (1 - F(k - 1)) - c_o * F(k - 1)``.

        Parameters
        ----------
        df : pandas.DataFrame
            Row-aligned forecast or decision frame.  Cost columns, when used,
            are read from this frame.
        distribution : DiscretePredictiveDistribution
            Batch of discrete predictive distributions aligned positionally
            with ``df``.
        underage_cost : str, float, int, or dict
            Unit shortage cost.  It may be a scalar, a column name, a mapping
            from series IDs to costs, or a mapping from ``(ID, time)`` pairs.
        overage_cost : str, float, int, or dict
            Unit excess cost, with the same accepted forms as
            ``underage_cost``.
        max_k : int, optional
            Largest non-negative unit evaluated when ``units`` is not supplied.
            Defaults to 10 and cannot be combined with ``units``.
        units : iterable of int, optional
            Explicit non-empty grid of unique, non-negative inventory units.
            Cannot be combined with ``max_k``.
        id_col : str, default="unique_id"
            Series identifier column used by dictionary cost inputs.
        time_col : str, default="ds"
            Time column used by dictionary cost inputs.
        column_template : str, default="MB(k={k})"
            Format string used for output columns.  It must contain ``{k}``.

        Returns
        -------
        pandas.DataFrame
            A copy of ``df`` with one marginal-benefit column per requested
            inventory unit.

        Notes
        -----
        Positive values favor adding unit ``k``; negative values indicate that
        its expected overage cost exceeds its expected shortage benefit.  The
        calculation uses the CDF directly rather than accumulating a truncated
        PMF, so it remains valid for discrete supports whose minimum is not zero.

        Row order is significant: distribution row ``i`` must describe frame
        row ``i``.  This method is intentionally unavailable for continuous
        predictive distributions, where an individual inventory unit does not
        define a probability-mass increment.
        """
        df, distribution = _resolve_distribution_inputs(df, distribution)
        if not isinstance(distribution, DiscretePredictiveDistribution):
            raise TypeError(
                "marginal benefit is available only for discrete predictive "
                "distributions."
            )

        df_res = df.copy()
        n_rows = len(df_res)
        _validate_distribution_batch(distribution, n_rows, method="cdf")
        unit_grid = _resolve_unit_grid(max_k=max_k, units=units)
        _validate_column_template(column_template)

        cu_arr = _extract_cost_array(df_res, underage_cost, id_col, time_col, n_rows)
        co_arr = _extract_cost_array(df_res, overage_cost, id_col, time_col, n_rows)
        for name, costs in (("underage_cost", cu_arr), ("overage_cost", co_arr)):
            invalid = (~np.isnan(costs)) & ((costs < 0) | ~np.isfinite(costs))
            if np.any(invalid):
                raise ValueError(
                    f"'{name}' must contain only non-negative finite values."
                )

        # A 2D matrix avoids ambiguity when the unit grid happens to have the
        # same length as the distribution batch (which otherwise means row-wise
        # input in the PredictiveDistribution protocol).
        thresholds = np.broadcast_to(unit_grid - 1, (n_rows, unit_grid.size))
        probability_less = np.asarray(distribution.cdf(thresholds), dtype=float)
        expected_shape = (n_rows, unit_grid.size)
        if probability_less.shape != expected_shape:
            raise ValueError(
                "distribution.cdf(unit_grid) must return a matrix with shape "
                f"{expected_shape}; got {probability_less.shape}."
            )

        marginal_benefit = (
            cu_arr[:, None] * (1.0 - probability_less)
            - co_arr[:, None] * probability_less
        )
        for index, unit in enumerate(unit_grid):
            df_res[column_template.format(k=int(unit))] = marginal_benefit[:, index]
        return df_res
