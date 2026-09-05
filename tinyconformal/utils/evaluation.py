# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

"""Operational diagnostics for first-stage conditional-mean forecasters."""

from __future__ import annotations

import numpy as np
import pandas as pd


class FirstStageEvaluator:
    r"""Operational diagnostics for a first-stage conditional-mean forecaster.

    Both the tabular and time-series CPS estimators calibrate a dispersion
    model on top of a first-stage location forecaster (``learner``). This
    evaluator checks that first-stage forecaster on its own, before any
    conformal scaling, using out-of-sample predictions supplied by the caller
    (e.g. ``sklearn.model_selection.cross_val_predict`` for tabular data or a
    Nixtla ``cross_validation`` backtest for panels).

    Notes on Metrics & Interpretation
    ---------------------------------
    - **WAPE**: Total absolute error divided by total observed demand. Lower is better.
    - **Score**: Composite operational loss defined as WAPE + |PBias|. Lower is better.
    - **Forecast Instability**: Relative change between consecutive forecasts of the same
      series. Only computed when ``id_col`` and ``time_col`` are both provided. Lower is better.
    - **PBias (Bias)**: Fractional global volume deviation ($\frac{\sum \hat{y} - \sum y}{\sum y}$).
      * *Interpretation*: Should be close to 0. A negative bias indicates overall under-forecasting (risk of stockouts),
        while a positive bias indicates over-forecasting (excess holding costs).
    - **False Demand on Zero-Days (Avg Pred)**: Average predicted value specifically on
      observations where the true target is strictly zero.
      * *Interpretation*: Measures the model's tendency to "smear" or leak intermittent demand into
        non-active periods, creating false expectations of activity.
    - **Peak Demand Deviation**: Fractional error of predicted values relative to true
      values restricted to periods of positive/peak demand.
      * *Interpretation*: Tracks the model's smoothing bias on positive-demand observations.
        Negative values indicate that the model under-forecasts realized peaks. Since this
        conditions on the observed target, it is an operational diagnostic rather than a
        direct test of conditional-mean calibration.
    """

    @classmethod
    def evaluate(
        cls,
        df_res: pd.DataFrame,
        target_col: str = "y",
        prediction_col: str = "y_pred",
        id_col: str | None = None,
        time_col: str | None = None,
    ) -> pd.DataFrame:
        """Evaluate the operational quality of out-of-sample mean forecasts.

        Parameters
        ----------
        df_res : pandas.DataFrame
            Out-of-sample predictions and observed targets.
        target_col : str, default="y"
            Column containing the observed target.
        prediction_col : str, default="y_pred"
            Column containing the first-stage conditional-mean prediction.
        id_col : str or None, default=None
            Column identifying each series. Required, together with
            ``time_col``, to compute Forecast Instability.
        time_col : str or None, default=None
            Column containing ordered timestamps. Required, together with
            ``id_col``, to compute Forecast Instability.

        Returns
        -------
        pandas.DataFrame
            Single-row table of operational metrics, matching tinyshift's
            ``FirstStageForecasterEvaluator`` column naming.

        Notes
        -----
        Input predictions should come from cross-validation, rolling-origin
        backtesting, or a held-out period. Evaluating in-sample fitted values
        gives optimistic results.
        """
        required = [target_col, prediction_col] + [
            column for column in (id_col, time_col) if column is not None
        ]
        missing = [column for column in required if column not in df_res.columns]
        if missing:
            raise KeyError(f"Columns not found in the input DataFrame: {missing}")

        valid = df_res[required].dropna().copy()
        if valid.empty:
            raise ValueError("No valid target/prediction pairs were found.")

        y_true = valid[target_col].to_numpy(dtype=float)
        y_pred = valid[prediction_col].to_numpy(dtype=float)
        cls._validate_mean_inputs(y_true, y_pred, prediction_col)

        total_true = np.sum(y_true)
        total_pred = np.sum(y_pred)
        total_abs_error = np.sum(np.abs(y_pred - y_true))
        if total_true > 0:
            wape = total_abs_error / total_true
            pbias = (total_pred - total_true) / total_true
        else:
            wape = 0.0 if total_abs_error == 0 else np.nan
            pbias = 0.0 if total_pred == 0 else np.nan

        zero_mask = y_true == 0
        pos_mask = y_true > 0

        false_alarm_zeros = np.mean(y_pred[zero_mask]) if np.sum(zero_mask) > 0 else 0.0
        peak_underestimation = (
            (np.mean(y_pred[pos_mask]) - np.mean(y_true[pos_mask]))
            / np.mean(y_true[pos_mask])
            if np.sum(pos_mask) > 0
            else 0.0
        )

        # Forecast Instability requires id_col/time_col; NaN keeps the column stable otherwise.
        forecast_instability = (
            cls._forecast_instability(
                valid, prediction_col=prediction_col, id_col=id_col, time_col=time_col
            )
            if id_col is not None and time_col is not None
            else np.nan
        )

        return pd.DataFrame(
            {
                "wape": [round(wape, 4)],
                "pbias": [round(pbias, 4)],
                "score": [round(wape + abs(pbias), 4)],
                "forecast_instability": [round(forecast_instability, 4)],
                "false_demand_on_zero_days_avg_pred": [round(false_alarm_zeros, 4)],
                "peak_demand_deviation": [round(peak_underestimation, 4)],
            }
        )

    @staticmethod
    def _validate_mean_inputs(
        y_true: np.ndarray, y_pred: np.ndarray, prediction_col: str
    ) -> None:
        if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
            raise ValueError("Target and prediction values must be finite.")
        if np.any(y_true < 0):
            raise ValueError("Target values must be non-negative.")
        if np.any(y_pred <= 0):
            raise ValueError(
                f"Conditional mean column '{prediction_col}' must be strictly positive."
            )

    @staticmethod
    def _forecast_instability(
        df_res: pd.DataFrame,
        prediction_col: str,
        id_col: str,
        time_col: str,
    ) -> float:
        ordered = df_res.sort_values([id_col, time_col])
        previous = ordered.groupby(id_col, observed=True)[prediction_col].shift(1)
        current = ordered[prediction_col]
        paired = previous.notna()
        if not paired.any():
            return np.nan

        prev_values = previous[paired].to_numpy(dtype=float)
        curr_values = current[paired].to_numpy(dtype=float)
        average_volume = 0.5 * (prev_values.sum() + curr_values.sum())
        if average_volume == 0:
            return 0.0
        revisions = prev_values - curr_values
        return float((np.abs(revisions).sum() + abs(revisions.sum())) / average_volume)

    @classmethod
    def calibration_table(
        cls,
        df_res: pd.DataFrame,
        target_col: str = "y",
        prediction_col: str = "y_pred",
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """Compare observed and predicted means across quantile-based bins.

        Parameters
        ----------
        df_res : pandas.DataFrame
            Out-of-sample predictions and observed targets.
        target_col : str, default="y"
            Column containing the observed target.
        prediction_col : str, default="y_pred"
            Column containing the first-stage conditional-mean prediction.
        n_bins : int, default=10
            Number of quantile-based bins built from ``prediction_col``.

        Returns
        -------
        pandas.DataFrame
            One row per bin with the observation count, mean prediction, mean
            observed target, and their difference (``mean_residual``),
            matching tinyshift's ``FirstStageForecasterEvaluator`` column naming.
        """
        if not isinstance(n_bins, int) or n_bins < 2:
            raise ValueError("n_bins must be an integer greater than or equal to 2.")
        missing = [c for c in (target_col, prediction_col) if c not in df_res.columns]
        if missing:
            raise KeyError(f"Columns not found in the input DataFrame: {missing}")

        valid = df_res[[target_col, prediction_col]].dropna().copy()
        if valid.empty:
            raise ValueError("No valid target/prediction pairs were found.")
        cls._validate_mean_inputs(
            valid[target_col].to_numpy(dtype=float),
            valid[prediction_col].to_numpy(dtype=float),
            prediction_col,
        )
        if valid[prediction_col].nunique() == 1:
            valid["calibration_bin"] = "all"
        else:
            valid["calibration_bin"] = pd.qcut(
                valid[prediction_col], q=n_bins, duplicates="drop"
            )
        result = (
            valid.groupby("calibration_bin", observed=True)
            .agg(
                count=(target_col, "size"),
                mean_prediction=(prediction_col, "mean"),
                mean_observed=(target_col, "mean"),
            )
            .reset_index()
        )
        result["mean_residual"] = result["mean_observed"] - result["mean_prediction"]
        return result
