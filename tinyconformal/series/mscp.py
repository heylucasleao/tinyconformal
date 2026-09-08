# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from tinyconformal.core import conformal as core_conformal
from tinyconformal.core.quantiles import (
    central_conformal_quantile_levels,
)
from tinyconformal.utils.imports import requires_extra

from .residual import ResidualConformalTimeSeriesRegressor


class MultiStepConformalTimeSeriesRegressor(ResidualConformalTimeSeriesRegressor):
    """
    Multi-Step Conformalized Quantile Regressor for Time Series.

    Applies conformal prediction over multi-step horizons for Nixtla-style
    estimators (MLForecast or StatsForecast) using sequential backtesting
    to build empirical nonconformity scores (signed residuals). Calibration
    quantiles are computed independently for every series identifier and
    horizon; signed residuals are not pooled across series.
    """

    def __init__(
        self,
        learner: BaseEstimator,
        alpha: float = 0.05,
    ):
        """Initialize MSCP with its forecasting learner and global alpha."""
        super().__init__(learner=learner)
        self.alpha = alpha

    def _get_alpha(self, alpha: float | None = None) -> float:
        """Resolve an optional override against the MSCP global alpha."""
        return self._validate_alpha(self.alpha if alpha is None else alpha)

    def _validate_fit_configuration(self) -> None:
        """Validate the global MSCP significance level before calibration."""
        self._get_alpha()
        self._validate_nexcp()

    @requires_extra("series")
    def fit(
        self,
        df: pd.DataFrame,
        horizon: int,
        n_windows: int = 15,
        step_size: int | None = None,
        static_features: list | None = None,
        nexcp: bool = True,
        decay: float = 0.99,
        weighted_refit: bool = True,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        n_jobs: int = -1,
    ):
        """Fit MSCP with rolling-origin residual calibration.

        Parameters
        ----------
        df : pandas.DataFrame
            Long-format training panel containing identifier, timestamp, target,
            and optional exogenous feature columns.
        horizon : int
            Maximum forecast horizon calibrated in every backtesting window.
        n_windows : int, default=15
            Number of rolling-origin windows used to collect residuals.
        step_size : int or None, default=None
            Distance between consecutive origins. ``None`` uses ``horizon``.
        static_features : list of str or None, default=None
            Time-invariant feature columns passed to the forecasting learner.
        nexcp : bool, default=True
            Give recent calibration windows exponentially larger weights.
        decay : float, default=0.99
            Exponential decay factor in ``(0, 1)`` used when ``nexcp=True``.
        weighted_refit : bool, default=True
            Pass recency weights to compatible learner fits when ``nexcp=True``.
        id_col : str, default="unique_id"
            Series identifier column.
        time_col : str, default="ds"
            Timestamp column.
        target_col : str, default="y"
            Target column.
        n_jobs : int, default=-1
            Parallel jobs used to process calibration windows.

        Returns
        -------
        self
            Fitted estimator with per-series, per-horizon residual scores.

        Raises
        ------
        TypeError
            If calibration parameters have invalid types.
        ValueError
            If parameters, columns, panel layout, or history length are invalid.
        RuntimeError
            If calibration produces no nonconformity scores.

        Notes
        -----
        The learner is fitted on temporary historical windows for calibration
        and finally refitted on the complete panel. Predictions cannot exceed
        the fitted ``horizon``.
        """
        return super().fit(
            df,
            horizon=horizon,
            n_windows=n_windows,
            step_size=step_size,
            static_features=static_features,
            nexcp=nexcp,
            decay=decay,
            weighted_refit=weighted_refit,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            n_jobs=n_jobs,
        )

    def _sample_correction(self, alpha: float):
        """Compute equal-tailed finite-sample conformal quantile levels."""
        return central_conformal_quantile_levels(
            self.n,
            alpha,
            warning_registry=self._quantile_warning_registry,
            context=self.__class__.__name__,
        )

    def _compute_bounds(
        self,
        y_hat: np.ndarray,
        model_name: str,
        h: int,
        prediction_ids: np.ndarray,
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the lower and upper conformal bounds over 1D vectors while
        preserving memory efficiency along the calibration-window axis.
        """
        low_q, high_q = self._sample_correction(alpha)
        scores_by_id = self._require_calibrated_model(model_name)
        series_ids = prediction_ids[::h]
        missing_ids = list(set(series_ids) - set(scores_by_id))
        if missing_ids:
            raise ValueError(
                "No calibration scores are available for forecast identifiers: "
                f"{missing_ids}"
            )

        lower_bound = np.empty_like(y_hat, dtype=float)
        upper_bound = np.empty_like(y_hat, dtype=float)
        weights = self.calibration_weights_
        for row, series_id in enumerate(series_ids):
            row_slice = slice(row * h, (row + 1) * h)
            ncscore = scores_by_id[series_id][:, :h]
            q_low_h = self._compute_qhat(ncscore, low_q, axis=0, weights=weights)
            q_high_h = self._compute_qhat(ncscore, high_q, axis=0, weights=weights)
            lower, upper = core_conformal.signed_residual_bounds(
                y_hat[row_slice], q_low_h, q_high_h
            )
            lower_bound[row_slice] = lower
            upper_bound[row_slice] = upper

        return lower_bound, upper_bound

    def _extract_bound_records(
        self, eval_df: pd.DataFrame, y_true: np.ndarray, alpha: float
    ) -> list[dict]:
        """Build one metrics record per lower/upper interval-column pair."""
        bound_pattern = re.compile(r"^(?P<model>.+)-lo-(?P<level>\d+(?:\.\d+)?)$")
        records = []
        for column in eval_df.columns:
            match = bound_pattern.match(column)
            if not match:
                continue
            model = match.group("model")
            level = match.group("level")
            high_column = f"{model}-hi-{level}"
            if high_column not in eval_df.columns:
                continue
            records.append(
                {
                    "model": model,
                    "level": f"{level}%",
                    "alpha": alpha,
                    **core_conformal.interval_metrics(
                        y_true,
                        eval_df[column].to_numpy(),
                        eval_df[high_column].to_numpy(),
                        alpha,
                    ),
                }
            )
        return records

    @staticmethod
    def _coverage_label(alpha: float) -> str:
        """Format percentage coverage without discarding fractional levels."""
        coverage = (1.0 - alpha) * 100.0
        return np.format_float_positional(coverage, precision=12, trim="-")

    @requires_extra("series")
    def predict_interval(
        self,
        h: int | None = None,
        X_df: pd.DataFrame | None = None,
        alpha: float | None = None,
    ) -> pd.DataFrame:
        """
        Generates prediction intervals [lower, upper] for Nixtla inputs.

        Parameters
        ----------
        h : int, optional
            Forecast horizon. Defaults to the calibrated horizon and cannot exceed
            it.
        X_df : pd.DataFrame, optional
            Future dynamic features. When provided, it must include the identifier,
            time, and all dynamic exogenous columns, with exactly ``h`` rows per
            series and a common timestamp grid.
        alpha : float, optional
            Significance level overriding the value configured at initialization.
            Fractional coverage percentages are preserved in output column names;
            for example, ``alpha=0.055`` produces a ``-94.5`` suffix.

        Returns
        -------
        pd.DataFrame
            Point forecasts and lower/upper interval columns for every model.
        """
        pred_df, h, _, _ = self._generate_forecast(h, X_df)
        model_cols = self._infer_model_cols(pred_df)
        prediction_ids = pred_df[self.id_col].to_numpy()
        alpha = self._get_alpha(alpha)
        level = self._coverage_label(alpha)

        for model in model_cols:
            y_hat = pred_df[model].to_numpy()

            lower_bound, upper_bound = self._compute_bounds(
                y_hat=y_hat,
                model_name=model,
                h=h,
                prediction_ids=prediction_ids,
                alpha=alpha,
            )

            pred_df[f"{model}-lo-{level}"] = lower_bound
            pred_df[f"{model}-hi-{level}"] = upper_bound

        return pred_df

    @requires_extra("series")
    def evaluate(
        self,
        df_test: pd.DataFrame,
        h: int | None = None,
        alpha: float | None = None,
    ) -> pd.DataFrame:
        """Evaluate MSCP intervals using one global or overridden alpha.

        ``df_test`` must provide exactly one non-missing target for every predicted
        identifier and timestamp. Duplicate or missing matches raise ``ValueError``.
        """
        alpha = self._get_alpha(alpha)
        eval_df = self.predict_interval(
            X_df=df_test if self.exog_cols_ else None,
            h=h,
            alpha=alpha,
        )
        eval_df = self._merge_predictions_with_targets(eval_df, df_test)

        y_true = eval_df[self.target_col].to_numpy()
        records = self._extract_bound_records(eval_df, y_true, alpha)

        return pd.DataFrame(records)
