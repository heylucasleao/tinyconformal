# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from tinyconformal.core import conformal as core_conformal
from tinyconformal.core.quantiles import (
    central_conformal_quantile_levels,
    validate_alpha,
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

    Notes
    -----
    Training and future rows must already be ordered chronologically within
    each series. The regressor preserves the supplied order and does not sort
    frames internally.
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
        return validate_alpha(self.alpha if alpha is None else alpha)

    def _validate_fit_configuration(self) -> None:
        """Validate the global MSCP significance level before calibration."""
        self._get_alpha()
        self._validate_nexcp()

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

        Notes
        -----
        Rows in ``X_df`` must already be ordered chronologically within each
        series. This method does not sort the input internally.
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
