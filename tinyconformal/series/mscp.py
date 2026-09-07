# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from tinyconformal.core import conformal as core_conformal
from tinyconformal.core.quantiles import central_conformal_quantile_levels
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
        alpha: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the lower and upper conformal bounds over 1D vectors while
        preserving memory efficiency along the calibration-window axis.
        """
        alpha = self._get_alpha(alpha)
        low_q, high_q = self._sample_correction(alpha)
        scores_by_id = self.ncscores_[model_name]
        missing_ids = sorted(set(prediction_ids) - set(scores_by_id), key=str)
        if missing_ids:
            raise ValueError(
                "No calibration scores are available for forecast identifiers: "
                f"{missing_ids}"
            )

        lower_bound = np.empty_like(y_hat, dtype=float)
        upper_bound = np.empty_like(y_hat, dtype=float)
        for series_id in pd.unique(prediction_ids):
            row_mask = prediction_ids == series_id
            ncscore = scores_by_id[series_id][:, :h]
            q_low_h = self._compute_qhat(ncscore, low_q, axis=0)
            q_high_h = self._compute_qhat(ncscore, high_q, axis=0)
            if row_mask.sum() != h:
                raise ValueError(
                    f"Forecast identifier {series_id!r} must contain exactly {h} rows."
                )
            lower, upper = core_conformal.signed_residual_bounds(
                y_hat[row_mask], q_low_h, q_high_h
            )
            lower_bound[row_mask] = lower
            upper_bound[row_mask] = upper

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
        """
        h = self._get_horizon(h)
        self._check_is_fitted()
        X_df = self._validate_prediction_features(X_df, h)

        pred_df = (
            self._invoke(
                self.learner.predict,
                h=h,
                X_df=X_df,
            )
            .sort_values(by=[self.id_col, self.time_col])
            .reset_index(drop=True)
        )
        model_cols = self._infer_model_cols(pred_df)
        self._validate_prediction_panel(pred_df, h)

        for model in model_cols:
            if model not in self.ncscores_:
                raise ValueError(
                    f"Model column '{model}' was not present during calibration. "
                    f"Calibrated model columns: {list(self.ncscores_)}"
                )
            y_hat = pred_df[model].to_numpy()

            lower_bound, upper_bound = self._compute_bounds(
                y_hat=y_hat,
                model_name=model,
                h=h,
                prediction_ids=pred_df[self.id_col].to_numpy(),
                alpha=alpha,
            )
            eff_alpha = self._get_alpha(alpha)
            level = self._coverage_label(eff_alpha)

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
            X_df=self._prediction_features(df_test), h=h, alpha=alpha
        )
        eval_df = self._merge_predictions_with_targets(eval_df, df_test)

        y_true = eval_df[self.target_col].to_numpy()
        records = self._extract_bound_records(eval_df, y_true, alpha)

        return (
            pd.DataFrame(records)
            .sort_values(by=["model", "level"])
            .reset_index(drop=True)
        )
