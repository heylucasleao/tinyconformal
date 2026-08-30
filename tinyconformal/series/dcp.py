# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from tinyconformal.distribution import (
    DistributionalConformalDistribution,
    QuantileGridDistribution,
)
from tinyconformal.distribution.distributional import validate_quantile_levels
from tinyconformal.utils.imports import requires_extra

from .mscp import ConformalDistributionTimeSeriesRegressor


class DistributionalConformalPredictiveSystemTimeSeriesRegressor(
    ConformalDistributionTimeSeriesRegressor
):
    """PIT-based, horizon-wise DCP for Nixtla-compatible estimators.

    Parameters
    ----------
    quantile_columns : dict
        Either ``{probability: column}`` for one model or
        ``{model: {probability: column}}`` for multiple models. Every configured
        column must be emitted by the Nixtla learner's ``predict`` method.
    """

    def __init__(
        self,
        learner: BaseEstimator,
        horizon: int,
        quantile_columns: dict,
        n_windows: int = 10,
        alpha: float = 0.05,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ):
        super().__init__(
            learner=learner,
            horizon=horizon,
            n_windows=n_windows,
            alpha=alpha,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
        )
        self.quantile_columns = quantile_columns
        self.quantile_columns_ = self._normalize_quantile_columns(quantile_columns)

    @staticmethod
    def _infer_model_name(columns: list[str]) -> str:
        prefixes = []
        for column in columns:
            match = re.match(r"^(?P<model>.+)-q-(?:\d+(?:\.\d+)?)$", column)
            if match is None:
                raise ValueError(
                    "For a flat quantile_columns mapping, column names must follow "
                    "'<model>-q-<percent>', or use the nested mapping form."
                )
            prefixes.append(match.group("model"))
        if len(set(prefixes)) != 1:
            raise ValueError("A flat quantile_columns mapping must refer to one model.")
        return prefixes[0]

    def _normalize_one_grid(self, mapping: dict) -> dict[float, str]:
        if not isinstance(mapping, dict) or len(mapping) < 2:
            raise ValueError("Each quantile column mapping must contain at least two levels.")
        try:
            pairs = sorted((float(level), column) for level, column in mapping.items())
        except (TypeError, ValueError) as exc:
            raise ValueError("Quantile mapping keys must be numeric probabilities.") from exc
        levels = validate_quantile_levels([level for level, _ in pairs])
        columns = [column for _, column in pairs]
        if not all(isinstance(column, str) for column in columns):
            raise TypeError("Quantile mapping values must be column names.")
        if len(set(columns)) != len(columns):
            raise ValueError("Quantile column names must be unique within each model.")
        return dict(zip(levels, columns))

    def _normalize_quantile_columns(
        self, quantile_columns: dict
    ) -> dict[str, dict[float, str]]:
        if not isinstance(quantile_columns, dict) or not quantile_columns:
            raise ValueError("quantile_columns must be a non-empty dictionary.")
        if all(isinstance(value, str) for value in quantile_columns.values()):
            grid = self._normalize_one_grid(quantile_columns)
            return {self._infer_model_name(list(grid.values())): grid}
        if not all(isinstance(value, dict) for value in quantile_columns.values()):
            raise ValueError(
                "quantile_columns must be entirely flat or entirely nested by model."
            )
        return {
            str(model): self._normalize_one_grid(mapping)
            for model, mapping in quantile_columns.items()
        }

    def _validate_fit_configuration(self) -> None:
        super()._validate_fit_configuration()
        # Re-normalize because sklearn parameters may have been changed via set_params.
        self.quantile_columns_ = self._normalize_quantile_columns(
            self.quantile_columns
        )

    def _all_quantile_columns(self) -> tuple[str, ...]:
        return tuple(
            column
            for mapping in self.quantile_columns_.values()
            for column in mapping.values()
        )

    def _compute_window_residuals(
        self,
        fcst: pd.DataFrame,
        val_df: pd.DataFrame,
        n_series: int,
        residuals_by_model: dict[str, list],
    ) -> None:
        """Compute PIT matrices instead of signed residual matrices."""
        self._require_forecast_columns(fcst, self._all_quantile_columns())
        target_pivot, y_true = self._extract_target_panel(val_df, n_series)

        for model, mapping in self.quantile_columns_.items():
            levels = np.asarray(list(mapping), dtype=float)
            forecast_pivots = [
                self._pivot_panel(fcst, column) for column in mapping.values()
            ]
            arrays = [pivot.to_numpy() for pivot in forecast_pivots]
            self._validate_calibration_forecasts(
                forecast_pivots[0].index, target_pivot, *arrays
            )
            grid = np.stack(arrays, axis=-1).reshape(-1, levels.size)
            base = QuantileGridDistribution(grid, levels)
            pits = np.asarray(base.cdf(y_true.reshape(-1))).reshape(
                n_series, self.horizon
            )
            residuals_by_model.setdefault(model, []).append(pits)

    def _prediction_frame(
        self, h: int | None, X_df: pd.DataFrame | None
    ) -> tuple[pd.DataFrame, int]:
        h = self._get_horizon(h)
        self._check_is_fitted()
        X_df = self._validate_prediction_features(X_df, h)
        pred_df = (
            self._invoke(self.learner.predict, h=h, X_df=X_df)
            .sort_values([self.id_col, self.time_col])
            .reset_index(drop=True)
        )
        self._require_forecast_columns(pred_df, self._all_quantile_columns())
        n_series = self._validate_prediction_panel(pred_df, h)
        return pred_df, n_series

    def _build_distributions(
        self, pred_df: pd.DataFrame, h: int, n_series: int
    ) -> dict[str, DistributionalConformalDistribution]:
        horizon_steps = np.tile(np.arange(h), n_series)
        distributions = {}
        for model, mapping in self.quantile_columns_.items():
            if model not in self.ncscores_:
                raise ValueError(
                    f"Model '{model}' was not present during PIT calibration."
                )
            levels = np.asarray(list(mapping), dtype=float)
            grid = pred_df[list(mapping.values())].to_numpy(dtype=float)
            base = QuantileGridDistribution(grid, levels)
            horizon_pits = self.ncscores_[model][:, :h]
            row_pits = horizon_pits[:, horizon_steps].T
            distributions[model] = DistributionalConformalDistribution(
                base, row_pits
            )
        return distributions

    @requires_extra("series")
    def predict_distribution(
        self,
        h: int | None = None,
        X_df: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, dict[str, DistributionalConformalDistribution]]:
        """Return the Nixtla quantile frame and aligned PIT-calibrated distributions."""
        h = self._get_horizon(h)
        pred_df, n_series = self._prediction_frame(h, X_df)
        return pred_df, self._build_distributions(pred_df, h, n_series)

    @requires_extra("series")
    def predict_quantiles(
        self,
        quantiles,
        h: int | None = None,
        X_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        levels = np.asarray(quantiles, dtype=float)
        if levels.ndim == 0:
            levels = levels.reshape(1)
        if levels.ndim != 1 or levels.size == 0 or not np.all(np.isfinite(levels)):
            raise ValueError("quantiles must be a non-empty finite scalar or 1D sequence.")
        if np.any((levels < 0.0) | (levels > 1.0)):
            raise ValueError("quantiles must lie in [0, 1].")

        pred_df, distributions = self.predict_distribution(h=h, X_df=X_df)
        for model, distribution in distributions.items():
            matrix = np.broadcast_to(levels, (len(distribution), levels.size))
            values = distribution.ppf(matrix)
            for index, level in enumerate(levels):
                label = np.format_float_positional(
                    level * 100.0, precision=12, trim="-"
                )
                pred_df[f"{model}-q-{label}-dcp"] = values[:, index]
        return pred_df

    @requires_extra("series")
    def predict_interval(
        self,
        h: int | None = None,
        X_df: pd.DataFrame | None = None,
        alpha: float | None = None,
    ) -> pd.DataFrame:
        alpha = self._get_alpha(alpha)
        pred_df, distributions = self.predict_distribution(h=h, X_df=X_df)
        level = self._coverage_label(alpha)
        for model, distribution in distributions.items():
            bounds = distribution.interval(1.0 - alpha)
            pred_df[f"{model}-lo-{level}-dcp"] = bounds[:, 0]
            pred_df[f"{model}-hi-{level}-dcp"] = bounds[:, 1]
        return pred_df

    @requires_extra("series")
    def evaluate(
        self,
        df_test: pd.DataFrame,
        h: int | None = None,
        alpha: float | None = None,
    ) -> pd.DataFrame:
        alpha = self._get_alpha(alpha)
        eval_df = self.predict_interval(
            X_df=self._prediction_features(df_test), h=h, alpha=alpha
        )
        eval_df = self._merge_predictions_with_targets(eval_df, df_test)
        y_true = eval_df[self.target_col].to_numpy()
        level = self._coverage_label(alpha)
        records = []
        for model in self.quantile_columns_:
            lower = eval_df[f"{model}-lo-{level}-dcp"].to_numpy()
            upper = eval_df[f"{model}-hi-{level}-dcp"].to_numpy()
            records.append(
                {
                    "model": model,
                    "level": f"{level}%",
                    "alpha": alpha,
                    "coverage_rate": np.round(
                        self._coverage_rate(y_true, lower, upper), 3
                    ),
                    "interval_width_mean": np.round(
                        self._interval_width_mean(lower, upper), 3
                    ),
                    "mwis": np.round(
                        self._mwi_score(y_true, lower, upper, alpha), 3
                    ),
                }
            )
        return pd.DataFrame(records).sort_values("model").reset_index(drop=True)
