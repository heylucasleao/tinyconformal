"""Conditional dispersion calibration for time-series CPS models."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from tinyconformal.core.quantiles import temporal_decay_weights


class ConditionalScaleCalibrator:
    """Cross-fit conditional residual scales by series and forecast horizon."""

    def __init__(
        self,
        learner: BaseEstimator,
        horizon: int,
        n_windows: int,
        nexcp: bool = False,
        decay: float = 0.99,
        weighted_refit: bool = True,
    ):
        self.learner = learner
        self.horizon = horizon
        self.n_windows = n_windows
        self.nexcp = nexcp
        self.decay = decay
        self.weighted_refit = weighted_refit

    def features(self, series_ids) -> pd.DataFrame:
        """Build one dispersion-feature row per series and horizon."""
        series_ids = list(series_ids)
        return pd.DataFrame(
            {
                "series_id": np.repeat(series_ids, self.horizon),
                "horizon": np.tile(np.arange(1, self.horizon + 1), len(series_ids)),
            }
        )

    def new_pipeline(self) -> Pipeline:
        """Create an unfitted conditional-dispersion pipeline."""
        return Pipeline(
            [
                (
                    "features",
                    ColumnTransformer(
                        [
                            (
                                "series",
                                OneHotEncoder(
                                    handle_unknown="ignore", sparse_output=False
                                ),
                                ["series_id"],
                            ),
                            ("horizon", "passthrough", ["horizon"]),
                        ]
                    ),
                ),
                ("learner", clone(self.learner)),
            ]
        )

    @staticmethod
    def _accepts_parameter(method, parameter: str) -> bool:
        signature = inspect.signature(method)
        return parameter in signature.parameters or any(
            item.kind == inspect.Parameter.VAR_KEYWORD
            for item in signature.parameters.values()
        )

    @staticmethod
    def _targets(residuals: np.ndarray) -> np.ndarray:
        return np.maximum(np.abs(residuals).reshape(-1), 1e-6)

    @staticmethod
    def _validate(scales: np.ndarray) -> None:
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError(
                "Dispersion learner predictions must be positive and finite."
            )

    @staticmethod
    def _by_series(values: np.ndarray, series_ids) -> dict:
        return {
            series_id: values[:, row, :] for row, series_id in enumerate(series_ids)
        }

    def _fit_pipeline(self, features, targets, windows) -> Pipeline:
        pipeline = self.new_pipeline()
        fit_kwargs = {}
        if (
            self.nexcp
            and self.weighted_refit
            and self._accepts_parameter(self.learner.fit, "sample_weight")
        ):
            window_weights = temporal_decay_weights(self.n_windows, self.decay)[windows]
            repeats = len(features) // len(windows)
            fit_kwargs["learner__sample_weight"] = np.repeat(window_weights, repeats)
        return pipeline.fit(features, targets, **fit_kwargs)

    def _oof_scales(
        self,
        residuals: np.ndarray,
        features: pd.DataFrame,
        n_series: int,
        n_jobs: int,
    ) -> np.ndarray:
        def process_window(window):
            train_windows = np.delete(np.arange(self.n_windows), window)
            train_residuals = residuals[train_windows]
            train_features = pd.concat(
                [features] * len(train_residuals), ignore_index=True
            )
            model = self._fit_pipeline(
                train_features,
                self._targets(train_residuals),
                train_windows,
            )
            scales = np.asarray(model.predict(features), dtype=float).reshape(
                n_series, self.horizon
            )
            return window, scales

        results = Parallel(n_jobs=n_jobs)(
            delayed(process_window)(window) for window in range(self.n_windows)
        )
        scales = np.empty_like(residuals, dtype=float)
        for window, window_scales in results:
            scales[window] = window_scales
        return scales

    def fit_transform(self, residuals_by_model, n_jobs: int = -1):
        """Fit scales and return standardized residuals and fitted artifacts."""
        if self.n_windows < 2:
            raise ValueError(
                "TSCPS requires at least two windows for scale cross-fitting."
            )
        if len(residuals_by_model) != 1:
            raise ValueError(
                "TSCPS requires a Nixtla learner configured with exactly one model; "
                f"found model columns {list(residuals_by_model)}."
            )

        standardized = {}
        oof_scales = {}
        fitted_models = {}
        for model, scores_by_id in residuals_by_model.items():
            series_ids = list(scores_by_id)
            features = self.features(series_ids)
            residuals = np.stack(
                [scores_by_id[series_id] for series_id in series_ids], axis=1
            )
            scales = self._oof_scales(residuals, features, len(series_ids), n_jobs)
            self._validate(scales)
            standardized[model] = self._by_series(residuals / scales, series_ids)
            oof_scales[model] = self._by_series(scales, series_ids)
            final_features = pd.concat([features] * self.n_windows, ignore_index=True)
            fitted_models[model] = self._fit_pipeline(
                final_features,
                self._targets(residuals),
                np.arange(self.n_windows),
            )
        return standardized, oof_scales, fitted_models

    def predict(self, model: Pipeline, series_ids, horizon_steps) -> np.ndarray:
        """Predict and validate scales for future series-step rows."""
        features = pd.DataFrame(
            {"series_id": series_ids, "horizon": np.asarray(horizon_steps) + 1}
        )
        scales = np.asarray(model.predict(features), dtype=float)
        self._validate(scales)
        return scales
