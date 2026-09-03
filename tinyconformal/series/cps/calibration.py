"""Conditional dispersion calibration for time-series CPS models."""

# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

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
        """Check whether a method accepts a named or arbitrary keyword argument."""
        signature = inspect.signature(method)
        return parameter in signature.parameters or any(
            item.kind == inspect.Parameter.VAR_KEYWORD
            for item in signature.parameters.values()
        )

    @staticmethod
    def _targets(residuals: np.ndarray) -> np.ndarray:
        """Convert residuals into positive dispersion targets."""
        return np.maximum(np.abs(residuals).reshape(-1), 1e-6)

    @staticmethod
    def _validate(scales: np.ndarray) -> None:
        """Require dispersion predictions to be finite and strictly positive."""
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError(
                "Dispersion learner predictions must be positive and finite."
            )

    @staticmethod
    def _by_series(values: np.ndarray, series_ids) -> dict:
        """Split a window-by-series tensor into matrices keyed by series."""
        return {
            series_id: values[:, row, :] for row, series_id in enumerate(series_ids)
        }

    def _fit_pipeline(self, features, targets, windows) -> Pipeline:
        """Fit a fresh dispersion pipeline, optionally with temporal weights."""
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
        """Generate leave-one-window-out conditional-scale predictions.

        Parameters
        ----------
        residuals : numpy.ndarray
            Signed rolling-origin residuals with shape
            ``(n_windows, n_series, horizon)``.
        features : pandas.DataFrame
            One copy of the complete ``(series_id, horizon)`` grid, ordered as
            series first and horizon second. It has ``n_series * horizon`` rows
            and deliberately contains neither a window identifier nor a target.
        n_series : int
            Number of series represented in ``residuals`` and ``features``.
        n_jobs : int
            Number of parallel jobs used to process held-out windows.

        Returns
        -------
        numpy.ndarray
            Positive OOF scale predictions with the same shape as ``residuals``.
            Slice ``result[w]`` was produced by a pipeline fitted without window
            ``w``.

        Notes
        -----
        Leakage prevention comes from excluding the held-out window's targets
        during fitting, not from presenting previously unseen feature values at
        prediction time. ``series_id`` and ``horizon`` are known independently
        of the observed residual and can therefore occur in both training and
        prediction. The pipeline sees the identity of a series and its forecast
        horizon, but never the held-out window's residual.

        For example, consider one series ``A`` at horizon 1 across three
        backtesting windows::

            window    features    dispersion target
              0        (A, 1)       abs(r[0, A, 1]) = 10
              1        (A, 1)       abs(r[1, A, 1]) = 20
              2        (A, 1)       abs(r[2, A, 1]) = 30

        When window 1 is held out, the pipeline is fitted with::

            X_train       y_train
             (A, 1)         10
             (A, 1)         30

        Calling ``pipeline.predict(features)`` then asks for the scale of
        ``(A, 1)`` using only those training targets. The excluded value 20 is
        not an input to ``predict`` and did not participate in ``fit``. The
        returned value is stored as the OOF scale for window 1 and later used
        to standardize its signed residual::

            standardized[1, A, 1] = r[1, A, 1] / scale_oof[1, A, 1]

        In the general panel case, each fit receives the same complete feature
        grid repeated once per included training window. A prediction receives
        one copy of that grid and returns ``n_series * horizon`` values, which
        are reshaped to ``(n_series, horizon)`` and assigned only to the current
        held-out window. After every window is processed, those slices form the
        final ``(n_windows, n_series, horizon)`` result.
        """

        def process_window(window):
            """Fit without one window and predict its conditional scales."""
            train_windows = np.delete(np.arange(self.n_windows), window)
            train_residuals = residuals[train_windows]
            train_features = pd.concat(
                [features] * len(train_residuals), ignore_index=True
            )
            pipeline = self._fit_pipeline(
                train_features,
                self._targets(train_residuals),
                train_windows,
            )
            # Predict one (series, horizon) grid for the held-out OOF window;
            # predictions for all windows are assembled after this function returns.
            scales = np.asarray(pipeline.predict(features), dtype=float).reshape(
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
        """Cross-fit conditional scales and standardize calibration residuals.

        Parameters
        ----------
        residuals_by_model : dict
            Signed rolling-origin residuals keyed first by forecast-model name
            and then by series identifier. Each per-series value must have shape
            ``(n_windows, horizon)`` and contain ``y_hat - y`` residuals.
            Exactly one forecast model is supported.
        n_jobs : int, default=-1
            Number of parallel jobs used by the leave-one-window-out fits.

        Returns
        -------
        standardized : dict
            Residuals keyed by forecast model and series identifier. Each value
            has shape ``(n_windows, horizon)`` and contains the signed scores
            ``residual / OOF scale``.
        oof_scales : dict
            Positive out-of-fold scale predictions with the same nested keys and
            per-series shapes as ``standardized``.
        fitted_models : dict
            Final dispersion pipelines keyed by forecast-model name. Each
            pipeline is refitted on every calibration window and is intended to
            predict scales for future forecasts.

        Notes
        -----
        Residuals are first stacked as a tensor with shape
        ``(n_windows, n_series, horizon)``. For each window, a fresh pipeline is
        trained on the absolute residuals from all other windows. Its inputs are
        series identity and numerical forecast horizon, and its predictions for
        the held-out window form the corresponding slice of ``oof_scales``.

        The window index is not a model feature. The same ``(series, horizon)``
        feature grid is repeated once per training window, providing repeated
        absolute-error targets for each combination. OOF predictions can still
        differ between windows because every leave-one-window-out pipeline sees
        a different training subset.

        Standardization uses only held-out scale predictions, preventing a
        residual from directly determining its own normalization. Once all OOF
        scores have been produced, a separate final pipeline is trained on all
        windows. Temporal weights are applied during these fits only when
        ``nexcp`` and ``weighted_refit`` are enabled and the dispersion learner
        accepts ``sample_weight``.
        """
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

    def predict(self, pipeline: Pipeline, series_ids, horizon_steps) -> np.ndarray:
        """Predict conditional scales for future series-horizon rows.

        Parameters
        ----------
        pipeline : Pipeline
            Fitted dispersion pipeline, normally one of the final pipelines
            returned by :meth:`fit_transform`.
        series_ids : array-like
            Series identifier for each future forecast row.
        horizon_steps : array-like
            Zero-based horizon position for each future forecast row. Values are
            converted to the one-based horizons used during scale fitting.

        Returns
        -------
        numpy.ndarray
            One finite, strictly positive scale per input row, in the same order
            as ``series_ids`` and ``horizon_steps``.

        Notes
        -----
        This method predicts dispersion, not a signed residual or a point
        forecast. The resulting scales convert standardized conformal residuals
        back to the target's units when the predictive distribution is built.
        Series unseen during fitting are accepted by the one-hot encoder and
        receive an all-zero series encoding; their predictions therefore depend
        on the learner's behavior for that representation and on horizon.
        """
        features = pd.DataFrame(
            {"series_id": series_ids, "horizon": np.asarray(horizon_steps) + 1}
        )
        scales = np.asarray(pipeline.predict(features), dtype=float)
        self._validate(scales)
        return scales
