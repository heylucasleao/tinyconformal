# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

from __future__ import annotations

import re
from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from tinyconformal.core.quantiles import temporal_decay_weights
from tinyconformal.distribution.base import (
    DiscretePredictiveDistribution,
    EmpiricalResidualDistribution,
    PredictiveDistribution,
)
from tinyconformal.utils.imports import requires_extra

from .mscp import ConformalDistributionTimeSeriesRegressor


def _validate_matrix(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError(f"{name} must be a non-empty two-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


class HorizonConformalDistribution(EmpiricalResidualDistribution):
    """Batch of residual-based predictive distributions calibrated by horizon.

    Each prediction is represented as a point forecast plus the empirical
    distribution of signed calibration residuals for its forecast step.  The
    object implements the common :class:`PredictiveDistribution` interface, so
    callers can evaluate CDFs, request arbitrary quantiles, construct central
    intervals, or draw samples without refitting the forecasting model.

    Parameters
    ----------
    locations : ndarray of shape (n_predictions,)
        Point forecasts in sorted Nixtla panel order.
    residuals : ndarray or mapping
        Signed residuals ``y - y_hat`` from sequential backtesting. A matrix
        applies the same calibration distribution to every prediction. A
        mapping associates each series identifier with its own matrix of shape
        ``(n_calibration_trajectories, horizon)``.
    horizon_steps : ndarray of shape (n_predictions,)
        Zero-based horizon index associated with every point forecast.
    series_ids : ndarray of shape (n_predictions,), optional
        Series identifier for every prediction. Required when ``residuals`` is
        a mapping.
    scales : ndarray of shape (n_predictions,), optional
        Positive conditional scale for each prediction. Defaults to one.
    weights : ndarray of shape (n_calibration_trajectories,), optional
        Non-negative calibration-window weights. They are normalized internally;
        equal conformal ranks are used when omitted.

    Attributes
    ----------
    locations : ndarray of shape (n_predictions,)
        Point forecasts defining the location of each predictive distribution.
    residuals : ndarray or dict
        Sorted signed calibration residuals, either pooled or keyed by series.
    horizon_steps : ndarray of shape (n_predictions,)
        Horizon step used to select the residual distribution for each row.
    scales : ndarray of shape (n_predictions,)
        Conditional scales applied to the standardized residuals.
    weights : ndarray or None
        Normalized calibration-window weights.

    Notes
    -----
    If ``r = y - y_hat`` denotes a signed calibration residual, the predictive
    distribution for row ``i`` is the empirical distribution of
    ``locations[i] + r[:, horizon_steps[i]]``.  The CDF uses conformal ranks with
    denominator ``n_calibration_trajectories + 1`` and the PPF uses the
    corresponding finite-sample ceiling rule. When ``weights`` are supplied,
    CDFs and quantiles use their normalized weighted empirical counterparts.

    Rows are positional.  Their order must remain identical to the associated
    Nixtla forecast DataFrame returned by ``predict_distribution``.
    """

    def __init__(
        self,
        locations,
        residuals,
        horizon_steps,
        series_ids=None,
        scales=None,
        weights=None,
    ):
        self.locations = self._validate_locations(locations)
        self.horizon_steps = self._validate_horizon_steps(horizon_steps)
        self.residuals, self.series_ids = self._prepare_residuals(residuals, series_ids)
        self.scales = self._validate_scales(scales)
        self._n_calibration, calibrated_horizon = self._residual_shape()
        self.weights = self._validate_weights(weights)
        self._validate_calibrated_horizon(calibrated_horizon)

    def _validate_scales(self, scales) -> np.ndarray:
        if scales is None:
            return np.ones_like(self.locations)
        scales = np.asarray(scales, dtype=float)
        if scales.shape != self.locations.shape or not np.all(np.isfinite(scales)):
            raise ValueError("scales must be finite and match locations.")
        if np.any(scales <= 0.0):
            raise ValueError("scales must be strictly positive.")
        return scales

    def _validate_weights(self, weights):
        if weights is None:
            return None
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (self._n_calibration,):
            raise ValueError("weights must match the number of calibration windows.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("weights must be finite, non-negative, and have positive mass.")
        return weights / weights.sum()

    @staticmethod
    def _validate_locations(locations) -> np.ndarray:
        locations = np.asarray(locations, dtype=float)
        if locations.ndim != 1 or not np.all(np.isfinite(locations)):
            raise ValueError("locations must be a finite one-dimensional array.")
        return locations

    def _validate_horizon_steps(self, horizon_steps) -> np.ndarray:
        horizon_steps = np.asarray(horizon_steps, dtype=int)
        if horizon_steps.shape != self.locations.shape:
            raise ValueError("horizon_steps and locations must have the same shape.")
        return horizon_steps

    def _prepare_residuals(self, residuals, series_ids):
        if isinstance(residuals, Mapping):
            return self._prepare_series_residuals(residuals, series_ids)
        residuals = _validate_matrix(residuals, "residuals")
        return residuals, None

    def _prepare_series_residuals(self, residuals, series_ids):
        if series_ids is None:
            raise ValueError("series_ids is required when residuals is a mapping.")
        series_ids = np.asarray(series_ids)
        if series_ids.shape != self.locations.shape:
            raise ValueError("series_ids and locations must have the same shape.")
        missing_ids = sorted(set(series_ids) - set(residuals), key=str)
        if missing_ids:
            raise ValueError(
                "No CPS calibration residuals are available for series: "
                f"{missing_ids}"
            )
        prepared = {
            series_id: _validate_matrix(values, f"residuals[{series_id!r}]")
            for series_id, values in residuals.items()
        }
        return prepared, series_ids

    def _residual_shape(self) -> tuple[int, int]:
        if not isinstance(self.residuals, Mapping):
            return self.residuals.shape
        shapes = {values.shape for values in self.residuals.values()}
        if len(shapes) != 1:
            raise ValueError("All series residual matrices must have the same shape.")
        return next(iter(shapes))

    def _validate_calibrated_horizon(self, calibrated_horizon: int) -> None:
        if np.any(
            (self.horizon_steps < 0) | (self.horizon_steps >= calibrated_horizon)
        ):
            raise ValueError("horizon_steps contains an uncalibrated horizon index.")

    def __len__(self) -> int:
        return self.locations.size

    @property
    def n_calibration(self) -> int:
        return self._n_calibration

    def _row_residuals(self) -> np.ndarray:
        if self.series_ids is not None:
            residuals = np.vstack(
                [
                    self.residuals[series_id][:, horizon_step]
                    for series_id, horizon_step in zip(
                        self.series_ids, self.horizon_steps
                    )
                ]
            )
        else:
            residuals = self.residuals[:, self.horizon_steps].T
        residuals = self.scales[:, None] * residuals
        return residuals if self.weights is not None else np.sort(residuals, axis=1)

    def cdf(self, values):
        if self.weights is None:
            return super().cdf(values)
        values, squeeze = self._rowwise_or_grid(values, "values")
        scores = values - self.locations[:, None]
        residuals = self._row_residuals()
        result = np.sum(
            self.weights[None, :, None]
            * (residuals[:, :, None] <= scores[:, None, :]),
            axis=1,
        )
        return result[:, 0] if squeeze else result

    def ppf(self, quantiles):
        if self.weights is None:
            return super().ppf(quantiles)
        quantiles, squeeze = self._rowwise_or_grid(quantiles, "quantiles")
        if np.any((quantiles < 0.0) | (quantiles > 1.0)):
            raise ValueError("quantiles must lie in [0, 1].")
        residuals = self._row_residuals()
        order = np.argsort(residuals, axis=1)
        sorted_residuals = np.take_along_axis(residuals, order, axis=1)
        row_weights = np.broadcast_to(self.weights, residuals.shape)
        sorted_weights = np.take_along_axis(row_weights, order, axis=1)
        cumulative = np.cumsum(sorted_weights, axis=1)
        indices = np.argmax(
            cumulative[:, :, None] >= quantiles[:, None, :], axis=1
        )
        selected = np.take_along_axis(sorted_residuals, indices, axis=1)
        result = self.locations[:, None] + selected
        return result[:, 0] if squeeze else result


class DiscreteHorizonConformalDistribution(
    HorizonConformalDistribution, DiscretePredictiveDistribution
):
    """Horizon-wise conformal predictive distributions on integer support.

    This specialization converts CPS quantiles to integers with ``ceil`` and
    optionally truncates the support below at ``minimum``.  It also inherits
    :meth:`~tinyconformal.distribution.base.DiscretePredictiveDistribution.pmf`,
    which evaluates probability masses as ``F(k) - F(k - 1)``.

    Parameters
    ----------
    locations : ndarray of shape (n_predictions,)
        Point forecasts in sorted Nixtla panel order.
    residuals : ndarray of shape (n_calibration_trajectories, horizon)
        Signed residuals ``y - y_hat`` obtained by sequential backtesting.
    horizon_steps : ndarray of shape (n_predictions,)
        Zero-based forecast step associated with each prediction row.
    minimum : int or None, default=0
        Lower boundary of the integer support.  If ``None``, no lower boundary
        is imposed.
    series_ids : ndarray of shape (n_predictions,), optional
        Series identifier for every prediction. Required when ``residuals`` is
        a mapping.
    scales : ndarray of shape (n_predictions,), optional
        Positive conditional scale for each prediction. Defaults to one.
    weights : ndarray of shape (n_calibration_trajectories,), optional
        Non-negative calibration-window weights, normalized internally.

    Notes
    -----
    The class is intended for genuinely discrete targets such as demand counts.
    Rounding a continuous target merely to expose a PMF changes the statistical
    object and is not equivalent to a continuous CPS.
    """

    def __init__(
        self,
        locations,
        residuals,
        horizon_steps,
        minimum: int | None = 0,
        series_ids=None,
        scales=None,
        weights=None,
    ):
        super().__init__(
            locations,
            residuals,
            horizon_steps,
            series_ids=series_ids,
            scales=scales,
            weights=weights,
        )
        if minimum is not None and not isinstance(minimum, (int, np.integer)):
            raise TypeError("minimum must be an integer or None.")
        self.minimum = None if minimum is None else int(minimum)

    def ppf(self, quantiles):
        result = np.ceil(super().ppf(quantiles))
        if self.minimum is not None:
            result = np.maximum(result, self.minimum)
        return result.astype(int)

    def cdf(self, values):
        values = np.floor(np.asarray(values, dtype=float))
        result = super().cdf(values)
        if self.minimum is None:
            return result
        below = values < self.minimum
        if values.ndim == 0:
            return np.zeros_like(result) if bool(below) else result
        if result.ndim == 1:
            return np.where(np.ravel(below), 0.0, result)
        return np.where(np.broadcast_to(below, np.shape(result)), 0.0, result)


class _PanelConformalForecast:
    """Panel-aligned facade over one horizon-wise predictive distribution."""

    def __init__(self, frame, distribution, model, id_col, time_col):
        self._frame = frame.copy()
        self._distribution = distribution
        self.model = model
        self.id_col = id_col
        self.time_col = time_col

    def __len__(self) -> int:
        return len(self._distribution)

    def to_frame(self) -> pd.DataFrame:
        """Return the point-forecast panel without distributional columns."""
        return self._frame.copy()

    def _output_frame(self) -> pd.DataFrame:
        return self._frame.copy()

    @staticmethod
    def _label(value) -> str:
        return np.format_float_positional(float(value), precision=12, trim="-")

    def _apply(self, method: str, inputs, prefix: str) -> pd.DataFrame:
        inputs_array = np.asarray(inputs)
        values = np.asarray(getattr(self._distribution, method)(inputs))
        result = self._output_frame()
        if values.ndim == 1:
            if inputs_array.ndim == 0:
                label = self._label(inputs_array)
                column = f"{self.model}-{prefix}-{label}"
            else:
                column = f"{self.model}-{prefix}"
            result[column] = values
            return result
        labels = np.ravel(inputs_array)
        if labels.size != values.shape[1]:
            labels = np.arange(values.shape[1])
        for index, label in enumerate(labels):
            result[f"{self.model}-{prefix}-{self._label(label)}"] = values[:, index]
        return result

    def cdf(self, values) -> pd.DataFrame:
        """Evaluate CDF values and return them on the forecast panel grid."""
        return self._apply("cdf", values, "cdf")

    def ppf(self, quantiles) -> pd.DataFrame:
        """Evaluate quantiles and return them on the forecast panel grid."""
        quantiles_array = np.asarray(quantiles, dtype=float)
        result = self._apply("ppf", quantiles, "q")
        if quantiles_array.ndim == 0:
            old = f"{self.model}-q-{self._label(quantiles_array)}"
            new = f"{self.model}-q-{self._label(100.0 * quantiles_array)}"
            return result.rename(columns={old: new})
        if quantiles_array.ndim == 1:
            return result.rename(
                columns={
                    f"{self.model}-q-{self._label(q)}":
                    f"{self.model}-q-{self._label(100.0 * q)}"
                    for q in quantiles_array
                }
            )
        return result

    def interval(self, coverage: float = 0.95) -> pd.DataFrame:
        """Return a central interval on the forecast panel grid."""
        bounds = np.asarray(self._distribution.interval(coverage))
        level = self._label(100.0 * float(coverage))
        result = self._output_frame()
        result[f"{self.model}-lo-{level}"] = bounds[:, 0]
        result[f"{self.model}-hi-{level}"] = bounds[:, 1]
        return result

    def sample(self, n_samples: int = 1, random_state=None) -> pd.DataFrame:
        """Draw samples and return one panel column per draw."""
        samples = np.asarray(self._distribution.sample(n_samples, random_state))
        result = self._output_frame()
        for index in range(samples.shape[1]):
            result[f"{self.model}-sample-{index + 1}"] = samples[:, index]
        return result

    def evaluate(self, y, coverages=(0.5, 0.8, 0.9, 0.95)) -> pd.DataFrame:
        """Evaluate the underlying predictive distribution."""
        return self._distribution.evaluate(y, coverages=coverages)


class _DiscretePanelConformalForecast(_PanelConformalForecast):
    """Panel-aligned facade that additionally exposes integer probability mass."""

    def pmf(self, values) -> pd.DataFrame:
        """Evaluate PMF values and return them on the forecast panel grid."""
        return self._apply("pmf", values, "pmf")


class _TSCPS(ConformalDistributionTimeSeriesRegressor):
    """Conformal predictive system for multi-step panel forecasting.

    The regressor calibrates complete residual distributions for each forecast
    horizon of a Nixtla-compatible learner, such as ``MLForecast`` or
    ``StatsForecast``.  Sequential rolling-origin backtesting produces signed
    residual trajectories. Unlike an interval-only conformal method, CPS keeps
    those empirical distributions and can therefore return CDFs, arbitrary
    quantiles, samples, and intervals after a single calibration fit.

    The Nixtla learner must expose exactly one model column. Consequently,
    ``predict_distribution`` returns one self-contained forecast whose methods
    produce pandas DataFrames aligned to the original panel grid.

    Parameters
    ----------
    learner : BaseEstimator
        Unfitted Nixtla-compatible forecasting estimator.  Its ``fit`` method
        must accept a long-format panel and its ``predict`` method must return
        ``id_col``, ``time_col``, and one or more model forecast columns.
    dispersion_learner : BaseEstimator
        Regression estimator for the positive conditional scale. It is
        cross-fitted on absolute rolling-origin errors using series and horizon.
    horizon : int
        Maximum forecast horizon calibrated during rolling-origin backtesting.
    n_windows : int, default=10
        Number of backtesting windows.  Each series contributes one residual
        trajectory per window.
    alpha : float, default=0.05
        Default significance level used by ``evaluate``. It does not restrict
        the intervals or quantiles available from the fitted CPS.
    nexcp : bool, default=False
        Whether to weight calibration windows by exponential recency decay.
    decay : float, default=0.99
        Decay factor in ``(0, 1)`` used when ``nexcp=True``.
    weighted_refit : bool, default=True
        Whether recency weights are also passed to the forecasting learner and,
        when supported, the dispersion learner during fitting.
    discrete : bool, default=False
        Whether to construct integer-support predictive distributions.
    minimum : int or None, default=0
        Lower support boundary used when ``discrete=True``. Use ``0`` for
        counts, ``1`` for strictly positive outcomes, another integer for a
        known lower bound, or ``None`` when negative integers are valid.
        Ignored for continuous distributions.
    id_col : str, default="unique_id"
        Column identifying the individual time series.
    time_col : str, default="ds"
        Column containing ordered timestamps.
    target_col : str, default="y"
        Column containing observed targets.

    Attributes
    ----------
    learner : BaseEstimator
        Fitted forecasting learner after ``fit`` completes.
    dispersion_learner : BaseEstimator
        Template estimator used to model conditional absolute-error scales.
    nexcp : bool
        Whether exponentially decayed calibration weights are enabled.
    decay : float
        Exponential recency-decay factor.
    weighted_refit : bool
        Whether recency weights are also used during model fitting.
    raw_residuals_ : dict
        Signed ``y_hat - y`` residual matrices keyed by model and series.
    oof_scales_ : dict
        Cross-fitted positive scale matrices keyed by model and series.
    ncscores_ : dict
        Standardized ``(y_hat - y) / scale`` matrices keyed by model and series.
    dispersion_learners_ : dict
        Final dispersion estimators fitted on all calibration windows, keyed by
        forecast model column.
    n : int
        Number of calibration trajectories available per horizon step.
    exog_cols_ : list of str
        Exogenous feature columns inferred from the training panel.

    Notes
    -----
    Calibration is horizon-specific: predictions at step ``h`` use only the
    residuals collected at that same step.  The conformal guarantee therefore
    applies marginally to each calibrated horizon under the exchangeability
    assumptions appropriate to the rolling-origin residual trajectories; it is
    not a simultaneous pathwise guarantee over the full forecast trajectory.

    The forecast DataFrame and returned distribution batches are positionally
    aligned after sorting by ``id_col`` and ``time_col``.  Reordering either one
    independently invalidates that correspondence.  Forecasts beyond the fitted
    ``horizon`` are not supported because no matching residual distribution was
    calibrated.
    """

    def __init__(
        self,
        learner: BaseEstimator,
        dispersion_learner: BaseEstimator,
        horizon: int,
        n_windows: int = 10,
        alpha: float = 0.05,
        nexcp: bool = False,
        decay: float = 0.99,
        weighted_refit: bool = True,
        discrete: bool = False,
        minimum: int | None = 0,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ):
        super().__init__(
            learner=learner,
            horizon=horizon,
            n_windows=n_windows,
            nexcp=nexcp,
            decay=decay,
            weighted_refit=weighted_refit,
            alpha=alpha,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
        )
        self.discrete = discrete
        self.minimum = minimum
        self.dispersion_learner = dispersion_learner

    def _scale_features(self, series_ids) -> pd.DataFrame:
        series_ids = list(series_ids)
        return pd.DataFrame(
            {
                "series_id": np.repeat(series_ids, self.horizon),
                "horizon": np.tile(np.arange(1, self.horizon + 1), len(series_ids)),
            }
        )

    def _new_dispersion_pipeline(self) -> Pipeline:
        return Pipeline(
            [
                (
                    "features",
                    ColumnTransformer(
                        [
                            (
                                "series",
                                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                                ["series_id"],
                            ),
                            ("horizon", "passthrough", ["horizon"]),
                        ]
                    ),
                ),
                ("learner", clone(self.dispersion_learner)),
            ]
        )

    def _prepare_scale_calibration(self, scores_by_id):
        """Align one model's residuals as ``(window, series, horizon)``."""
        series_ids = list(scores_by_id)
        features = self._scale_features(series_ids)
        residuals = np.stack([scores_by_id[sid] for sid in series_ids], axis=1)
        return series_ids, features, residuals

    @staticmethod
    def _scale_targets(residuals: np.ndarray) -> np.ndarray:
        """Return positive absolute-error targets for the dispersion learner."""
        return np.maximum(np.abs(residuals).reshape(-1), 1e-6)

    def _fit_oof_scales(
        self,
        residuals: np.ndarray,
        features: pd.DataFrame,
        n_series: int,
    ) -> np.ndarray:
        """Predict each window's scales with a model fitted on all other windows."""
        oof_scales = np.empty_like(residuals, dtype=float)
        for window in range(self.n_windows):
            train_windows = np.delete(np.arange(self.n_windows), window)
            train_residuals = residuals[train_windows]
            train_features = pd.concat(
                [features] * len(train_residuals), ignore_index=True
            )
            scale_model = self._fit_dispersion_pipeline(
                train_features,
                self._scale_targets(train_residuals),
                train_windows,
            )
            oof_scales[window] = np.asarray(
                scale_model.predict(features), dtype=float
            ).reshape(n_series, self.horizon)
        return oof_scales

    @staticmethod
    def _validate_scale_predictions(scales: np.ndarray) -> None:
        """Reject scale predictions that cannot normalize residuals safely."""
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError(
                "Dispersion learner predictions must be positive and finite."
            )

    @staticmethod
    def _matrices_by_series(values: np.ndarray, series_ids) -> dict:
        """Convert a ``(window, series, horizon)`` tensor to keyed matrices."""
        return {
            series_id: values[:, row, :]
            for row, series_id in enumerate(series_ids)
        }

    def _fit_final_dispersion_model(
        self, residuals: np.ndarray, features: pd.DataFrame
    ) -> Pipeline:
        """Refit the scale pipeline on residuals from every backtesting window."""
        final_features = pd.concat(
            [features] * self.n_windows, ignore_index=True
        )
        return self._fit_dispersion_pipeline(
            final_features,
            self._scale_targets(residuals),
            np.arange(self.n_windows),
        )

    def _fit_dispersion_pipeline(self, features, targets, windows) -> Pipeline:
        """Fit dispersion, forwarding recency weights when supported."""
        pipeline = self._new_dispersion_pipeline()
        fit_kwargs = {}
        if (
            self.nexcp
            and self.weighted_refit
            and self._accepts_parameter(self.dispersion_learner.fit, "sample_weight")
        ):
            window_weights = temporal_decay_weights(self.n_windows, self.decay)[windows]
            repeats = len(features) // len(windows)
            fit_kwargs["learner__sample_weight"] = np.repeat(
                window_weights, repeats
            )
        return pipeline.fit(features, targets, **fit_kwargs)

    def _fit_conditional_scales(self) -> None:
        """Cross-fit and apply conditional scales to rolling-origin residuals.

        For each forecast model, raw residuals are aligned in a tensor shaped
        ``(n_windows, n_series, horizon)``. A dispersion pipeline is fitted in a
        leave-one-window-out loop using series identity and horizon as features;
        its held-out predictions form ``oof_scales_``. Dividing raw residuals by
        those scales produces the standardized matrices stored in ``ncscores_``.
        Finally, one dispersion pipeline per forecast model is refitted on every
        window and retained in ``dispersion_learners_`` for future distributions.
        """
        if self.n_windows < 2:
            raise ValueError("TSCPS requires at least two windows for scale cross-fitting.")
        self.raw_residuals_ = self.ncscores_
        if len(self.raw_residuals_) != 1:
            raise ValueError(
                "TSCPS requires a Nixtla learner configured with exactly one model; "
                f"found model columns {list(self.raw_residuals_)}."
            )
        self.dispersion_learners_ = {}
        self.oof_scales_ = {}
        standardized = {}
        for model, scores_by_id in self.raw_residuals_.items():
            series_ids, features, residuals = self._prepare_scale_calibration(
                scores_by_id
            )
            oof_scales = self._fit_oof_scales(
                residuals, features, len(series_ids)
            )
            self._validate_scale_predictions(oof_scales)
            standardized[model] = self._matrices_by_series(
                residuals / oof_scales, series_ids
            )
            self.oof_scales_[model] = self._matrices_by_series(
                oof_scales, series_ids
            )
            self.dispersion_learners_[model] = self._fit_final_dispersion_model(
                residuals, features
            )
        self.ncscores_ = standardized

    def _validate_fit_configuration(self) -> None:
        super()._validate_fit_configuration()
        configured_models = getattr(self.learner, "models", None)
        if isinstance(configured_models, (Mapping, list, tuple)) and len(
            configured_models
        ) != 1:
            raise ValueError(
                "TSCPS requires a Nixtla learner configured with exactly one model."
            )
        if not isinstance(self.discrete, (bool, np.bool_)):
            raise TypeError("discrete must be a boolean.")
        if (
            self.discrete
            and self.minimum is not None
            and not isinstance(self.minimum, (int, np.integer))
        ):
            raise TypeError("minimum must be an integer or None.")

    def fit(self, df, step_size=None, static_features=None, n_jobs=-1):
        if self.discrete:
            self._validate_columns(df)
            target = np.asarray(df[self.target_col], dtype=float)
            if not np.all(np.isfinite(target)) or np.any(target != np.floor(target)):
                raise ValueError(
                    "Discrete time-series CPS targets must be finite integers."
                )
            if self.minimum is not None and np.any(target < self.minimum):
                raise ValueError(
                    f"Discrete time-series CPS targets must be >= {self.minimum}."
                )
        super().fit(
            df,
            step_size=step_size,
            static_features=static_features,
            n_jobs=n_jobs,
        )
        self._fit_conditional_scales()
        return self

    def _prediction_frame(
        self, h: int | None, X_df: pd.DataFrame | None
    ) -> tuple[pd.DataFrame, list[str], int]:
        h = self._get_horizon(h)
        self._check_is_fitted()
        X_df = self._validate_prediction_features(X_df, h)
        pred_df = self._invoke(self.learner.predict, h=h, X_df=X_df)
        if X_df is not None:
            extra_cols = [
                column
                for column in X_df.columns
                if column not in pred_df.columns
            ]
            if extra_cols:
                pred_df = pred_df.merge(
                    X_df[[self.id_col, self.time_col, *extra_cols]],
                    on=[self.id_col, self.time_col],
                    how="left",
                    validate="one_to_one",
                )
        pred_df = pred_df.sort_values(
            [self.id_col, self.time_col]
        ).reset_index(drop=True)
        model_cols = self._infer_model_cols(pred_df)
        n_series = self._validate_prediction_panel(pred_df, h)
        return pred_df, model_cols, n_series

    def _build_distribution(
        self, pred_df: pd.DataFrame, model: str, h: int, n_series: int
    ) -> PredictiveDistribution:
        horizon_steps = np.tile(np.arange(h), n_series)
        weights = (
            temporal_decay_weights(self.n, self.decay) if self.nexcp else None
        )
        if model not in self.ncscores_:
            raise ValueError(
                f"Model column '{model}' was not present during calibration. "
                f"Calibrated model columns: {list(self.ncscores_)}"
            )
        # MSCP stores y_hat - y; predictive distributions use y - y_hat.
        residuals = {
            series_id: -scores[:, :h]
            for series_id, scores in self.ncscores_[model].items()
        }
        locations = pred_df[model].to_numpy(dtype=float)
        series_ids = pred_df[self.id_col].to_numpy()
        scale_features = pd.DataFrame(
            {
                "series_id": series_ids,
                "horizon": horizon_steps + 1,
            }
        )
        scales = np.asarray(
            self.dispersion_learners_[model].predict(scale_features), dtype=float
        )
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("Dispersion learner predictions must be positive and finite.")
        if self.discrete:
            return DiscreteHorizonConformalDistribution(
                locations,
                residuals,
                horizon_steps,
                minimum=self.minimum,
                series_ids=series_ids,
                scales=scales,
                weights=weights,
            )
        return HorizonConformalDistribution(
            locations,
            residuals,
            horizon_steps,
            series_ids=series_ids,
            scales=scales,
            weights=weights,
        )

    @requires_extra("series")
    def predict_distribution(
        self,
        h: int | None = None,
        X_df: pd.DataFrame | None = None,
    ) -> _PanelConformalForecast:
        """Return one predictive forecast aligned to the Nixtla panel grid."""
        h = self._get_horizon(h)
        pred_df, model_cols, n_series = self._prediction_frame(h, X_df)
        if len(model_cols) != 1:
            raise ValueError(
                "TSCPS requires exactly one forecast model column; "
                f"found {model_cols}."
            )
        model = model_cols[0]
        distribution = self._build_distribution(
            pred_df, model, h=h, n_series=n_series
        )
        forecast_type = (
            _DiscretePanelConformalForecast if self.discrete else _PanelConformalForecast
        )
        return forecast_type(
            pred_df, distribution, model, self.id_col, self.time_col
        )

    @property
    def predict_interval(self):
        """Intervals are available from the predictive forecast object."""
        raise AttributeError(
            "TSCPS does not expose predict_interval; call "
            "predict_distribution(...).interval(coverage) instead."
        )

    @requires_extra("series")
    def evaluate(
        self,
        df_test: pd.DataFrame,
        h: int | None = None,
        alpha: float | None = None,
    ) -> pd.DataFrame:
        """Evaluate an interval obtained from the predictive forecast object."""
        alpha = self._get_alpha(alpha)
        forecast = self.predict_distribution(
            h=h, X_df=self._prediction_features(df_test)
        )
        eval_df = self._merge_predictions_with_targets(
            forecast.interval(1.0 - alpha), df_test
        )

        y_true = eval_df[self.target_col].to_numpy()
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
            lower = eval_df[column].to_numpy()
            upper = eval_df[high_column].to_numpy()
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
        return (
            pd.DataFrame(records)
            .sort_values(by=["model", "level"])
            .reset_index(drop=True)
        )

class ContinuousTimeSeriesConformalPredictiveSystem(_TSCPS):
    """Continuous-target CPS for multi-step Nixtla panel forecasts.

    This convenience class configures the internal CPS implementation with
    ``discrete=False``. Predictive forecasts retain their real-valued support;
    CDF, PPF, interval, and sampling operations return panel-aligned DataFrames.

    Parameters
    ----------
    learner : BaseEstimator
        Unfitted Nixtla-compatible forecasting estimator.
    dispersion_learner : BaseEstimator
        Estimator used to cross-fit conditional absolute-error scales.
    horizon : int
        Maximum forecast horizon to calibrate.
    n_windows : int, default=10
        Number of sequential backtesting windows.
    alpha : float, default=0.05
        Default significance level for evaluation.
    nexcp : bool, default=False
        Whether to weight calibration windows by exponential recency decay.
    decay : float, default=0.99
        Decay factor in ``(0, 1)`` used when ``nexcp=True``.
    weighted_refit : bool, default=True
        Whether recency weights are also used while fitting the forecast and
        dispersion learners.
    id_col : str, default="unique_id"
        Series identifier column.
    time_col : str, default="ds"
        Timestamp column.
    target_col : str, default="y"
        Target column.

    Notes
    -----
    A continuous CPS does not define a probability mass function.  Use ``cdf``
    and ``ppf`` on the forecast returned by ``predict_distribution``.
    """

    def __init__(
        self,
        learner: BaseEstimator,
        dispersion_learner: BaseEstimator,
        horizon: int,
        n_windows: int = 10,
        alpha: float = 0.05,
        nexcp: bool = False,
        decay: float = 0.99,
        weighted_refit: bool = True,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ):
        super().__init__(
            learner=learner,
            dispersion_learner=dispersion_learner,
            horizon=horizon,
            n_windows=n_windows,
            alpha=alpha,
            nexcp=nexcp,
            decay=decay,
            weighted_refit=weighted_refit,
            discrete=False,
            minimum=None,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
        )


class DiscreteTimeSeriesConformalPredictiveSystem(_TSCPS):
    """Integer-target CPS for multi-step Nixtla panel forecasts.

    This convenience class validates integer training targets and constructs
    :class:`DiscreteHorizonConformalDistribution` objects.  Their quantiles are
    integer-valued and their PMFs are obtained from adjacent CDF differences.

    Parameters
    ----------
    learner : BaseEstimator
        Unfitted Nixtla-compatible forecasting estimator.
    dispersion_learner : BaseEstimator
        Estimator used to cross-fit conditional absolute-error scales.
    horizon : int
        Maximum forecast horizon to calibrate.
    n_windows : int, default=10
        Number of sequential backtesting windows.
    alpha : float, default=0.05
        Default significance level for evaluation.
    nexcp : bool, default=False
        Whether to weight calibration windows by exponential recency decay.
    decay : float, default=0.99
        Decay factor in ``(0, 1)`` used when ``nexcp=True``.
    weighted_refit : bool, default=True
        Whether recency weights are also used while fitting the forecast and
        dispersion learners.
    minimum : int or None, default=0
        Lower boundary of the target support. Use ``0`` for counts, ``1`` for
        strictly positive outcomes, another integer for a known lower bound,
        or ``None`` when negative integers are valid.
    id_col : str, default="unique_id"
        Series identifier column.
    time_col : str, default="ds"
        Timestamp column.
    target_col : str, default="y"
        Integer target column.

    Notes
    -----
    ``fit`` rejects non-finite, non-integer targets and observations below a
    configured ``minimum``.  The learner's point forecasts may remain real
    valued; discretization is applied when the predictive distribution is
    queried.
    """

    def __init__(
        self,
        learner: BaseEstimator,
        dispersion_learner: BaseEstimator,
        horizon: int,
        n_windows: int = 10,
        alpha: float = 0.05,
        nexcp: bool = False,
        decay: float = 0.99,
        weighted_refit: bool = True,
        minimum: int | None = 0,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ):
        super().__init__(
            learner=learner,
            dispersion_learner=dispersion_learner,
            horizon=horizon,
            n_windows=n_windows,
            alpha=alpha,
            nexcp=nexcp,
            decay=decay,
            weighted_refit=weighted_refit,
            discrete=True,
            minimum=minimum,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
        )
