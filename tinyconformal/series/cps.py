# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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

    Attributes
    ----------
    locations : ndarray of shape (n_predictions,)
        Point forecasts defining the location of each predictive distribution.
    residuals : ndarray or dict
        Sorted signed calibration residuals, either pooled or keyed by series.
    horizon_steps : ndarray of shape (n_predictions,)
        Horizon step used to select the residual distribution for each row.

    Notes
    -----
    If ``r = y - y_hat`` denotes a signed calibration residual, the predictive
    distribution for row ``i`` is the empirical distribution of
    ``locations[i] + r[:, horizon_steps[i]]``.  The CDF uses conformal ranks with
    denominator ``n_calibration_trajectories + 1`` and the PPF uses the
    corresponding finite-sample ceiling rule.

    Rows are positional.  Their order must remain identical to the associated
    Nixtla forecast DataFrame returned by ``predict_distribution``.
    """

    def __init__(
        self, locations, residuals, horizon_steps, series_ids=None, scales=None
    ):
        self.locations = self._validate_locations(locations)
        self.horizon_steps = self._validate_horizon_steps(horizon_steps)
        self.residuals, self.series_ids = self._prepare_residuals(residuals, series_ids)
        self.scales = self._validate_scales(scales)
        self._n_calibration, calibrated_horizon = self._residual_shape()
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
        residuals = np.sort(_validate_matrix(residuals, "residuals"), axis=0)
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
            series_id: np.sort(
                _validate_matrix(values, f"residuals[{series_id!r}]"), axis=0
            )
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
        return self.scales[:, None] * residuals


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
    ):
        super().__init__(
            locations,
            residuals,
            horizon_steps,
            series_ids=series_ids,
            scales=scales,
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
        if values.ndim == 1 and values.size == len(self):
            return np.where(below, 0.0, result)
        return np.where(np.broadcast_to(below, np.shape(result)), 0.0, result)


class _TSCPS(ConformalDistributionTimeSeriesRegressor):
    """Conformal predictive system for multi-step panel forecasting.

    The regressor calibrates complete residual distributions for each forecast
    horizon of a Nixtla-compatible learner, such as ``MLForecast`` or
    ``StatsForecast``.  Sequential rolling-origin backtesting produces signed
    residual trajectories. Unlike an interval-only conformal method, CPS keeps
    those empirical distributions and can therefore return CDFs, arbitrary
    quantiles, samples, and intervals after a single calibration fit.

    ``predict_distribution`` returns a dictionary because Nixtla estimators may
    expose more than one model column. Each value is aligned row-for-row with the
    DataFrame returned alongside it.

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
        Default significance level used by ``predict_interval`` and ``evaluate``.
        It does not restrict the quantiles available from the fitted CPS.
    discrete : bool, default=False
        Whether to construct integer-support predictive distributions.
    minimum : int or None, default=0
        Lower support boundary used when ``discrete=True``.  Ignored for
        continuous distributions.
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
    ncscores_ : dict of str to ndarray
        Calibration residual matrices keyed first by model column and then by
        series identifier. Each per-series matrix has shape
        ``(n_windows, horizon)`` and stores ``y_hat - y``; signs are reversed
        when CPS distributions are constructed.
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
            train_residuals = np.delete(residuals, window, axis=0)
            train_features = pd.concat(
                [features] * len(train_residuals), ignore_index=True
            )
            scale_model = self._new_dispersion_pipeline().fit(
                train_features, self._scale_targets(train_residuals)
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
        return self._new_dispersion_pipeline().fit(
            final_features, self._scale_targets(residuals)
        )

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
        pred_df = (
            self._invoke(self.learner.predict, h=h, X_df=X_df)
            .sort_values([self.id_col, self.time_col])
            .reset_index(drop=True)
        )
        model_cols = self._infer_model_cols(pred_df)
        n_series = self._validate_prediction_panel(pred_df, h)
        return pred_df, model_cols, n_series

    def _build_distributions(
        self, pred_df: pd.DataFrame, model_cols: list[str], h: int, n_series: int
    ) -> dict[str, PredictiveDistribution]:
        horizon_steps = np.tile(np.arange(h), n_series)
        distributions = {}
        for model in model_cols:
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
                distributions[model] = DiscreteHorizonConformalDistribution(
                    locations,
                    residuals,
                    horizon_steps,
                    minimum=self.minimum,
                    series_ids=series_ids,
                    scales=scales,
                )
            else:
                distributions[model] = HorizonConformalDistribution(
                    locations,
                    residuals,
                    horizon_steps,
                    series_ids=series_ids,
                    scales=scales,
                )
        return distributions

    @requires_extra("series")
    def predict_distribution(
        self,
        h: int | None = None,
        X_df: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, dict[str, PredictiveDistribution]]:
        """Return the Nixtla forecast frame and aligned CPS distributions."""
        h = self._get_horizon(h)
        pred_df, model_cols, n_series = self._prediction_frame(h, X_df)
        return pred_df, self._build_distributions(
            pred_df, model_cols, h=h, n_series=n_series
        )

    @requires_extra("series")
    def predict_quantiles(
        self,
        quantiles,
        h: int | None = None,
        X_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Return arbitrary CPS quantiles in a Nixtla long-format DataFrame."""
        quantiles = np.asarray(quantiles, dtype=float)
        if quantiles.ndim == 0:
            quantiles = quantiles.reshape(1)
        if quantiles.ndim != 1 or quantiles.size == 0:
            raise ValueError("quantiles must be a non-empty scalar or 1D sequence.")
        if not np.all(np.isfinite(quantiles)) or np.any(
            (quantiles < 0.0) | (quantiles > 1.0)
        ):
            raise ValueError("quantiles must contain finite values in [0, 1].")

        pred_df, distributions = self.predict_distribution(h=h, X_df=X_df)
        for model, distribution in distributions.items():
            quantile_matrix = np.broadcast_to(
                quantiles, (len(distribution), quantiles.size)
            )
            values = distribution.ppf(quantile_matrix)
            for index, quantile in enumerate(quantiles):
                label = np.format_float_positional(
                    quantile * 100.0, precision=12, trim="-"
                )
                pred_df[f"{model}-q-{label}"] = values[:, index]
        return pred_df

    @requires_extra("series")
    def predict_interval(
        self,
        h: int | None = None,
        X_df: pd.DataFrame | None = None,
        alpha: float | None = None,
    ) -> pd.DataFrame:
        """Return equal-tailed CPS intervals in the MSCP column convention."""
        alpha = self._get_alpha(alpha)
        pred_df, distributions = self.predict_distribution(h=h, X_df=X_df)
        level = self._coverage_label(alpha)
        for model, distribution in distributions.items():
            bounds = distribution.interval(1.0 - alpha)
            pred_df[f"{model}-lo-{level}"] = bounds[:, 0]
            pred_df[f"{model}-hi-{level}"] = bounds[:, 1]
        return pred_df


class ContinuousTSCPS(_TSCPS):
    """Continuous-target CPS for multi-step Nixtla panel forecasts.

    This convenience class configures the internal CPS implementation with
    ``discrete=False``. Predictive distributions retain their real-valued
    support and expose CDF, PPF, interval, and sampling operations.

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
        Default significance level for interval prediction and evaluation.
    id_col : str, default="unique_id"
        Series identifier column.
    time_col : str, default="ds"
        Timestamp column.
    target_col : str, default="y"
        Target column.

    Notes
    -----
    A continuous CPS does not define a probability mass function.  Use ``cdf``
    and ``ppf`` on the distributions returned by ``predict_distribution``.
    """

    def __init__(
        self,
        learner: BaseEstimator,
        dispersion_learner: BaseEstimator,
        horizon: int,
        n_windows: int = 10,
        alpha: float = 0.05,
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
            discrete=False,
            minimum=None,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
        )


class DiscreteTSCPS(_TSCPS):
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
        Default significance level for interval prediction and evaluation.
    minimum : int or None, default=0
        Lower boundary of the target support.  Set to ``None`` to allow all
        integers.
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
            discrete=True,
            minimum=minimum,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
        )
