"""Time-series conformal predictive-system estimators."""

# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from tinyconformal.core.quantiles import temporal_decay_weights
from tinyconformal.distribution.base import PredictiveDistribution
from tinyconformal.utils.imports import requires_extra
from tinyconformal.utils.inspection import call_with_supported_kwargs

from ..residual import ResidualConformalTimeSeriesRegressor
from .calibration import ConditionalScaleCalibrator
from .distribution import (
    DiscreteHorizonConformalDistribution,
    HorizonConformalDistribution,
)
from .forecast import (
    DiscretePanelConformalForecast,
    PanelConformalForecast,
)


class TSCPS(ResidualConformalTimeSeriesRegressor):
    """Conformal predictive system for multi-step panel forecasting.

    The regressor calibrates complete residual distributions for each forecast
    horizon of a Nixtla-compatible learner, such as ``MLForecast`` or
    ``StatsForecast``.  Sequential rolling-origin backtesting produces signed
    residual trajectories. Unlike an interval-only conformal method, CPS keeps
    those empirical distributions and can therefore return CDFs, arbitrary
    quantiles and intervals after a single calibration fit.

    The Nixtla learner must expose exactly one model column. Consequently,
    ``predict_distribution`` returns one self-contained forecast whose methods
    produce pandas DataFrames aligned to the original panel grid.

    Constructor parameters
    ----------
    learner : BaseEstimator
        Unfitted Nixtla-compatible forecasting estimator.  Its ``fit`` method
        must accept a long-format panel and its ``predict`` method must return
        ``id_col``, ``time_col``, and one or more model forecast columns.
    dispersion_learner : BaseEstimator
        Regression estimator for the positive conditional scale. It is
        cross-fitted on absolute rolling-origin errors using series and horizon.
    discrete : bool, default=False
        Whether to construct integer-support predictive distributions.
    minimum : int or None, default=0
        Lower support boundary used when ``discrete=True``. Ignored for
        continuous distributions.

    Fit parameters
    --------------
    horizon : int
        Maximum forecast horizon calibrated during rolling-origin backtesting.
    n_windows : int, default=15
        Number of backtesting windows.  Each series contributes one residual
        trajectory per window.
    nexcp : bool, default=True
        Whether to weight calibration windows by exponential recency decay.
    decay : float, default=0.99
        Decay factor in ``(0, 1)`` used when ``nexcp=True``.
    weighted_refit : bool, default=True
        Whether recency weights are also passed to the forecasting learner and,
        when supported, the dispersion learner during fitting.
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
        discrete: bool = False,
        minimum: int | None = 0,
    ):
        """Configure forecasting, conditional-scale, and support behavior."""
        super().__init__(learner=learner)
        self.discrete = discrete
        self.minimum = minimum
        self.dispersion_learner = dispersion_learner

    def _fit_conditional_scales(self, n_jobs: int = -1) -> None:
        """Cross-fit and apply conditional scales to rolling-origin residuals.

        For each forecast model, raw residuals are aligned in a tensor shaped
        ``(n_windows, n_series, horizon)``. A dispersion pipeline is fitted in a
        leave-one-window-out loop using series identity and horizon as features;
        its held-out predictions form ``oof_scales_``. Dividing raw residuals by
        those scales produces the standardized matrices stored in ``ncscores_``.
        Finally, one dispersion pipeline per forecast model is refitted on every
        window and retained in ``dispersion_learners_`` for future distributions.
        """
        self.raw_residuals_ = self.ncscores_
        self.scale_calibration_ = self._scale_calibrator.fit(
            self.raw_residuals_, n_jobs=n_jobs
        )
        self.ncscores_ = self.scale_calibration_.standardized_residuals
        self.oof_scales_ = self.scale_calibration_.oof_scales
        self.dispersion_learners_ = self.scale_calibration_.pipelines

    def _validate_fit_configuration(self) -> None:
        """Validate CPS-specific learner and discrete-target configuration."""
        super()._validate_fit_configuration()
        configured_models = getattr(self.learner, "models", None)
        if (
            isinstance(configured_models, (Mapping, list, tuple))
            and len(configured_models) != 1
        ):
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

    @requires_extra("series")
    def fit(
        self,
        df,
        horizon: int,
        n_windows: int = 15,
        step_size=None,
        static_features=None,
        nexcp: bool = True,
        decay: float = 0.99,
        weighted_refit: bool = True,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        n_jobs=-1,
    ):
        """Fit TSCPS residual distributions and conditional scales.

        Parameters
        ----------
        df : pandas.DataFrame
            Long-format training panel with identifier, timestamp, target, and
            optional exogenous feature columns.
        horizon : int
            Maximum forecast horizon calibrated in each rolling-origin window.
        n_windows : int, default=15
            Number of rolling-origin windows used to collect residual paths.
        step_size : int or None, default=None
            Distance between consecutive origins. ``None`` uses ``horizon``.
        static_features : list of str or None, default=None
            Time-invariant feature columns passed to the forecasting learner.
        nexcp : bool, default=True
            Give recent calibration windows exponentially larger weights.
        decay : float, default=0.99
            Exponential decay factor in ``(0, 1)`` used when ``nexcp=True``.
        weighted_refit : bool, default=True
            Pass recency weights to compatible forecast and dispersion learner
            fits when ``nexcp=True``.
        id_col : str, default="unique_id"
            Series identifier column.
        time_col : str, default="ds"
            Timestamp column.
        target_col : str, default="y"
            Target column.
        n_jobs : int, default=-1
            Parallel jobs for rolling-origin and conditional-scale calibration.

        Returns
        -------
        self
            Fitted predictive system with standardized residual distributions
            and final forecast and dispersion learners.

        Raises
        ------
        TypeError
            If calibration or discrete-support parameters have invalid types.
        ValueError
            If parameters, columns, panel layout, targets, learner outputs, or
            history length are invalid.
        RuntimeError
            If no residual scores are produced or scale calibration fails.

        Notes
        -----
        Point residuals are collected by rolling-origin backtesting. The scale
        learner is cross-fitted by calibration window, residuals are
        standardized, and both final learners are fitted using all available
        training information. Predictions cannot exceed the fitted horizon.
        """
        self.id_col, self.time_col, self.target_col = id_col, time_col, target_col
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
        self._scale_calibrator = ConditionalScaleCalibrator(
            learner=self.dispersion_learner,
            horizon=horizon,
            n_windows=n_windows,
            nexcp=nexcp,
            decay=decay,
            weighted_refit=weighted_refit,
        )
        self._fit_conditional_scales(n_jobs=n_jobs)
        return self

    def _prediction_frame(
        self, h: int | None, X_df: pd.DataFrame | None
    ) -> tuple[pd.DataFrame, list[str], int]:
        """Predict and validate a sorted future panel for distribution building."""
        h = self._get_horizon(h)
        self._check_is_fitted()
        X_df = self._validate_prediction_features(X_df, h)
        pred_df = call_with_supported_kwargs(self.learner.predict, h=h, X_df=X_df)
        if X_df is not None:
            extra_cols = [
                column for column in X_df.columns if column not in pred_df.columns
            ]
            if extra_cols:
                pred_df = pred_df.merge(
                    X_df[[self.id_col, self.time_col, *extra_cols]],
                    on=[self.id_col, self.time_col],
                    how="left",
                    validate="one_to_one",
                )
        pred_df = pred_df.sort_values([self.id_col, self.time_col]).reset_index(
            drop=True
        )
        model_cols = self._infer_model_cols(pred_df)
        n_series = self._validate_prediction_panel(pred_df, h)
        return pred_df, model_cols, n_series

    def _build_distribution(
        self, pred_df: pd.DataFrame, model: str, h: int, n_series: int
    ) -> PredictiveDistribution:
        """Combine point forecasts, scales, and residuals into a distribution."""
        horizon_steps = np.tile(np.arange(h), n_series)
        weights = temporal_decay_weights(self.n, self.decay) if self.nexcp else None
        if model not in self.ncscores_:
            raise ValueError(
                f"Model column '{model}' was not present during calibration. "
                f"Calibrated model columns: {list(self.ncscores_)}"
            )
        # Convert OOF-standardized scores from (y_hat - y) / scale to
        # the (y - y_hat) / scale orientation used by predictive distributions.
        residuals = {
            series_id: -scores[:, :h]
            for series_id, scores in self.ncscores_[model].items()
        }
        locations = pred_df[model].to_numpy(dtype=float)
        series_ids = pred_df[self.id_col].to_numpy()
        scales = self._scale_calibrator.predict(
            self.dispersion_learners_[model], series_ids, horizon_steps
        )
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
    ) -> PanelConformalForecast:
        """Return predictive distributions aligned to the Nixtla panel grid.

        Parameters
        ----------
        h : int or None, default=None
            Number of future steps to forecast for each series. If ``None``, the
            horizon supplied to ``fit`` is used. ``h``
            cannot exceed the calibrated horizon.
        X_df : pandas.DataFrame or None, default=None
            Future exogenous features in Nixtla long format. It must contain
            ``id_col`` and ``time_col``, one row per series and forecast step,
            and every exogenous column used during fitting. Pass ``None`` when
            the forecasting learner does not require future exogenous features.

        Returns
        -------
        PanelConformalForecast
            Row-aligned predictive forecast sorted by ``id_col`` and
            ``time_col``. :meth:`cdf`, :meth:`sf`, :meth:`ppf`, :meth:`interval`, and
            :meth:`to_frame` return pandas DataFrames on the
            same panel grid. Forecasts from a discrete CPS additionally expose
            :meth:`pmf` and return integer quantiles.

        Raises
        ------
        ValueError
            If ``h`` is outside the calibrated horizon, the future-feature panel
            is incomplete or malformed, the learner returns an invalid panel,
            or its prediction contains other than exactly one model column.

        Notes
        -----
        The result contains one predictive distribution for each series-step
        pair. Its base frame contains ``id_col``, ``time_col``, the learner's
        point-forecast column, and any future-feature columns merged from
        ``X_df``. Rows must not be reordered independently of distributional
        results because calibration is positionally aligned.
        """
        h = self._get_horizon(h)
        pred_df, model_cols, n_series = self._prediction_frame(h, X_df)
        if len(model_cols) != 1:
            raise ValueError(
                f"TSCPS requires exactly one forecast model column; found {model_cols}."
            )
        model = model_cols[0]
        distribution = self._build_distribution(pred_df, model, h=h, n_series=n_series)
        forecast_type = (
            DiscretePanelConformalForecast if self.discrete else PanelConformalForecast
        )
        return forecast_type(pred_df, distribution, model, self.id_col, self.time_col)
