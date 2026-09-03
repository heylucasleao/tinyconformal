"""Time-series conformal predictive-system estimators."""

from __future__ import annotations

import re
from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from tinyconformal.core.quantiles import temporal_decay_weights
from tinyconformal.distribution.base import PredictiveDistribution
from tinyconformal.utils.imports import requires_extra

from ..mscp import MultiStepConformalTimeSeriesRegressor
from .dispersion import ConditionalScaleCalibrator
from .distributions import (
    DiscreteHorizonConformalDistribution,
    HorizonConformalDistribution,
)
from .forecast import (
    _DiscretePanelConformalForecast,
    _PanelConformalForecast,
)


class _TSCPS(MultiStepConformalTimeSeriesRegressor):
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
        self._scale_calibrator = ConditionalScaleCalibrator(
            learner=dispersion_learner,
            horizon=horizon,
            n_windows=n_windows,
            nexcp=nexcp,
            decay=decay,
            weighted_refit=weighted_refit,
        )

    def _scale_features(self, series_ids) -> pd.DataFrame:
        """Build the series-and-horizon features used for dispersion modeling."""
        return self._scale_calibrator.features(series_ids)

    def _new_dispersion_pipeline(self) -> Pipeline:
        """Create an unfitted conditional-dispersion pipeline."""
        return self._scale_calibrator.new_pipeline()

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
        (
            self.ncscores_,
            self.oof_scales_,
            self.dispersion_learners_,
        ) = self._scale_calibrator.fit_transform(self.raw_residuals_, n_jobs=n_jobs)

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
        self._fit_conditional_scales(n_jobs=n_jobs)
        return self

    def _prediction_frame(
        self, h: int | None, X_df: pd.DataFrame | None
    ) -> tuple[pd.DataFrame, list[str], int]:
        """Predict and validate a sorted future panel for distribution building."""
        h = self._get_horizon(h)
        self._check_is_fitted()
        X_df = self._validate_prediction_features(X_df, h)
        pred_df = self._invoke(self.learner.predict, h=h, X_df=X_df)
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
    ) -> _PanelConformalForecast:
        """Return predictive distributions aligned to the Nixtla panel grid.

        Parameters
        ----------
        h : int or None, default=None
            Number of future steps to forecast for each series. If ``None``, the
            horizon supplied when constructing the estimator is used. ``h``
            cannot exceed the calibrated horizon.
        X_df : pandas.DataFrame or None, default=None
            Future exogenous features in Nixtla long format. It must contain
            ``id_col`` and ``time_col``, one row per series and forecast step,
            and every exogenous column used during fitting. Pass ``None`` when
            the forecasting learner does not require future exogenous features.

        Returns
        -------
        _PanelConformalForecast
            Row-aligned predictive forecast sorted by ``id_col`` and
            ``time_col``. :meth:`cdf`, :meth:`ppf`, :meth:`interval`, and
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
            _DiscretePanelConformalForecast
            if self.discrete
            else _PanelConformalForecast
        )
        return forecast_type(pred_df, distribution, model, self.id_col, self.time_col)

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
                    "mwis": np.round(self._mwi_score(y_true, lower, upper, alpha), 3),
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

    @requires_extra("series")
    def predict_distribution(
        self,
        h: int | None = None,
        X_df: pd.DataFrame | None = None,
    ) -> _DiscretePanelConformalForecast:
        """Return a discrete predictive forecast on the Nixtla panel grid.

        The returned object exposes :meth:`cdf`, :meth:`ppf`, :meth:`pmf`,
        :meth:`interval`, :meth:`evaluate`, and
        :meth:`to_frame`. See :meth:`_TSCPS.predict_distribution` for the
        complete input, output, and error contract.
        """
        return super().predict_distribution(h=h, X_df=X_df)
