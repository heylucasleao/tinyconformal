"""Time-series conformal predictive-system estimators."""

from __future__ import annotations


import pandas as pd
from sklearn.base import BaseEstimator
from tinyconformal.utils.imports import requires_extra

from .forecast import (
    _DiscretePanelConformalForecast,
)
from .base import TSCPS


class ContinuousTimeSeriesConformalPredictiveSystem(TSCPS):
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


class DiscreteTimeSeriesConformalPredictiveSystem(TSCPS):
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
