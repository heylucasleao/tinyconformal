"""Time-series conformal predictive-system estimators."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator

from tinyconformal.utils.imports import requires_extra

from .base import TSCPS
from .forecast import DiscretePanelConformalForecast


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

    Notes
    -----
    A continuous CPS does not define a probability mass function.  Use ``cdf``
    and ``ppf`` on the forecast returned by ``predict_distribution``.
    """

    def __init__(
        self,
        learner: BaseEstimator,
        dispersion_learner: BaseEstimator,
    ):
        """Configure a continuous time-series conformal predictive system."""
        super().__init__(
            learner=learner,
            dispersion_learner=dispersion_learner,
            discrete=False,
            minimum=None,
        )


class DiscreteTimeSeriesConformalPredictiveSystem(TSCPS):
    """Integer-target CPS for multi-step Nixtla panel forecasts.

    This convenience class constructs panel-aligned integer predictive
    distributions. Their quantiles are integer-valued and their PMFs are
    obtained from adjacent CDF differences.

    Parameters
    ----------
    learner : BaseEstimator
        Unfitted Nixtla-compatible forecasting estimator.
    dispersion_learner : BaseEstimator
        Estimator used to cross-fit conditional absolute-error scales.
    minimum : int or None, default=0
        Lower boundary of the target support. Use ``0`` for counts, ``1`` for
        strictly positive outcomes, another integer for a known lower bound,
        or ``None`` when negative integers are valid.

    The learner's point forecasts may remain real valued; discretization is
    applied when the predictive distribution is queried.
    """

    def __init__(
        self,
        learner: BaseEstimator,
        dispersion_learner: BaseEstimator,
        minimum: int | None = 0,
    ):
        """Configure an integer-support time-series conformal predictive system."""
        super().__init__(
            learner=learner,
            dispersion_learner=dispersion_learner,
            discrete=True,
            minimum=minimum,
        )

    @requires_extra("series")
    def predict_distribution(
        self,
        h: int | None = None,
        X_df: pd.DataFrame | None = None,
    ) -> DiscretePanelConformalForecast:
        """Return a discrete predictive forecast on the Nixtla panel grid.

        The returned object exposes :meth:`cdf`, :meth:`ppf`, :meth:`pmf`,
        :meth:`interval`, :meth:`evaluate`, and
        :meth:`to_frame`. See :meth:`_TSCPS.predict_distribution` for the
        complete input, output, and error contract.
        """
        return super().predict_distribution(h=h, X_df=X_df)
