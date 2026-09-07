"""Evaluation utilities for time-series conformal predictive forecasts."""

from __future__ import annotations

import pandas as pd

from .forecast import PanelConformalForecast


class TimeSeriesCPSEvaluator:
    """Evaluate a fitted panel forecast independently of estimator prediction."""

    @classmethod
    def evaluate(
        cls,
        forecast: PanelConformalForecast,
        y,
        coverages=(0.5, 0.8, 0.9, 0.95),
    ) -> pd.DataFrame:
        """Return coverage, width, and Winkler metrics by central coverage."""
        if not isinstance(forecast, PanelConformalForecast):
            raise TypeError(
                "forecast must be a PanelConformalForecast returned by "
                "predict_distribution()."
            )
        return forecast.distribution.evaluate(y, coverages=coverages)
