"""Panel-aligned facades over CPS predictive distributions."""

# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

from __future__ import annotations

import numpy as np
import pandas as pd

from tinyconformal.distribution.base import PredictiveDistribution

__all__ = ["DiscretePanelConformalForecast", "PanelConformalForecast"]


class PanelConformalForecast:
    """Panel-aligned facade over one conformal predictive distribution batch."""

    def __init__(self, frame, distribution, model, id_col, time_col):
        self._frame = frame.copy()
        self._distribution = distribution
        self.model = model
        self.id_col = id_col
        self.time_col = time_col

    def __len__(self) -> int:
        return len(self._distribution)

    @property
    def distribution(self) -> PredictiveDistribution:
        """Return the row-aligned predictive distribution."""
        return self._distribution

    def to_frame(self) -> pd.DataFrame:
        """Return an isolated copy of the point-forecast panel."""
        return self._frame.copy()

    @staticmethod
    def _label(value) -> str:
        return np.format_float_positional(float(value), precision=12, trim="-")

    def _apply(self, method: str, inputs, labeler, row_label: str) -> pd.DataFrame:
        inputs_array = np.asarray(inputs)
        values = np.asarray(getattr(self._distribution, method)(inputs))
        result = self.to_frame()
        if values.ndim == 1:
            column = labeler(inputs_array) if inputs_array.ndim == 0 else row_label
            result[column] = values
            return result
        labels = np.ravel(inputs_array)
        common_grid = inputs_array.ndim == 1 and labels.size == values.shape[1]
        for index in range(values.shape[1]):
            column = labeler(labels[index]) if common_grid else f"{row_label}-{index}"
            result[column] = values[:, index]
        return result

    def cdf(self, values) -> pd.DataFrame:
        """Evaluate ``P(Y <= value)`` on the forecast panel."""
        return self._apply(
            "cdf", values, lambda value: f"P(Y<={self._label(value)})", "P(Y<=value)"
        )

    def sf(self, values) -> pd.DataFrame:
        """Evaluate ``P(Y > value)`` on the forecast panel."""
        return self._apply(
            "sf", values, lambda value: f"P(Y>{self._label(value)})", "P(Y>value)"
        )

    def ppf(self, quantiles) -> pd.DataFrame:
        """Evaluate predictive quantiles on the forecast panel."""
        return self._apply(
            "ppf", quantiles, lambda value: f"Q({self._label(value)})", "Q(p)"
        )

    def interval(self, coverage: float = 0.95) -> pd.DataFrame:
        """Return an equal-tailed central predictive interval."""
        bounds = np.asarray(self._distribution.interval(coverage))
        alpha = 1.0 - float(coverage)
        result = self.to_frame()
        result[f"Q({self._label(alpha / 2.0)})"] = bounds[:, 0]
        result[f"Q({self._label(1.0 - alpha / 2.0)})"] = bounds[:, 1]
        return result


class DiscretePanelConformalForecast(PanelConformalForecast):
    """Panel conformal forecast that additionally exposes a PMF."""

    def pmf(self, values) -> pd.DataFrame:
        """Evaluate ``P(Y = value)`` on the forecast panel."""
        inputs = np.asarray(values)
        if inputs.size == 0:
            raise ValueError("pmf values must not be empty.")
        return self._apply(
            "pmf", values, lambda value: f"P(Y={self._label(value)})", "P(Y=value)"
        )
