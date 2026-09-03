"""Panel-aligned facades over CPS predictive distributions."""

# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

from __future__ import annotations

import numpy as np
import pandas as pd


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
        """Return an isolated frame to which distribution outputs can be added."""
        return self._frame.copy()

    @staticmethod
    def _label(value) -> str:
        """Format a numeric value as a stable column-name component."""
        return np.format_float_positional(float(value), precision=12, trim="-")

    def _single_output_column(self, inputs, prefix, label_transform) -> str:
        """Name a one-dimensional distribution result."""
        if inputs.ndim != 0:
            return f"{self.model}-{prefix}"
        label = inputs if label_transform is None else label_transform(inputs)
        return f"{self.model}-{prefix}-{self._label(label)}"

    def _matrix_output_labels(self, inputs, n_columns, label_transform):
        """Resolve labels for the columns of a matrix result."""
        labels = np.ravel(inputs)
        if labels.size != n_columns:
            return np.arange(n_columns)
        return labels if label_transform is None else label_transform(labels)

    def _apply(
        self, method: str, inputs, prefix: str, label_transform=None
    ) -> pd.DataFrame:
        """Evaluate a distribution method and append its output to the panel."""
        inputs_array = np.asarray(inputs)
        values = np.asarray(getattr(self._distribution, method)(inputs))
        result = self._output_frame()
        if values.ndim == 1:
            column = self._single_output_column(inputs_array, prefix, label_transform)
            result[column] = values
            return result
        labels = self._matrix_output_labels(
            inputs_array, values.shape[1], label_transform
        )
        for index, label in enumerate(labels):
            result[f"{self.model}-{prefix}-{self._label(label)}"] = values[:, index]
        return result

    def cdf(self, values) -> pd.DataFrame:
        """Evaluate the cumulative distribution function on the forecast panel.

        Parameters
        ----------
        values : float or array-like
            Target values at which to evaluate each predictive CDF. A scalar is
            applied to every forecast row. A one-dimensional array defines a
            common evaluation grid for every row. A two-dimensional array with
            ``len(self)`` rows is evaluated row-wise.

        Returns
        -------
        pandas.DataFrame
            Forecast panel sorted by ``id_col`` and ``time_col``, including the
            original point-forecast columns and the evaluated probabilities. A
            scalar produces ``<model>-cdf-<value>``; a common grid produces one
            such column per value. Row-wise input produces one output column per
            input column, numbered when no common value labels are available.

        Raises
        ------
        ValueError
            If a value is non-finite or the input shape is not a scalar, a
            one-dimensional grid, or a matrix with ``len(self)`` rows.

        Notes
        -----
        Output rows remain positionally aligned with the rows returned by
        :meth:`to_frame`. CDF values lie in ``[0, 1]``.
        """
        return self._apply("cdf", values, "cdf")

    def ppf(self, quantiles) -> pd.DataFrame:
        """Evaluate predictive quantiles on the forecast panel.

        Parameters
        ----------
        quantiles : float or array-like
            Probabilities in ``[0, 1]``. A scalar is applied to every forecast
            row. A one-dimensional array defines common quantile levels for
            every row. A two-dimensional array with ``len(self)`` rows specifies
            row-wise quantile levels.

        Returns
        -------
        pandas.DataFrame
            Forecast panel sorted by ``id_col`` and ``time_col``, including the
            original point-forecast columns and the requested quantiles. Scalar
            and common-grid columns are named ``<model>-q-<percentage>``; for
            example, quantile ``0.9`` produces ``<model>-q-90``. Row-wise input
            produces one output column per input column, numbered when no common
            quantile labels are available.

        Raises
        ------
        ValueError
            If any quantile lies outside ``[0, 1]`` or the input shape is not a
            scalar, a one-dimensional grid, or a matrix with ``len(self)`` rows.

        Notes
        -----
        Output rows remain positionally aligned with the rows returned by
        :meth:`to_frame`. Discrete CPS forecasts return integer quantiles.
        """
        return self._apply("ppf", quantiles, "q", label_transform=lambda q: 100.0 * q)

    def interval(self, coverage: float = 0.95) -> pd.DataFrame:
        """Return a central interval on the forecast panel grid."""
        bounds = np.asarray(self._distribution.interval(coverage))
        level = self._label(100.0 * float(coverage))
        result = self._output_frame()
        result[f"{self.model}-lo-{level}"] = bounds[:, 0]
        result[f"{self.model}-hi-{level}"] = bounds[:, 1]
        return result

    def evaluate(self, y, coverages=(0.5, 0.8, 0.9, 0.95)) -> pd.DataFrame:
        """Evaluate the underlying predictive distribution."""
        return self._distribution.evaluate(y, coverages=coverages)


class _DiscretePanelConformalForecast(_PanelConformalForecast):
    """Panel-aligned facade that additionally exposes integer probability mass."""

    def pmf(self, values) -> pd.DataFrame:
        """Evaluate probability masses on a discrete forecast panel.

        Parameters
        ----------
        values : int or array-like of int
            Integer support values at which to evaluate each predictive PMF. A
            scalar is applied to every forecast row. A one-dimensional array
            defines a common support grid for every row. A two-dimensional array
            with ``len(self)`` rows is evaluated row-wise.

        Returns
        -------
        pandas.DataFrame
            Forecast panel sorted by ``id_col`` and ``time_col``, including the
            original point-forecast columns and the probability masses. A scalar
            produces ``<model>-pmf-<value>``; a common grid produces one such
            column per value. Row-wise input produces one output column per input
            column, numbered when no common value labels are available.

        Raises
        ------
        ValueError
            If a value is non-finite or non-integer, or if the input shape is not
            a scalar, a one-dimensional grid, or a matrix with ``len(self)``
            rows.

        Notes
        -----
        This method is available only on forecasts returned by a discrete CPS.
        Each mass is computed as ``CDF(k) - CDF(k - 1)``.
        """
        return self._apply("pmf", values, "pmf")
