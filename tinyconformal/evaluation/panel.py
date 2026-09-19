"""Evaluation of already-produced panel prediction intervals."""

from __future__ import annotations

import re
from collections.abc import Mapping

import pandas as pd

from .regressor import RegressorEvaluator

_LOWER_BOUND = re.compile(
    r"^(?P<model>.+)-lo-(?P<level>[0-9]+(?:\.[0-9]+)?)(?P<suffix>.*)$"
)


class PanelEvaluator:
    """Align and evaluate interval forecasts on an identifier/time panel."""

    @staticmethod
    def _infer_intervals(forecast: pd.DataFrame) -> list[tuple[str, tuple]]:
        intervals = []
        for lower_col in forecast.columns:
            match = _LOWER_BOUND.match(str(lower_col))
            if match is None:
                continue
            upper_col = (
                f"{match.group('model')}-hi-{match.group('level')}"
                f"{match.group('suffix')}"
            )
            if upper_col not in forecast.columns:
                continue
            name = f"{match.group('model')}{match.group('suffix')}"
            coverage = float(match.group("level")) / 100.0
            intervals.append((name, (lower_col, upper_col, coverage)))
        if not intervals:
            raise ValueError(
                "No interval pairs were inferred from forecast columns. Pass "
                "intervals={name: (lower_column, upper_column, coverage)}."
            )
        return intervals

    @classmethod
    def evaluate(
        cls,
        y_true: pd.DataFrame,
        forecast: pd.DataFrame,
        intervals: Mapping | None = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> pd.DataFrame:
        """Evaluate panel intervals after key-based target alignment.

        Interval columns following ``<model>-lo-<coverage>`` and
        ``<model>-hi-<coverage>`` are inferred automatically, including pairs
        with suffixes such as ``-cqr``. Use ``intervals`` for other naming
        conventions.
        """
        if not isinstance(y_true, pd.DataFrame) or not isinstance(
            forecast, pd.DataFrame
        ):
            raise TypeError("y_true and forecast must be pandas DataFrames.")

        keys = [id_col, time_col]
        required_true = [*keys, target_col]
        for frame, required, name in (
            (y_true, required_true, "y_true"),
            (forecast, keys, "forecast"),
        ):
            missing = [column for column in required if column not in frame.columns]
            if missing:
                raise KeyError(f"Columns not found in {name}: {missing}")
            if frame.duplicated(keys).any():
                raise ValueError(f"{name} contains duplicate identifier/time rows.")

        aligned = forecast.merge(
            y_true[required_true], on=keys, how="left", validate="one_to_one"
        )
        if aligned[target_col].isna().any():
            raise ValueError("y_true must contain a target for every forecast row.")

        specifications = (
            cls._infer_intervals(forecast)
            if intervals is None
            else list(dict(intervals).items())
        )
        records = []
        for name, specification in specifications:
            if len(specification) != 3:
                raise ValueError(
                    "Each interval specification must be "
                    "(lower_column, upper_column, coverage)."
                )
            lower_col, upper_col, coverage = specification
            missing = [
                column
                for column in (lower_col, upper_col)
                if column not in aligned.columns
            ]
            if missing:
                raise KeyError(f"Interval columns not found in forecast: {missing}")
            metrics = RegressorEvaluator.evaluate(
                aligned[target_col],
                aligned[[lower_col, upper_col]].to_numpy(),
                coverage,
            ).iloc[0]
            records.append({"model": name, **metrics.to_dict()})
        return pd.DataFrame.from_records(records)
