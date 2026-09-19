"""Evaluation of already-produced panel prediction intervals."""

from __future__ import annotations

import re
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .cps import CPSEvaluator
from .regressor import RegressorEvaluator

_LOWER_BOUND = re.compile(
    r"^(?P<model>.+)-lo-(?P<level>[0-9]+(?:\.[0-9]+)?)(?P<suffix>.*)$"
)


class PanelEvaluator:
    """Evaluate interval and distribution forecasts on a time-series panel."""

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

    @staticmethod
    def _align_targets(
        y_true: pd.DataFrame,
        forecast_frame: pd.DataFrame,
        id_col: str,
        time_col: str,
        target_col: str,
    ) -> pd.DataFrame:
        """Align targets to forecast order using panel keys."""
        if not isinstance(y_true, pd.DataFrame):
            raise TypeError("y_true must be a pandas DataFrame.")
        keys = [id_col, time_col]
        required_true = [*keys, target_col]
        for frame, required, name in (
            (y_true, required_true, "y_true"),
            (forecast_frame, keys, "forecast"),
        ):
            missing = [column for column in required if column not in frame.columns]
            if missing:
                raise KeyError(f"Columns not found in {name}: {missing}")
            if frame.duplicated(keys).any():
                raise ValueError(f"{name} contains duplicate identifier/time rows.")
        aligned = forecast_frame.merge(
            y_true[required_true], on=keys, how="left", validate="one_to_one"
        )
        if aligned[target_col].isna().any():
            raise ValueError("y_true must contain a target for every forecast row.")
        try:
            observed = aligned[target_col].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("y_true must contain numeric target values.") from exc
        if not np.all(np.isfinite(observed)):
            raise ValueError("y_true must contain only finite target values.")
        return aligned

    @classmethod
    def evaluate_interval(
        cls,
        y_true: pd.DataFrame,
        forecast,
        intervals: Mapping | None = None,
        coverages=(0.5, 0.8, 0.9, 0.95),
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> pd.DataFrame:
        """Evaluate precomputed or distribution-derived panel intervals.

        DataFrame columns following ``<model>-lo-<coverage>`` and
        ``<model>-hi-<coverage>`` are inferred automatically, including pairs
        with suffixes such as ``-cqr``. A panel CPS forecast instead derives
        equal-tailed intervals for every requested coverage.
        """
        forecast_frame = forecast if isinstance(forecast, pd.DataFrame) else forecast.to_frame()
        aligned = cls._align_targets(
            y_true, forecast_frame, id_col, time_col, target_col
        )

        if not isinstance(forecast, pd.DataFrame):
            observed = aligned[target_col].to_numpy(dtype=float)
            if len(forecast.distribution) != len(observed):
                raise ValueError(
                    "y_true and the predictive distribution must have equal length."
                )
            records = []
            for coverage in coverages:
                bounds = forecast.distribution.interval(coverage)
                metrics = RegressorEvaluator.evaluate(
                    observed, bounds, coverage
                ).iloc[0]
                records.append({"model": forecast.model, **metrics.to_dict()})
            return pd.DataFrame.from_records(records)

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

    @classmethod
    def evaluate_distribution(
        cls,
        y_true: pd.DataFrame,
        forecast,
        train_df: pd.DataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> pd.DataFrame:
        """Return per-series CRPS normalized by each training-target scale."""
        if not hasattr(forecast, "distribution") or not hasattr(forecast, "to_frame"):
            raise TypeError("forecast must be a panel predictive forecast.")
        if not isinstance(train_df, pd.DataFrame):
            raise TypeError("train_df must be a pandas DataFrame.")
        missing = [
            column for column in (id_col, target_col) if column not in train_df.columns
        ]
        if missing:
            raise KeyError(f"Columns not found in train_df: {missing}")

        aligned = cls._align_targets(
            y_true, forecast.to_frame(), id_col, time_col, target_col
        )
        observed = aligned[target_col].to_numpy(dtype=float)
        if len(forecast.distribution) != len(observed):
            raise ValueError(
                "y_true and the predictive distribution must have equal length."
            )
        try:
            train_targets = train_df[target_col].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("train_df must contain numeric target values.") from exc
        scale_frame = pd.DataFrame(
            {id_col: train_df[id_col].to_numpy(), target_col: train_targets}
        )
        train_scales = scale_frame.groupby(id_col, observed=True)[target_col].std()
        row_scores = pd.DataFrame(
            {
                id_col: aligned[id_col].to_numpy(),
                "crps": CPSEvaluator._crps(forecast.distribution, observed),
            }
        )
        result = (
            row_scores.groupby(id_col, observed=True, sort=False)["crps"]
            .agg(crps="mean", n_obs="size")
            .reset_index()
        )
        result["target_std"] = result[id_col].map(train_scales)
        result["ncrps"] = [
            CPSEvaluator._ncrps(crps, scale)
            if np.isfinite(scale) and scale > 0.0
            else np.nan
            for crps, scale in zip(result["crps"], result["target_std"])
        ]
        return result[[id_col, "crps", "target_std", "ncrps", "n_obs"]]
