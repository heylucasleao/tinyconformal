"""TSCQR example: CQR scores kept separate for each unique_id."""

import numpy as np
import pandas as pd

from tinyconformal.series import ConformalQuantileTimeSeriesRegressor


class LastValueQuantileForecaster:
    """Deterministic Nixtla-compatible quantile forecaster for demonstration."""

    def fit(self, df, id_col="unique_id", time_col="ds", target_col="y"):
        ordered = df.sort_values([id_col, time_col])
        self.id_col = id_col
        self.time_col = time_col
        self.last_ = ordered.groupby(id_col, sort=True).tail(1).set_index(id_col)
        return self

    def predict(self, h, X_df=None):
        records = []
        for series_id, row in self.last_.iterrows():
            dates = pd.date_range(row[self.time_col], periods=h + 1, freq="D")[1:]
            records.extend(
                {
                    self.id_col: series_id,
                    self.time_col: date,
                    "LastValue-lo-50": row["y"] - 0.5,
                    "LastValue-hi-50": row["y"] + 0.5,
                }
                for date in dates
            )
        return pd.DataFrame(records)


dates = pd.date_range("2026-01-01", periods=20, freq="D")
signal = np.arange(20, dtype=float) ** 2
panel = pd.DataFrame(
    {
        "unique_id": np.repeat(["stable", "volatile"], len(dates)),
        "ds": list(dates) * 2,
        "y": np.r_[signal, signal * 10],
    }
)

pair = ("LastValue-lo-50", "LastValue-hi-50")
tscqr = ConformalQuantileTimeSeriesRegressor(
    LastValueQuantileForecaster(),
    horizon=2,
    n_windows=4,
    intervals=pair,
).fit(panel, n_jobs=1)

score_key = ":".join(pair)
assert set(tscqr.ncscores_[score_key]) == {"stable", "volatile"}
assert tscqr.ncscores_[score_key]["stable"].shape == (4, 2)

forecast = tscqr.predict_interval(h=2)
forecast["width"] = (
    forecast["LastValue-hi-50-cqr"] - forecast["LastValue-lo-50-cqr"]
)
print(forecast[["unique_id", "ds", "width"]])

# The correction is estimated from each series' own backtesting errors.
widths = forecast.groupby("unique_id")["width"].mean()
assert widths["volatile"] > widths["stable"]
