"""MSCP example: horizon-wise calibration kept separate for each unique_id."""

import numpy as np
import pandas as pd

from tinyconformal.series import ConformalDistributionTimeSeriesRegressor


class LastValueForecaster:
    """Small Nixtla-compatible forecaster used to make the example reproducible."""

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
                    "LastValue": row["y"],
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

mscp = ConformalDistributionTimeSeriesRegressor(
    LastValueForecaster(), horizon=2, n_windows=4, alpha=0.5
).fit(panel, n_jobs=1)

assert set(mscp.ncscores_["LastValue"]) == {"stable", "volatile"}
assert mscp.ncscores_["LastValue"]["stable"].shape == (4, 2)

forecast = mscp.predict_interval(h=2)
forecast["width"] = forecast["LastValue-hi-50"] - forecast["LastValue-lo-50"]
print(forecast[["unique_id", "ds", "LastValue", "width"]])

# The volatile series has larger local residuals and therefore wider intervals.
widths = forecast.groupby("unique_id")["width"].mean()
assert widths["volatile"] > widths["stable"]
