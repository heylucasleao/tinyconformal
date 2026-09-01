# Conformal Time Series (`series`)

The `series` submodule provides rolling-origin conformal calibration for
Nixtla-style panel forecasters such as MLForecast and StatsForecast. Training
data use long format with `unique_id`, `ds`, and `y` by default.

## Public models

| Model | Output | Use case |
|---|---|---|
| `ConformalDistributionTimeSeriesRegressor` | MSCP bands | Point forecasters |
| `ConformalQuantileTimeSeriesRegressor` | TSCQR bands | Quantile forecasters |
| `ContinuousTimeSeriesConformalPredictiveSystem` | Complete continuous distributions | Arbitrary quantiles, CDFs and intervals |
| `DiscreteTimeSeriesConformalPredictiveSystem` | Complete integer distributions | Counts, PMFs and inventory decisions |

## MSCP bands

```python
from tinyconformal.series import ConformalDistributionTimeSeriesRegressor

model = ConformalDistributionTimeSeriesRegressor(
    learner=nixtla_point_forecaster,
    horizon=14,
    n_windows=5,
    alpha=0.10,
)
model.fit(train_df, step_size=14)
intervals = model.predict_interval(h=14, X_df=future_exog)
```

## TSCQR bands

```python
from tinyconformal.series import ConformalQuantileTimeSeriesRegressor

model = ConformalQuantileTimeSeriesRegressor(
    learner=nixtla_quantile_forecaster,
    horizon=14,
    n_windows=5,
    intervals=("model-lo-90", "model-hi-90"),
)
model.fit(train_df, step_size=14)
intervals = model.predict_interval(h=14, X_df=future_exog)
```

## Complete predictive distributions

```python
from sklearn.ensemble import RandomForestRegressor
from tinyconformal.series import ContinuousTimeSeriesConformalPredictiveSystem

cps = ContinuousTimeSeriesConformalPredictiveSystem(
    learner=nixtla_point_forecaster,
    dispersion_learner=RandomForestRegressor(min_samples_leaf=5),
    horizon=14,
    n_windows=5,
).fit(train_df, step_size=14)

forecast = cps.predict_distribution(h=14, X_df=future_exog)
median = forecast.ppf(0.5)
probabilities = forecast.cdf(values)
quantiles = forecast.ppf([0.1, 0.5, 0.9])
intervals = forecast.interval(coverage=0.95)
```

TSCPS accepts a Nixtla learner configured with exactly one forecast model. Its
`cdf`, `ppf`, and `interval` methods return long pandas DataFrames on
the original panel grid. The discrete system has the same workflow and adds
`pmf`; its `minimum` parameter defines the integer support boundary.

Set `nexcp=True` to apply exponential recency weights controlled by `decay`.
When `weighted_refit=True`, compatible forecasting and dispersion learners also
receive those weights during refitting.

## CPS retraining flow

Calling `fit` performs calibration and final refitting in one operation. Each
rolling-origin window fits a temporary clone of the forecasting learner on the
history available at that origin and forecasts the next `horizon` steps. These
window models are used only to collect out-of-sample residuals; they are not
retained for future prediction.

The dispersion learner is cross-fitted by leaving out one complete calibration
window at a time. Its held-out scale estimates standardize the residuals without
using the same window for fitting and scoring. After calibration, one final
dispersion model is fitted on all calibration windows, and the forecasting
learner is fitted once on the complete training panel.

```text
fit(train_df)
|
+-- rolling-origin calibration
|   |
|   +-- window 1: fit forecaster clone -> forecast -> residuals
|   +-- window 2: fit forecaster clone -> forecast -> residuals
|   +-- ...
|   +-- window N: fit forecaster clone -> forecast -> residuals
|
+-- conditional scale calibration
|   |
|   +-- leave one window out -> held-out scales
|   +-- standardize residuals by series and horizon
|   +-- refit final dispersion model on all windows
|
+-- refit forecasting learner on the complete training panel
|
`-- fitted CPS
    |
    `-- predict_distribution(h, X_df)
        +-- point forecast from the final forecasting learner
        +-- scale from the final dispersion model
        `-- calibrated distribution from stored standardized residuals
```

`predict_distribution` does not refit either learner. When new observations
must become part of the training or calibration data, call `fit` again with the
updated panel; this reruns the complete flow above. With `nexcp=True`, recency
weights affect the stored calibration distribution. With both `nexcp=True` and
`weighted_refit=True`, compatible learners also receive recency weights in the
calibration-window fits and the final fits.

## Internal implementation

The CPS implementation is divided into model orchestration, conditional-scale
calibration, predictive distributions, and panel-aligned forecast adapters. See
[`cps/README.md`](cps/README.md) for the module boundaries, invariants, and
extension rules.
