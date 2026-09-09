# Conformal Time Series (`series`)

The `series` submodule provides rolling-origin conformal calibration for
Nixtla-style panel forecasters such as MLForecast and StatsForecast. Training
data use long format with `unique_id`, `ds`, and `y` by default.

## Public models

| Model | Output | Use case |
|---|---|---|
| `MultiStepConformalTimeSeriesRegressor` | MSCP bands | Point forecasters |
| `ConformalizedQuantileTimeSeriesRegressor` | TSCQR bands | Quantile forecasters |
| `ContinuousTimeSeriesConformalPredictiveSystem` | Complete continuous distributions | Arbitrary quantiles, CDFs and intervals |
| `DiscreteTimeSeriesConformalPredictiveSystem` | Complete integer distributions | Counts, PMFs and inventory decisions |

## MSCP bands

```python
from tinyconformal.series import MultiStepConformalTimeSeriesRegressor

model = MultiStepConformalTimeSeriesRegressor(
    learner=nixtla_point_forecaster,
    alpha=0.10,
)
model.fit(train_df, horizon=14, n_windows=15, step_size=14)
intervals = model.predict_interval(h=14, X_df=future_exog)
```

## TSCQR bands

```python
from tinyconformal.series import ConformalizedQuantileTimeSeriesRegressor

model = ConformalizedQuantileTimeSeriesRegressor(
    learner=nixtla_quantile_forecaster,
    intervals=("model-lo-90", "model-hi-90"),
)
model.fit(train_df, horizon=14, n_windows=15, step_size=14)
intervals = model.predict_interval(h=14, X_df=future_exog)
```

## Complete predictive distributions

```python
from sklearn.ensemble import RandomForestRegressor
from tinyconformal.series import ContinuousTimeSeriesConformalPredictiveSystem

cps = ContinuousTimeSeriesConformalPredictiveSystem(
    learner=nixtla_point_forecaster,
    dispersion_learner=RandomForestRegressor(min_samples_leaf=5),
).fit(train_df, horizon=14, n_windows=15, step_size=14)

forecast = cps.predict_distribution(h=14, X_df=future_exog)
median = forecast.ppf(0.5)
probabilities = forecast.cdf(values)
exceedance = forecast.sf(values)
quantiles = forecast.ppf([0.1, 0.5, 0.9])
intervals = forecast.interval(coverage=0.95)
```

TSCPS accepts a Nixtla learner configured with exactly one forecast model. Its
`cdf`, `sf`, `ppf`, and `interval` methods return long pandas DataFrames on
the original panel grid. The discrete system has the same workflow and adds
`pmf`; its `minimum` parameter defines the integer support boundary. Results use
mathematical column names such as `Q(0.9)`, `P(Y<=5)`, `P(Y>5)`, and `P(Y=5)`.

By default, `nexcp=False` gives every calibration window equal weight. Set
`nexcp=True` to apply exponential recency weights controlled by `decay`. The
`weighted_refit` option remains enabled by default, but only has an effect when
`nexcp=True`; compatible forecasting and dispersion learners then also receive
those weights during refitting.

## Calibration rank and number of windows

The default `n_windows=15` is a practical compromise between quantile
resolution, computation, and temporal relevance. For TSCQR, the finite-sample
conformal rank is

```text
ceil((n_windows + 1) * (1 - alpha)).
```

To keep this rank within the observed calibration scores, the theoretical
minimum is `n_windows >= 1 / alpha - 1`:

| Coverage | `alpha` | Mathematical minimum |
|---:|---:|---:|
| 80% | 0.20 | 4 |
| 90% | 0.10 | 9 |
| 95% | 0.05 | 19 |
| 99% | 0.01 | 99 |

Thus, the default supports a 90% TSCQR interval without rank clipping, but a
95% interval needs at least 19 windows. More windows generally improve quantile
resolution, while older windows may be less representative under temporal
drift. With `nexcp=True`, recency weighting reduces the influence of those
older windows.

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

The CPS implementation is divided into estimator orchestration (`base.py`),
public convenience wrappers (`wrapper.py`), conditional-scale calibration
(`calibration.py`), predictive distributions (`distribution.py`), and
panel-aligned forecast adapters (`forecast.py`). See
[`cps/README.md`](cps/README.md) for the module boundaries, invariants, and
extension rules.
