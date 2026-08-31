# Conformal Time Series (`series`)

The `series` submodule provides rolling-origin conformal calibration for
Nixtla-style panel forecasters such as MLForecast and StatsForecast. Training
data use long format with `unique_id`, `ds`, and `y` by default.

## Public models

| Model | Output | Use case |
|---|---|---|
| `ConformalDistributionTimeSeriesRegressor` | MSCP bands | Point forecasters |
| `ConformalQuantileTimeSeriesRegressor` | TSCQR bands | Quantile forecasters |
| `ContinuousTimeSeriesConformalPredictiveSystem` | Complete continuous distributions | Arbitrary quantiles, CDFs, samples and intervals |
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

forecast, distributions = cps.predict_distribution(h=14, X_df=future_exog)
quantiles = cps.predict_quantiles([0.1, 0.5, 0.9], h=14, X_df=future_exog)
intervals = cps.predict_interval(h=14, X_df=future_exog)
```

The discrete system has the same workflow and adds `pmf` to each returned
distribution. Its `minimum` parameter defines the integer support boundary.

Set `nexcp=True` to apply exponential recency weights controlled by `decay`.
When `weighted_refit=True`, compatible forecasting and dispersion learners also
receive those weights during refitting.
