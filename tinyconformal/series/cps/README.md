# Time-Series Conformal Predictive Systems (`series.cps`)

The `series.cps` package implements complete predictive distributions for
Nixtla-style panel forecasters. It is an internal package behind
`ContinuousTimeSeriesConformalPredictiveSystem` and
`DiscreteTimeSeriesConformalPredictiveSystem`; users should normally import
those models from `tinyconformal.series`.

## Public models

| Model | Support | Available functionals |
|---|---|---|
| `ContinuousTimeSeriesConformalPredictiveSystem` | Real-valued | CDF, quantiles and central intervals |
| `DiscreteTimeSeriesConformalPredictiveSystem` | Integer-valued | CDF, PMF, integer quantiles and central intervals |

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
interval = forecast.interval(coverage=0.9)
probabilities = forecast.cdf(values)
```

The returned forecast owns both the point-forecast panel and its calibrated
distribution. Its methods return pandas DataFrames aligned with the panel rows.
Discrete forecasts additionally expose `pmf`.

## Internal modules

| Module | Responsibility |
|---|---|
| `model.py` | Estimator lifecycle, panel prediction, distribution construction and evaluation |
| `dispersion.py` | Cross-fitting conditional scales and fitting the final dispersion model |
| `distributions.py` | Horizon- and series-specific empirical predictive distributions |
| `forecast.py` | DataFrame facade that keeps distribution results aligned with the forecast panel |
| `__init__.py` | Package exports and compatibility imports |

Dependencies flow toward the smaller components: `model.py` coordinates the
other modules, while `dispersion.py`, `distributions.py`, and `forecast.py` do
not import the model. This direction avoids circular imports and keeps the
statistical objects independent from the estimator lifecycle.

## Calibration flow

Calling `fit` first delegates rolling-origin residual collection to the MSCP
implementation. `ConditionalScaleCalibrator` then cross-fits the dispersion
learner, leaving out one complete calibration window at a time. The resulting
out-of-fold scales standardize residuals by series and forecast horizon. A final
dispersion model is fitted on all windows for future scale predictions.

```text
rolling-origin forecasts
        |
        v
raw signed residuals by model, series and horizon
        |
        v
leave-one-window-out dispersion cross-fitting
        |
        +--> out-of-fold scales
        +--> standardized residual distributions
        `--> final dispersion model
                    |
                    v
point forecast + future scale + stored residuals
                    |
                    v
panel-aligned predictive forecast
```

With `nexcp=True`, calibration windows receive exponential recency weights.
When `weighted_refit=True`, compatible forecasting and dispersion learners also
receive recency weights during fitting.

## Distribution semantics

Calibration matrices have shape `(n_windows, horizon)` and are stored
separately for each series identifier. A prediction at horizon step `h` uses
only residuals collected at that same step for the same series.

`HorizonConformalDistribution` combines each point forecast with its selected
standardized residuals and predicted conditional scale. Its continuous support
provides `cdf`, `ppf`, and `interval`. The discrete specialization rounds
quantiles upward, enforces the configured `minimum`, and calculates
`pmf(k) = cdf(k) - cdf(k - 1)`.

These distribution classes remain implementation details and are intentionally
excluded from the package `__all__`. They are re-exported from
`tinyconformal.series.cps` to preserve existing internal imports and tests.

## Alignment and extension rules

- Keep forecast rows sorted by identifier and timestamp. Distribution rows are
  positional and must not be reordered independently.
- Preserve one calibration matrix per series; do not pool residuals across
  identifiers implicitly.
- Reject horizons beyond the fitted value because no corresponding residual
  distribution was calibrated.
- Validate conditional scales as positive and finite before standardizing or
  constructing a predictive distribution.
- Put distribution mathematics in `distributions.py`, panel formatting in
  `forecast.py`, scale calibration in `dispersion.py`, and orchestration in
  `model.py`.

Tests for this package live in `tests/test_series_cps.py`. Changes should also
run `tests/test_mscp.py` because the CPS estimator inherits the rolling-origin
calibration workflow from MSCP.
