# TinyConformal

Related project: [tinyshift](https://github.com/HeyLucasLeao/tinyshift)

TinyConformal is a Python library for conformal prediction in classification and regression.
It provides tools to build valid prediction sets and prediction intervals with a target significance level (`alpha`).

It also provides conformal predictive systems (CPS) that turn fitted point
regressors into complete predictive distributions for continuous or ordered
discrete outcomes.

For more information on a previous project related to Out-of-Bag (OOB) solutions, visit [this link](https://github.com/HeyLucasLeao/cp-study).

## Recent updates
- Added out-of-fold regression calibration through `CrossValidationCalibration`.
- Added `fit_from_scores` for ICP/CQR and standardized residual calibration for CPS.
- Classifiers and regressors can now reuse out-of-fold cross-validation outputs for conformal calibration without reserving a separate calibration split.
- Added `tinyconformal.series` support with `ConformalDistributionTimeSeriesRegressor` and `ConformalQuantileTimeSeriesRegressor` for multi-step time series interval forecasting with customizable backtesting strides (`step_size`).
- Added support for Conformalized Quantile Regression (CQR) on multi-step time series using base estimators producing quantile forecasts.

Previously, `calibrate` used `Balanced Accuracy Score`; it can now also be calibrated with `Matthews Correlation Coefficient` or `Bookmaker Informedness Score` for improved reliability. The `evaluate` method also reports `bm` and `mcc`.

Currently, TinyConformal supports Out-of-Bag (OOB) solutions for `RandomForestClassifier` in binary classification problems, as well as `RandomForestRegressor` and `RandomForestQuantileRegressor` for regression tasks. For additional options and advanced features, you may want to explore [Crepes](https://github.com/henrikbostrom/crepes).

## Installation

### Using pip

```bash
pip install tinyconformal
```

Optional extras:

```bash
pip install "tinyconformal[plot]"
pip install "tinyconformal[notebook]"
pip install "tinyconformal[dev]"
```

### Using uv

Install in the current environment:

```bash
uv pip install tinyconformal
```

Add as a dependency in a project:

```bash
uv add tinyconformal
```

Optional extras with uv:

```bash
uv pip install "tinyconformal[plot]"
uv pip install "tinyconformal[notebook]"
uv pip install "tinyconformal[dev]"
```

## Submodules and usage

TinyConformal is organized into six public submodules:

- [`tinyconformal.classifier`](tinyconformal/classifier/README.md): conformal
  prediction sets for binary classification.
- [`tinyconformal.regressor`](tinyconformal/regressor/README.md): ICP and CQR
  prediction intervals.
- [`tinyconformal.distribution`](tinyconformal/distribution/README.md):
  cross-fitted conformal predictive distributions.
- [`tinyconformal.series`](tinyconformal/series/README.md): MSCP, TSCQR, and
  complete predictive systems for time series.
- [`tinyconformal.core`](tinyconformal/core/README.md): out-of-fold calibration
  primitives.
- [`tinyconformal.utils`](tinyconformal/utils/README.md): multi-quantile modeling
  and Newsvendor decision utilities.

### Predictive distribution submodule

Cross-fit location and scale estimators on the training data, then request any
quantile, central interval, CDF value, or random sample:

```python
from tinyconformal.distribution import ContinuousCrossConformalPredictiveSystem

cps = ContinuousCrossConformalPredictiveSystem(
    learner=location_regressor,
    dispersion_learner=scale_regressor,
    cv=5,
)
cps.fit(X_train, y_train)
predictive = cps.predict_distribution(X_test)

median = predictive.ppf(0.5)
intervals = predictive.interval(coverage=0.90)
probabilities = predictive.cdf(values)
```

The dispersion learner must return a strictly positive conditional scale
estimate directly (not a variance). CPS internally generates OOF location
residuals, cross-fits the scale learner on their absolute values, standardizes
the residuals, and finally refits both models on all training rows.

For ordered integer targets such as demand counts, use
`DiscreteCrossConformalPredictiveSystem`. Its predictive object additionally exposes
`pmf(values)` and returns integer quantiles. For nominal, unordered labels, use
the classifiers in `tinyconformal.classifier` instead.

The complete distribution can be passed directly to the Newsvendor solver:

```python
from tinyconformal.utils import NewsvendorSolver

result = NewsvendorSolver.optimize_distribution(
    forecast_frame,
    predictive,
    underage_cost="shortage_cost",
    overage_cost="holding_cost",
)
```

For a discrete predictive distribution, the solver can also report the expected
net benefit of adding each inventory unit. The calculation uses the conformal
CDF directly and accepts either ``max_k`` or an explicit unit grid:

```python
marginal_benefit = NewsvendorSolver.marginal_benefit_distribution(
    forecast_frame,
    predictive,
    underage_cost="shortage_cost",
    overage_cost="holding_cost",
    units=[0, 5, 10, 15],
)
```

The tabular CPS supports cross-fitting only; it does not implement split
calibration or CV+. For time series, use the horizon-wise series CPS below.

For Nixtla-compatible estimators, use the horizon-wise series CPS. It shares the
sequential rolling-origin backtesting machinery and panel contract used by MSCP
and TSCQR:

```python
from sklearn.ensemble import RandomForestRegressor

from tinyconformal.series import (
    ContinuousTimeSeriesConformalPredictiveSystem,
)

cps = ContinuousTimeSeriesConformalPredictiveSystem(
    learner=mlforecast_or_statsforecast,
    dispersion_learner=RandomForestRegressor(min_samples_leaf=5),
    horizon=14,
    n_windows=5,
    nexcp=True,
    decay=0.99,
    weighted_refit=True,
)
cps.fit(train_df, step_size=14, static_features=["store_type"])

forecast_df, distributions = cps.predict_distribution(h=14, X_df=future_exog)
quantile_df = cps.predict_quantiles([0.1, 0.5, 0.9], h=14, X_df=future_exog)
interval_df = cps.predict_interval(h=14, X_df=future_exog, alpha=0.1)
```

The scale estimator is cross-fitted on absolute rolling-origin errors using
series identity and forecast horizon as conditional features. The returned
distribution dictionary is keyed by the Nixtla model column, and
each distribution is aligned row-for-row with `forecast_df`. Use
`DiscreteTimeSeriesConformalPredictiveSystem` for ordered integer/count
targets; those distributions additionally provide `pmf`.

MSCP, TSCQR, and TSCPS share the optional NexCP-style temporal weighting
contract. With `nexcp=False` (the default), calibration windows have equal
weight. With `nexcp=True`, weights decay exponentially from the newest window
using `decay=0.99`, the value used in the NexCP paper experiments. This weights
calibration scores and, when `weighted_refit=True`, adds an internal recency
weight column to every rolling-origin fit and to the final learner refit. A
learner without `weight_col` support raises an explicit error. TSCPS also passes
window weights to dispersion estimators that support `sample_weight`.

Runnable distribution examples are organized in `examples/distribution/`:

- `cps_continuous.ipynb`
- `cps_discrete.ipynb`

They cover CDF, PMF where applicable, PPF, arbitrary quantiles, empirical
coverage, and Newsvendor optimization.

The complete OOF workflow for ICP, CQR, CPS, and both binary classification
strategies is available in `examples/calibration/cross_validation.ipynb`.

### Classifier submodule

Import from `tinyconformal.classifier`:

```python
from tinyconformal.classifier import BinaryMarginalConformalClassifier
from tinyconformal.classifier import BinaryClassConditionalConformalClassifier
```

### Regressor submodule

Import from `tinyconformal.regressor`:

```python
from tinyconformal.regressor import ConformalizedRegressor
from tinyconformal.regressor import ConformalizedQuantileRegressor
from tinyconformal.core.calibration import CrossValidationCalibration
```

### Time series submodule

Use the cross-fitted, horizon-wise predictive systems exported by
`tinyconformal.series`:

```python
from tinyconformal.series import (
    ConformalDistributionTimeSeriesRegressor,
    ConformalQuantileTimeSeriesRegressor,
    ContinuousTimeSeriesConformalPredictiveSystem,
    DiscreteTimeSeriesConformalPredictiveSystem,
)
```

### Example

Example usage of `BinaryClassConditionalConformalClassifier`:

```python
from sklearn.ensemble import RandomForestClassifier
from tinyconformal.classifier import BinaryClassConditionalConformalClassifier

# Create and fit a RandomForestClassifier
learner = RandomForestClassifier(n_estimators=100, oob_score=True)
X_train, y_train = ...  # your training data
learner.fit(X_train, y_train)

# Create and fit the conformal classifier
conformal_classifier = BinaryClassConditionalConformalClassifier(learner)
conformal_classifier.fit(y=y_train, oob=True)

# Make predictions
X_test = ...  # your test data
predictions = conformal_classifier.predict(X_test)
```

### Cross-validation calibration example

Use out-of-fold probabilities to calibrate a classifier without reserving a
separate calibration split:

```python
from sklearn.ensemble import RandomForestClassifier
from tinyconformal.core.calibration import CrossValidationCalibration
from tinyconformal.classifier import BinaryMarginalConformalClassifier

learner = RandomForestClassifier(n_estimators=100, oob_score=True)
probabilities = CrossValidationCalibration.classification_probabilities(
    learner, X_train, y_train, cv=5
)
learner.fit(X_train, y_train)

conformal_classifier = BinaryMarginalConformalClassifier(learner)
conformal_classifier.fit_from_probabilities(probabilities, y_train)

predictions = conformal_classifier.predict(X_test)
```

For regressors, generate out-of-fold scores and then fit the final learner on all
available training data:

```python
from sklearn.ensemble import RandomForestRegressor
from tinyconformal.core.calibration import CrossValidationCalibration
from tinyconformal.regressor import ConformalizedRegressor

learner = RandomForestRegressor(random_state=42)
scores = CrossValidationCalibration.icp_scores(
    learner, X_train, y_train, cv=5
)

# Fit learner before using conformal regressor
learner.fit(X_train, y_train)

regressor = ConformalizedRegressor(learner, alpha=0.05)
regressor.fit_from_scores(scores)

intervals = regressor.predict_interval(X_test)
```

### Evaluating the Classifier

Evaluate the performance of the conformal classifier using the `evaluate` method:

```python
results = conformal_classifier.evaluate(X_test, y_test)
print(results)
```

### Time Series Example

`ContinuousTimeSeriesConformalPredictiveSystem` cross-fits a location forecaster
and a conditional-scale model over rolling-origin windows. It returns complete,
horizon-specific predictive distributions for Nixtla-style learners
(MLForecast or StatsForecast):

```python
from lightgbm import LGBMRegressor
from mlforecast import MLForecast
from tinyconformal.series import ContinuousTimeSeriesConformalPredictiveSystem

# Wrap a base forecaster
mlf = MLForecast(
    models=[LGBMRegressor(random_state=42)],
    freq="D",
    lags=[1, 7],
)

conformal_ts = ContinuousTimeSeriesConformalPredictiveSystem(
    learner=mlf,
    dispersion_learner=LGBMRegressor(random_state=42),
    horizon=7,
    n_windows=5,
    alpha=0.10,
)

conformal_ts.fit(df, step_size=7)
forecast_df, distributions = conformal_ts.predict_distribution(h=7)
intervals_df = conformal_ts.predict_interval(h=7)
```

For ordered integer targets such as demand, use the discrete cross-fitted system.
Its `minimum` is the lower support bound: keep `0` for counts, use `1` for
strictly positive quantities, or `None` if negative integers are valid:

```python
from lightgbm import LGBMRegressor
from mlforecast import MLForecast
from tinyconformal.series import DiscreteTimeSeriesConformalPredictiveSystem

mlf = MLForecast(
    models=[LGBMRegressor(random_state=42)],
    freq="D",
    lags=[1, 7],
)

conformal_count_ts = DiscreteTimeSeriesConformalPredictiveSystem(
    learner=mlf,
    dispersion_learner=LGBMRegressor(random_state=42),
    horizon=7,
    n_windows=5,
    minimum=0,
)

conformal_count_ts.fit(df, step_size=7)
forecast_df, distributions = conformal_count_ts.predict_distribution(h=7)
```

#### Future features and evaluation data

Columns passed through `static_features` belong to each series and are supplied to
the learner only during fitting. All other non-structural columns in the training
data are treated as dynamic exogenous features and must be available for future
timestamps through `X_df`:

```python
conformal_ts.fit(
    train_df,
    static_features=["region"],
)
intervals_df = conformal_ts.predict_interval(
    h=7,
    X_df=future_df[["unique_id", "ds", "temperature"]],
)
```

An explicit `X_df` must contain the identifier, time, and every dynamic exogenous
column used during fitting. It must also contain exactly `h` unique timestamps per
series, using the same timestamp grid for every series. The prediction horizon must
be positive and cannot exceed the `horizon` used for calibration.

`evaluate(df_test, h=...)` uses dynamic features from `df_test` and requires exactly
one non-missing target for every predicted identifier/timestamp pair. Duplicate or
missing targets raise an error instead of being silently omitted from the metrics.

MSCP supports fractional coverage levels. For example, `alpha=0.055` produces
columns such as `Model-lo-94.5` and `Model-hi-94.5`.

Finite-sample conformal correction uses discrete order statistics. When the
requested coverage cannot be attained with the available calibration sample, a
`RuntimeWarning` is emitted and the rank is clipped to the observed score range.
Increasing the number of calibration trajectories, usually through more windows or
series, permits more extreme coverage levels.

### Time Series Mechanics: Horizon vs. Step Size

When calibrating over time series, nonconformity scores are extracted by performing sequential backtesting across multiple calibration windows. The calibration movement is controlled by two parameters:
- `horizon` ($H$): The forecast horizon step count generated in each window.
- `step_size` ($S$): The stride length used to advance the origin between backtesting windows.

Below are three typical backtesting movement patterns assuming a forecast horizon ($H = 4$):

#### Small `step_size` ($S = 1 < H$) — Overlapping Windows
The calibration origin advances by 1 step at a time. This creates significant overlap between consecutive forecast windows, maximizing sample size ($n$) for short historical series.

```plaintext
Time Axis:      | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10|
------------------------------------------------------------------
Window 1:       [=== Initial Train ===]  [--- H=4 (t5 to t8) ---]
Window 2:       [==== Train + 1 ====]    [--- H=4 (t6 to t9) ---]   (Shifted S=1)
Window 3:       [===== Train + 2 =====]    [--- H=4 (t7 to t10) --] (Shifted S=1)
```

#### Default `step_size` ($S = H = 4$) — Disjoint Windows
The calibration origin shifts by the full forecast horizon ($S = H$). Each window starts exactly where the previous forecast ended, eliminating overlap and ensuring independence among calibration residuals.

```
Time Axis:      | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10| t11| t12|
----------------------------------------------------------------------------
Window 1:       [=== Initial Train ===]  [--- H=4 (t5 to t8) ---]
Window 2:       [======= Expanded Train =======] [--- H=4 (t9 to t12) --] (Shifted S=4)
```

#### Large step_size ($S = 6 > H$) — Windows with Gaps
The stride between windows exceeds the forecast horizon ($S > H$). This introduces temporal gaps between evaluation windows, mimicking real-world systems with infrequent retraining schedules.
```plaintext
Time Axis:      | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10| t11| t12| t13| t14|
----------------------------------------------------------------------------------------
Window 1:       [=== Initial Train ===]  [--- H=4 (t5 to t8) ---]
               |                      |                          |
               |<- Evaluated Train -->| <-- Gap (t9, t10) -----> | (Shifted S=6)
               |                      |                          v
Window 2:       [============ Expanded Train ============] [--- H=4 (t11 to t14) --]
```


## Classes

### Regression

Import these classes from `tinyconformal.regressor`:

- `ConformalizedRegressor`: conformalizes a fitted point regressor and produces
  prediction intervals. It supports split, OOB, and precomputed OOF-score
  calibration.
- `ConformalizedQuantileRegressor`: implements conformalized quantile regression
  (CQR) for learners that produce lower and upper quantile predictions.
- `CrossValidationCalibration`: generates OOF ICP/CQR scores and cross-fitted
  CPS location residuals, scales, and standardized residuals.

### Classification

Import these classes from `tinyconformal.classifier`:

- `BinaryMarginalConformalClassifier`: constructs binary prediction sets with
  marginal coverage.
- `BinaryClassConditionalConformalClassifier`: constructs binary prediction sets
  with coverage calibrated separately for each class.

Both classifiers support split calibration with `fit(X, y)`, OOB calibration
with `fit(y=y_train, oob=True)`, and precomputed OOF calibration with
`fit_from_probabilities(probabilities, y)`.

### Distribution

Import these classes from `tinyconformal.distribution`:

- `ContinuousCrossConformalPredictiveSystem`: cross-fits location and scale
  regressors and produces continuous predictive distributions.
- `DiscreteCrossConformalPredictiveSystem`: produces conformal predictive
  distributions for ordered integer or count targets.

### Time Series Distribution

Import these classes from `tinyconformal.series`:

- `ConformalDistributionTimeSeriesRegressor`: produces MSCP prediction bands
  from horizon-specific signed residuals.
- `ConformalQuantileTimeSeriesRegressor`: produces TSCQR prediction bands from
  horizon-specific conformalized quantile scores.
- `ContinuousTimeSeriesConformalPredictiveSystem`: produces a complete
  continuous predictive distribution for every series and forecast horizon.
- `DiscreteTimeSeriesConformalPredictiveSystem`: produces complete
  predictive distributions for ordered integer or count time-series targets and
  supports PMF evaluation.
The time-series CPS classes use rolling-origin calibration with
`fit(df, step_size=...)`. They expose `predict_distribution`, `predict_quantiles`,
and `predict_interval` for multi-step forecasts.

## License

This project is licensed under the MIT License.
