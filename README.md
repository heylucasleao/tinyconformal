# TinyConformal

Related project: [tinyshift](https://github.com/HeyLucasLeao/tinyshift)

TinyConformal is a Python library for conformal prediction in classification and regression.
It provides tools to build valid prediction sets and prediction intervals with a target significance level (`alpha`).

It also provides conformal predictive systems (CPS) that turn fitted point
regressors into complete predictive distributions for continuous or ordered
discrete outcomes.

For more information on a previous project related to Out-of-Bag (OOB) solutions, visit [this link](https://github.com/HeyLucasLeao/cp-study).

## Recent updates
- Added support for exactness-bound-based calibration through `ExactnessBound` for ICP and CQR workflows.
- Added `unlabeled_fit` support for conformal classifiers and regressors, enabling calibration without labeled calibration data when an exactness bound is available.
- Classifiers can now be calibrated from unlabeled data using pseudo-labels derived from model predictions, while regressors can use a pre-estimated exactness bound to build the conformity scores.
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

TinyConformal is organized into two main submodules:

- `tinyconformal.classifier`: conformal classifiers for binary classification.
- `tinyconformal.regressor`: conformal regressors and exactness-bound utilities.
- `tinyconformal.series`: multi-step conformal prediction for time series forecasting.
- `tinyconformal.distribution`: split conformal predictive distributions and CPS wrappers.

### Predictive distribution submodule

Calibrate a fitted regressor with labeled out-of-sample data, then request any
quantile, central interval, CDF value, or random sample:

```python
from tinyconformal.distribution import ContinuousConformalPredictiveSystem

cps = ContinuousConformalPredictiveSystem(fitted_regressor)
cps.fit(X_cal, y_cal)
predictive = cps.predict_distribution(X_test)

median = predictive.ppf(0.5)
intervals = predictive.interval(coverage=0.90)
probabilities = predictive.cdf(values)
```

For ordered integer targets such as demand counts, use
`DiscreteConformalPredictiveSystem`. Its predictive object additionally exposes
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

Standard split CPS calibration assumes exchangeability. For time series, build
the calibration set with rolling-origin predictions and choose a windowing or
weighting scheme appropriate to the temporal dependence and drift.
Forecasting wrappers that do not implement `predict(X)` can use
`fit_from_predictions(y_cal, y_pred_cal)` followed by
`predict_distribution_from_predictions(y_pred_test)`.

For Nixtla-compatible estimators, use the horizon-wise series CPS. It shares the
sequential rolling-origin backtesting machinery and panel contract used by MSCP
and TSCQR:

```python
from tinyconformal.series import (
    ContinuousConformalPredictiveSystemTimeSeriesRegressor,
)

cps = ContinuousConformalPredictiveSystemTimeSeriesRegressor(
    learner=mlforecast_or_statsforecast,
    horizon=14,
    n_windows=5,
)
cps.fit(train_df, step_size=14, static_features=["store_type"])

forecast_df, distributions = cps.predict_distribution(h=14, X_df=future_exog)
quantile_df = cps.predict_quantiles([0.1, 0.5, 0.9], h=14, X_df=future_exog)
interval_df = cps.predict_interval(h=14, X_df=future_exog, alpha=0.1)
```

The returned distribution dictionary is keyed by the Nixtla model column, and
each distribution is aligned row-for-row with `forecast_df`. Use
`DiscreteConformalPredictiveSystemTimeSeriesRegressor` for ordered integer/count
targets; those distributions additionally provide `pmf`.

### Distributional conformal prediction (DCP)

For heteroscedastic learners that predict a conditional quantile grid, DCP
calibrates probability integral transform (PIT) ranks rather than additive
residuals:

```python
from tinyconformal.distribution import DistributionalConformalPredictiveSystem

dcp = DistributionalConformalPredictiveSystem(
    fitted_quantile_learner,
    quantiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
)
dcp.fit(X_cal, y_cal)
predictive = dcp.predict_distribution(X_test)
```

Native conditional-distribution models are supported without an adapter-specific
dependency by passing batches that implement `cdf`, `ppf`, and `len` to
`fit_from_distribution` and `predict_distribution_from_base`.

For Nixtla-compatible forecasting estimators, configure the quantile columns and
calibrate PITs separately for each forecast horizon:

```python
from tinyconformal.series import (
    DistributionalConformalPredictiveSystemTimeSeriesRegressor,
)

dcp = DistributionalConformalPredictiveSystemTimeSeriesRegressor(
    learner=quantile_forecaster,
    horizon=14,
    n_windows=5,
    quantile_columns={
        0.10: "LGBM-q-10",
        0.50: "LGBM-q-50",
        0.90: "LGBM-q-90",
    },
)
dcp.fit(train_df, step_size=14)
forecast_df, distributions = dcp.predict_distribution(h=14)
```

For ordered integer/count series, use
`DiscreteDistributionalConformalPredictiveSystemTimeSeriesRegressor`. It uses
randomized PIT calibration, returns integer quantiles, and exposes `pmf` on the
returned distributions. Quantile crossings are repaired before interpolation.
The implementation has no dependency on `tinyshift` or its two-stage Negative
Binomial model.

Runnable distribution examples are organized in `examples/distribution/`:

- `cps_continuous.ipynb`
- `cps_discrete.ipynb`
- `dcp_continuous.ipynb`
- `dcp_discrete.ipynb`

They compare scikit-learn and LightGBM learners, add quantile-forest to DCP, and
cover CDF, PMF where applicable, PPF, arbitrary quantiles, empirical coverage,
and Newsvendor optimization.

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
from tinyconformal.regressor import ExactnessBound
```

###  Time Series submodule
Import from `tinyconformal.series`:

```python
from tinyconformal.series import ConformalDistributionTimeSeriesRegressor
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

### Unlabeled calibration example

For settings where labeled calibration data are unavailable, you can fit the conformal model directly on unlabeled data:

```python
from sklearn.ensemble import RandomForestClassifier
from tinyconformal.classifier import BinaryMarginalConformalClassifier
from sklearn.model_selection import cross_val_score

learner = RandomForestClassifier(n_estimators=100, oob_score=True)
learner.fit(X_train, y_train)

score = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
beta = round(np.mean(1 - score), 3)

conformal_classifier = BinaryMarginalConformalClassifier(learner)
conformal_classifier.unlabeled_fit(X_unlabeled, beta)

predictions = conformal_classifier.predict(X_test)
```

For regressors, you can combine an exactness bound estimate with unlabeled calibration:

```python
from sklearn.ensemble import RandomForestRegressor
from tinyconformal.regressor import ConformalizedRegressor, ExactnessBound

learner = RandomForestRegressor(random_state=42)
tilde_beta, beta = ExactnessBound.estimate_icp_bound(
    learner, X_train, y_train, p=0.95, cv=5
)

# Fit learner before using conformal regressor
learner.fit(X_train, y_train)

regressor = ConformalizedRegressor(learner, alpha=0.05)
regressor.unlabeled_fit(X_unlabeled, tilde_beta=tilde_beta, beta=beta)

intervals = regressor.predict_interval(X_test)
```

### Evaluating the Classifier

Evaluate the performance of the conformal classifier using the `evaluate` method:

```python
results = conformal_classifier.evaluate(X_test, y_test)
print(results)
```

### Time Series Example

For point-forecast models, `ConformalDistributionTimeSeriesRegressor` extracts signed empirical residuals ($R = \hat{y} - y$) across backtesting windows to build horizon-specific prediction intervals for Nixtla-style learners (MLForecast or StatsForecast):

```python
from lightgbm import LGBMRegressor
from mlforecast import MLForecast
from tinyconformal.series import ConformalDistributionTimeSeriesRegressor

# Wrap a base forecaster
mlf = MLForecast(
    models=[LGBMRegressor(random_state=42)],
    freq="D",
    lags=[1, 7],
)

conformal_ts = ConformalDistributionTimeSeriesRegressor(
    learner=mlf,
    horizon=7,
    n_windows=5,
    alpha=0.10,
)

conformal_ts.fit(df, step_size=7)  # Window displacement stride
intervals_df = conformal_ts.predict_interval(h=7)
```

For models outputting lower and upper quantile forecasts, `ConformalQuantileTimeSeriesRegressor` computes CQR nonconformity scores $E = \max(\hat{q}_{\text{low}} - y, y - \hat{q}_{\text{high}})$ to produce calibrated prediction intervals:

```python
from lightgbm import LGBMRegressor
from mlforecast import MLForecast
from tinyconformal.series import ConformalQuantileTimeSeriesRegressor

# Forecaster configured to output quantile columns
mlf = MLForecast(
    models={
        "LGBM-lo-90": LGBMRegressor(objective="quantile", alpha=0.05),
        "LGBM-hi-90": LGBMRegressor(objective="quantile", alpha=0.95),
    },
    freq="D",
    lags=[1, 7],
)

conformal_cqr_ts = ConformalQuantileTimeSeriesRegressor(
    learner=mlf,
    horizon=7,
    n_windows=5,
    intervals=("LGBM-lo-90", "LGBM-hi-90"),
)

conformal_cqr_ts.fit(df, step_size=7)
intervals_df = conformal_cqr_ts.predict_interval(h=7)
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

### BinaryMarginalConformalClassifier

`BinaryMarginalConformalClassifier` is a marginal-coverage conformal classifier that uses a classifier as the underlying learner.

- Training via labeled calibration: `fit(X, y)`
- Training via OOB calibration: `fit(y=y_train, oob=True)`
- Training via unlabeled calibration: `unlabeled_fit(X, beta=...)`

### BinaryClassConditionalConformalClassifier

`BinaryClassConditionalConformalClassifier` is a class-conditional conformal classifier that uses a classifier as the underlying learner.

- Training via labeled calibration: `fit(X, y)`
- Training via OOB calibration: `fit(y=y_train, oob=True)`
- Training via unlabeled calibration: `unlabeled_fit(X, beta=...)` using pseudo-labels derived from the model probabilities

### ConformalizedRegressor

`ConformalizedRegressor` is a conformal regressor built on a regression learner.

- Training via labeled calibration: `fit(X, y)`
- Training via OOB calibration: `fit(X, y, oob=True)`
- Training via unlabeled calibration: `unlabeled_fit(X, tilde_beta=..., beta=...)` using an exactness bound

### ConformalizedQuantileRegressor

`ConformalizedQuantileRegressor` is a conformal quantile regressor built on a quantile regressor.

- Training via labeled calibration: `fit(X, y)`
- Training via OOB calibration: `fit(X, y, oob=True)`
- Training via unlabeled calibration: `unlabeled_fit(X, tilde_beta=..., beta=...)` using an exactness bound

### ConformalDistributionTimeSeriesRegressor
`ConformalDistributionTimeSeriesRegressor` is a multi-step time series conformal regressor compatible with Nixtla interface estimators (`MLForecast` or `StatsForecast`).

Training & Residual extraction via backtesting: `fit(df, step_size=...)`

The significance level is configured globally with `alpha` and may be overridden
when predicting: `predict_interval(h=..., alpha=...)`.

Multi-step interval forecasting: `predict_interval(h=..., alpha=...)`

### ConformalQuantileTimeSeriesRegressor
`ConformalQuantileTimeSeriesRegressor` is a multi-step time series conformal quantile regressor (CQR) compatible with Nixtla interface estimators that output quantile predictions.

Training & Nonconformity score calculation via backtesting:` fit(df, step_size=...)`

The significance level is inferred independently for each interval from its
coverage suffix (`-90` means `alpha=0.10`, `-50` means `alpha=0.50`).

Multi-step interval forecasting: `predict_interval(h=...)`

### ExactnessBound

`ExactnessBound` provides helper methods to estimate the exactness bound used in unlabeled conformal calibration for ICP and CQR workflows.

## License

This project is licensed under the MIT License.
