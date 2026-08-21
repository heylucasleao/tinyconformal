# TinyConformal

Related project: [tinyshift](https://github.com/HeyLucasLeao/tinyshift)

TinyConformal is a Python library for conformal prediction in classification and regression.
It provides tools to build valid prediction sets and prediction intervals with a target significance level (`alpha`).

For more information on a previous project related to Out-of-Bag (OOB) solutions, visit [this link](https://github.com/HeyLucasLeao/cp-study).

## Recent updates
- Added support for exactness-bound-based calibration through `ExactnessBound` for ICP and CQR workflows.
- Added `unlabeled_fit` support for conformal classifiers and regressors, enabling calibration without labeled calibration data when an exactness bound is available.
- Classifiers can now be calibrated from unlabeled data using pseudo-labels derived from model predictions, while regressors can use a pre-estimated exactness bound to build the conformity scores.
- Added `tinyconformal.series` support with `ConformalDistributionTimeSeriesRegressor` for multi-step time series interval forecasting with customizable backtesting strides (`step_size`).

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

For time series, ConformalDistributionTimeSeriesRegressor extracts signed empirical residuals ($R = \hat{y} - y$) across backtesting windows to build horizon-specific prediction intervals for Nixtla-style learners (MLForecast or StatsForecast):

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
    step_size=7,  # Window displacement stride
    alpha=0.10,
)

conformal_ts.fit(df)
intervals_df = conformal_ts.predict_interval(h=7)
```
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

Multi-step interval forecasting: `predict_interval(h=..., alpha=...)`

ExactnessBound

### ExactnessBound

`ExactnessBound` provides helper methods to estimate the exactness bound used in unlabeled conformal calibration for ICP and CQR workflows.

## License

This project is licensed under the MIT License.
