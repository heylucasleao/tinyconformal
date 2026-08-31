# Conformal Regression (`regressor`)

The `regressor` submodule calibrates finite-sample prediction intervals around
point or quantile regression models.

## Public models

- `ConformalizedRegressor`: split inductive conformal prediction (ICP) using
  absolute residuals and symmetric intervals.
- `ConformalizedQuantileRegressor`: conformalized quantile regression (CQR),
  preserving asymmetric lower and upper quantile forecasts.

## ICP usage

```python
from sklearn.ensemble import RandomForestRegressor
from tinyconformal.regressor import ConformalizedRegressor

learner = RandomForestRegressor(random_state=42).fit(X_train, y_train)
model = ConformalizedRegressor(learner, alpha=0.10)
model.fit(X_calibration, y_calibration)

intervals = model.predict_interval(X_test)
```

## CQR usage

The learner must return lower and upper quantiles from
`predict(X, quantiles=[...])`. `MultiQuantileRegressor` from
`tinyconformal.utils` can adapt estimators that fit only one quantile at a time.

```python
from tinyconformal.regressor import ConformalizedQuantileRegressor

model = ConformalizedQuantileRegressor(quantile_learner, alpha=0.10)
model.fit(X_calibration, y_calibration)
intervals = model.predict_interval(X_test)
```

Both estimators also accept precomputed out-of-sample scores through
`fit_from_scores`. Use `tinyconformal.core.CrossValidationCalibration` when you
want out-of-fold calibration without reserving a separate calibration split.
