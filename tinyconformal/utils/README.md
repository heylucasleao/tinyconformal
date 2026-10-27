# Modeling Utilities (`utils`)

The `utils` submodule contains public helpers that connect conformal models to
quantile estimators.

## Public utilities

- `MultiQuantileRegressor`: fits one clone of a mono-quantile estimator for each
  requested quantile and exposes a joint prediction interface.

## Multi-quantile regression

```python
from sklearn.ensemble import GradientBoostingRegressor
from tinyconformal.utils import MultiQuantileRegressor

learner = MultiQuantileRegressor(
    GradientBoostingRegressor(loss="quantile", random_state=42),
    quantiles=(0.05, 0.5, 0.95),
).fit(X_train, y_train)

quantile_predictions = learner.predict(X_test)
```
