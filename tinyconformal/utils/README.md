# Modeling and Decision Utilities (`utils`)

The `utils` submodule contains public helpers that connect conformal models to
quantile estimators and operational decisions.

## Public utilities

- `MultiQuantileRegressor`: fits one clone of a mono-quantile estimator for each
  requested quantile and exposes a joint prediction interface.
- `NewsvendorSolver`: converts interval or predictive-distribution forecasts
  into cost-sensitive inventory decisions.

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

## Newsvendor optimization

```python
from tinyconformal.utils import NewsvendorSolver

result = NewsvendorSolver.optimize_distribution(
    time_series_predictive_forecast,
    underage_cost="shortage_cost",
    overage_cost="holding_cost",
)
```

For tabular CPS, pass the row-aligned DataFrame and NumPy-backed distribution as
the first two arguments. For TSCPS, pass its self-contained panel forecast as
the only forecast argument, as above.

For discrete predictive distributions, `pmf_distribution` evaluates unit
probabilities and `marginal_benefit_distribution` calculates whether each
additional inventory unit has positive expected value. The distribution rows
must remain in the same order as the corresponding forecast DataFrame.
