# Conformal Predictive Distributions (`distribution`)

The `distribution` submodule cross-fits a location estimator and a conditional
scale estimator, then returns one complete conformal predictive distribution
per input row.

## Public models

- `ContinuousCrossConformalPredictiveSystem`: real-valued targets.
- `DiscreteCrossConformalPredictiveSystem`: ordered integer targets and counts.

## Continuous usage

```python
from tinyconformal.distribution import ContinuousCrossConformalPredictiveSystem

cps = ContinuousCrossConformalPredictiveSystem(
    learner=location_regressor,
    dispersion_learner=scale_regressor,
    cv=5,
).fit(X_train, y_train)

distribution = cps.predict_distribution(X_test)
median = distribution.ppf(0.5)
intervals = distribution.interval(coverage=0.90)
probabilities = distribution.cdf(values)
```

There is intentionally no `predict` shortcut: request the predictive
distribution and select the required functional, such as `ppf(0.5)` for its
median.

## Discrete usage and `minimum`

```python
from tinyconformal.distribution import DiscreteCrossConformalPredictiveSystem

cps = DiscreteCrossConformalPredictiveSystem(
    learner=count_regressor,
    dispersion_learner=scale_regressor,
    minimum=0,
).fit(X_train, y_train)

distribution = cps.predict_distribution(X_test)
masses = distribution.pmf([0, 1, 2])
```

Use `minimum=0` for counts, `minimum=1` for strictly positive integer outcomes,
another integer for a known lower boundary, or `None` when negative integers
are valid.

The concrete distribution and base classes returned by these models are
implementation details and are not exported from this submodule.
