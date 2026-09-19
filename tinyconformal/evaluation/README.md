# Evaluation (`evaluation`)

The `evaluation` submodule evaluates predictions independently of the models
that produced them.

## First-stage forecaster diagnostics

`FirstStageEvaluator` checks the conditional-mean forecaster behind a CPS
before conformal scaling. Predictions must come from a held-out period.

```python
from tinyconformal.evaluation import FirstStageEvaluator

metrics = FirstStageEvaluator.evaluate(
    test_predictions,
    prediction_col="LinearRegression",
)

calibration = FirstStageEvaluator.calibration_table(
    test_predictions,
    prediction_col="LinearRegression",
    n_bins=10,
)
```

By default, panel identifiers and timestamps are read from `unique_id` and
`ds`. Use `id_col` and `time_col` for nonstandard schemas.
