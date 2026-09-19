# Evaluation (`evaluation`)

The `evaluation` submodule evaluates predictions independently of the models
that produced them.

## Binary classification

Use `ClassifierEvaluator` separately for conformal prediction sets and point
classification. Labels must be `0` and `1`, and both set and probability
columns must follow that order:

```python
from tinyconformal.evaluation import ClassifierEvaluator

set_metrics = ClassifierEvaluator.evaluate_set(
    y_test,
    prediction_sets,
    coverage=0.95,
)
classification_metrics = ClassifierEvaluator.evaluate_classification(
    y_test,
    y_pred,
    y_prob,
)
```

## Predictive systems

Use `CPSEvaluator` for a tabular CPS distribution:

```python
from tinyconformal.evaluation import CPSEvaluator

interval_metrics = CPSEvaluator.evaluate_interval(y_test, distribution)
distribution_metrics = CPSEvaluator.evaluate_distribution(
    y_test, distribution, scale=y_train.std()
)
```

Use `PanelEvaluator` for MSCP/TSCQR interval DataFrames and time-series CPS
forecasts. `evaluate_distribution` calculates CRPS per series and normalizes it
by each series' training-target standard deviation:

```python
distribution_metrics = PanelEvaluator.evaluate_distribution(
    y_true=test_df,
    forecast=forecast,
    train_df=train_df,
)
```

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
