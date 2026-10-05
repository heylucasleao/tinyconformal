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

For conditional-mean forecast diagnostics, use
`tinyshift.forecasting.MeanForecasterEvaluator` before evaluating conformal
intervals or predictive distributions.
