# Binary Conformal Classification (`classifier`)

The `classifier` submodule builds prediction sets for binary classifiers. It
contains marginal and class-conditional calibration strategies.

## Public models

- `BinaryMarginalConformalClassifier`: calibrates one score distribution over
  all observations. Use it when overall marginal coverage is the objective.
- `BinaryClassConditionalConformalClassifier`: calibrates each class
  separately. Use it when each class should receive its own coverage control.

## Usage

The underlying classifier must already be fitted and expose class
probabilities. Calibration can use a held-out set, out-of-bag probabilities, or
precomputed out-of-fold probabilities.

```python
from sklearn.ensemble import RandomForestClassifier
from tinyconformal.classifier import BinaryMarginalConformalClassifier
from tinyconformal.evaluation import ClassifierEvaluator

learner = RandomForestClassifier(n_estimators=300, random_state=42)
learner.fit(X_train, y_train)

conformal = BinaryMarginalConformalClassifier(learner, alpha=0.05)
conformal.fit(X_calibration, y_calibration)

prediction_sets = conformal.predict_set(X_test)
y_pred = conformal.predict(X_test)
y_prob = conformal.predict_proba(X_test)

set_metrics = ClassifierEvaluator.evaluate_set(
    y_test, prediction_sets, coverage=1 - conformal.alpha
)
classification_metrics = ClassifierEvaluator.evaluate_classification(
    y_test, y_pred, y_prob
)
```

`ClassifierEvaluator` assumes labels `0` and `1`. Column 0 of prediction sets
and probabilities represents class 0; column 1 represents class 1.

For cross-validated calibration, generate out-of-fold probabilities with
`tinyconformal.core.CrossValidationCalibration` and pass them to
`fit_from_probabilities`.

## Choosing a strategy

| Requirement | Model |
|---|---|
| Overall population coverage | `BinaryMarginalConformalClassifier` |
| Coverage controlled separately by class | `BinaryClassConditionalConformalClassifier` |

Both models are intended for binary labels. Their `predict` method returns
prediction sets rather than a single forced label.
