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

learner = RandomForestClassifier(n_estimators=300, random_state=42)
learner.fit(X_train, y_train)

conformal = BinaryMarginalConformalClassifier(learner, alpha=0.05)
conformal.fit(X_calibration, y_calibration)

prediction_sets = conformal.predict(X_test)
metrics = conformal.evaluate(X_test, y_test)
```

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
