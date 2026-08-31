# Calibration Primitives (`core`)

The `core` submodule contains reusable calibration outputs and cross-validation
helpers. Most users can work directly with the estimators in `classifier`,
`regressor`, `distribution`, and `series`; use `core` when controlling the
out-of-fold calibration workflow explicitly.

## Public API

- `CrossValidationCalibration`: produces out-of-fold scores or probabilities
  for ICP, CQR, CPS, and classification.
- `CrossFittedCPSCalibration`: immutable result containing raw residuals,
  conditional scales, and standardized CPS residuals.

## Usage

```python
from tinyconformal.core import CrossValidationCalibration
from tinyconformal.regressor import ConformalizedRegressor

scores = CrossValidationCalibration.icp_scores(
    learner,
    X_train,
    y_train,
    cv=5,
    n_jobs=-1,
)

fitted_learner = learner.fit(X_train, y_train)
conformal = ConformalizedRegressor(fitted_learner).fit_from_scores(scores)
intervals = conformal.predict_interval(X_test)
```

Available workflows include `icp_scores`, `cqr_scores`, `cps_scores`, and
classification probability generation. All calibration predictions are
out-of-fold: an observation is never scored by a model fitted on that same
observation.

Lower-level functions in `core.conformal` and `core.quantiles` support the
library implementation but are not part of the submodule's top-level exports.
