import warnings

import numpy as np
import pytest

from tinyconformal.core.quantiles import (
    central_conformal_quantile_levels,
    conformal_quantile_level,
    temporal_decay_weights,
    validate_alpha,
    weighted_quantile,
)


def test_upper_level_selects_exact_conformal_order_statistic():
    scores = np.arange(1, 101)
    level = conformal_quantile_level(100, 0.05)

    assert level == pytest.approx(95 / 99)
    assert np.quantile(scores, level, method="higher") == 96


def test_central_levels_select_exact_equal_tailed_order_statistics():
    scores = np.arange(1, 21)
    low_level, high_level = central_conformal_quantile_levels(20, 0.10)

    assert np.quantile(scores, low_level, method="higher") == 1
    assert np.quantile(scores, high_level, method="higher") == 20


def test_unattainable_coverage_warns_only_once_per_registry():
    registry = set()

    with pytest.warns(RuntimeWarning, match="not attainable"):
        conformal_quantile_level(7, 0.10, warning_registry=registry)
    with warnings.catch_warnings(record=True) as warnings_record:
        warnings.simplefilter("always")
        conformal_quantile_level(7, 0.10, warning_registry=registry)

    assert not warnings_record


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.1, True, None])
def test_validate_alpha_rejects_invalid_values(alpha):
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        validate_alpha(alpha)


def test_temporal_decay_weights_favor_recent_observations():
    weights = temporal_decay_weights(3, decay=0.5)
    np.testing.assert_allclose(weights, np.array([0.25, 0.5, 1.0]) / 1.75)
    assert weighted_quantile([-10.0, 0.0, 10.0], 0.5, weights) == 10.0


@pytest.mark.parametrize("decay", [0.0, 1.0, -0.1, 1.1, "0.99"])
def test_temporal_decay_weights_reject_invalid_decay(decay):
    with pytest.raises(ValueError, match="decay"):
        temporal_decay_weights(3, decay=decay)
