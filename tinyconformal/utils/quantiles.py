# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

import warnings

import numpy as np


def validate_alpha(alpha: float) -> float:
    """Validate and normalize a significance level."""
    if not isinstance(alpha, (int, float, np.integer, np.floating)) or not (
        0.0 < alpha < 1.0
    ):
        raise ValueError("alpha must be a number strictly between 0 and 1.")
    return float(alpha)


def _validate_sample_size(n: int) -> int:
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n <= 0:
        raise ValueError("The calibration sample size n must be a positive integer.")
    return int(n)


def _warn_unattainable(
    *,
    n: int,
    alpha: float,
    tails: int,
    warning_registry: set | None,
    context: str,
) -> None:
    key = (context, n, alpha, tails)
    if warning_registry is not None and key in warning_registry:
        return
    if warning_registry is not None:
        warning_registry.add(key)

    qualifier = "two-sided " if tails == 2 else ""
    warnings.warn(
        f"The requested {qualifier}coverage is not attainable for {context} "
        f"with the available calibration sample (n={n}); the conformal rank "
        "will be clipped to the observed score range.",
        RuntimeWarning,
        stacklevel=3,
    )


def conformal_quantile_level(
    n: int,
    alpha: float,
    *,
    warning_registry: set | None = None,
    context: str = "the estimator",
) -> float:
    """Return the exact upper conformal rank as a NumPy ``higher`` quantile level."""
    n = _validate_sample_size(n)
    alpha = validate_alpha(alpha)
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        _warn_unattainable(
            n=n,
            alpha=alpha,
            tails=1,
            warning_registry=warning_registry,
            context=context,
        )
    rank = int(np.clip(rank, 1, n))
    return 0.0 if n == 1 else (rank - 1) / (n - 1)


def central_conformal_quantile_levels(
    n: int,
    alpha: float,
    *,
    warning_registry: set | None = None,
    context: str = "the estimator",
) -> tuple[float, float]:
    """Return equal-tailed conformal ranks as NumPy ``higher`` quantile levels."""
    n = _validate_sample_size(n)
    alpha = validate_alpha(alpha)
    low_rank = int(np.floor((n + 1) * alpha / 2.0))
    high_rank = int(np.ceil((n + 1) * (1.0 - alpha / 2.0)))
    if low_rank < 1 or high_rank > n:
        _warn_unattainable(
            n=n,
            alpha=alpha,
            tails=2,
            warning_registry=warning_registry,
            context=context,
        )
    low_rank = int(np.clip(low_rank, 1, n))
    high_rank = int(np.clip(high_rank, 1, n))
    if n == 1:
        return 0.0, 0.0
    return (low_rank - 1) / (n - 1), (high_rank - 1) / (n - 1)
