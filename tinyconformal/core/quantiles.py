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


def temporal_decay_weights(n: int, decay: float = 0.99) -> np.ndarray:
    """Return normalized exponential weights from oldest to newest observation.

    Parameters
    ----------
    n : int
        Number of chronologically ordered observations.
    decay : float, default=0.99
        Multiplicative decay in ``(0, 1)``. The newest observation has unit
        unnormalized weight and older observations receive successive powers.

    Returns
    -------
    ndarray of shape (n,)
        Positive weights summing to one, ordered oldest to newest.
    """
    n = _validate_sample_size(n)
    if not isinstance(decay, (int, float, np.integer, np.floating)) or not (
        0.0 < decay < 1.0
    ):
        raise ValueError("decay must be a number strictly between 0 and 1.")
    weights = float(decay) ** np.arange(n - 1, -1, -1, dtype=float)
    return weights / weights.sum()


def weighted_quantile(values, quantile: float, weights, axis=None):
    """Compute a higher-style weighted quantile along the calibration axis.

    Parameters
    ----------
    values : array-like
        Values whose weighted quantile is requested.
    quantile : float
        Quantile level in ``[0, 1]``.
    weights : array-like of shape (values.shape[axis],)
        Finite non-negative weights with positive total mass.
    axis : int or None, default=None
        Calibration axis. ``None`` flattens ``values``.

    Returns
    -------
    scalar or ndarray
        Smallest sorted value whose cumulative normalized weight reaches the
        requested quantile.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1].")
    if axis is None:
        values = values.reshape(-1)
        axis = 0
    values = np.moveaxis(values, axis, 0)
    if weights.shape != (values.shape[0],):
        raise ValueError("weights must match the selected calibration axis.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("weights must be finite, non-negative, and have positive mass.")
    weights = weights / weights.sum()
    order = np.argsort(values, axis=0)
    sorted_values = np.take_along_axis(values, order, axis=0)
    expanded_weights = np.broadcast_to(
        weights.reshape((-1,) + (1,) * (values.ndim - 1)), values.shape
    )
    sorted_weights = np.take_along_axis(expanded_weights, order, axis=0)
    indices = np.argmax(np.cumsum(sorted_weights, axis=0) >= quantile, axis=0)
    return np.take_along_axis(sorted_values, indices[None, ...], axis=0)[0]


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
