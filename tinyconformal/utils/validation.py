"""Shared validation helpers."""

# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

import numpy as np


def validate_integer_support(minimum: int | None) -> int | None:
    """Validate and normalize an optional lower integer-support boundary."""
    if minimum is not None and not isinstance(minimum, (int, np.integer)):
        raise TypeError("minimum must be an integer or None.")
    return None if minimum is None else int(minimum)


def validate_discrete_targets(values, minimum: int | None, *, name: str) -> np.ndarray:
    """Return finite integer-valued targets within an optional lower boundary."""
    targets = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(targets)) or np.any(targets != np.floor(targets)):
        raise ValueError(f"{name} must be finite integers.")
    if minimum is not None and np.any(targets < minimum):
        raise ValueError(f"{name} must be >= {minimum}.")
    return targets
