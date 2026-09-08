"""Shared callable-inspection helpers."""

# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

import inspect


def accepts_parameter(method, parameter: str) -> bool:
    """Return whether a callable accepts a named or arbitrary keyword argument."""
    signature = inspect.signature(method)
    return parameter in signature.parameters or any(
        item.kind == inspect.Parameter.VAR_KEYWORD
        for item in signature.parameters.values()
    )
