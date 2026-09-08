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


def call_with_supported_kwargs(method, **kwargs):
    """Call a method with its supported, non-None keyword arguments."""
    signature = inspect.signature(method)
    accepts_arbitrary_kwargs = any(
        item.kind == inspect.Parameter.VAR_KEYWORD
        for item in signature.parameters.values()
    )
    supported_kwargs = {
        name: value
        for name, value in kwargs.items()
        if value is not None
        and (accepts_arbitrary_kwargs or name in signature.parameters)
    }
    return method(**supported_kwargs)
