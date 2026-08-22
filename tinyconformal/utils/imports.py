# Copyright (c) 2024-2026 Lucas Leão
# tinyconformal - A small toolbox for mlops
# Licensed under the MIT License

from functools import wraps
import importlib.util
from typing import Dict, List, Callable, Any

EXTRA_DEPENDENCIES: Dict[str, List[str]] = {
    "series": ["statsforecast", "mlforecast", "utilsforecast", "statsmodels"],
    "notebook": ["nbformat", "ipykernel"],
}

EXTRA_DEPENDENCIES["all"] = list(
    {module for extra in ["series", "notebook"] for module in EXTRA_DEPENDENCIES[extra]}
)


def check_extra(extra_name: str) -> None:
    """
    Checks whether all Python modules required for a given extra are installed.

    Parameters
    ----------
    extra_name : str
        The name of the optional extra dependency group (e.g., 'series').

    Raises
    ------
    ImportError
        If one or more modules required by the extra are missing from the environment.
    """
    required_modules = EXTRA_DEPENDENCIES.get(extra_name, [])

    missing_modules = [
        mod for mod in required_modules if importlib.util.find_spec(mod) is None
    ]

    if missing_modules:
        missing_fmt = ", ".join(f"'{m}'" for m in missing_modules)
        raise ImportError(
            f"The requested functionality requires the '{extra_name}' extra. "
            f"Missing required module(s): {missing_fmt}. "
            f"Please install them via: pip install tinyshift[{extra_name}]"
        )


def requires_extra(
    extra_name: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator that checks for required optional dependencies before executing the wrapped function.

    Parameters
    ----------
    extra_name : str
        The name of the optional extra dependency group required by the function or method.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            check_extra(extra_name)
            return func(*args, **kwargs)

        return wrapper

    return decorator
