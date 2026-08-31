# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License


from .cqr import ConformalizedQuantileRegressor
from .icp import ConformalizedRegressor

__all__ = ["ConformalizedQuantileRegressor", "ConformalizedRegressor"]
