# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License


from .class_conditional import BinaryClassConditionalConformalClassifier
from .marginal import BinaryMarginalConformalClassifier

__all__ = [
    "BinaryClassConditionalConformalClassifier",
    "BinaryMarginalConformalClassifier",
]
