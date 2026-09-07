# Copyright (c) 2024-2026 Lucas Leão
# TinyConformal - A small toolbox for conformal prediction
# Licensed under the MIT License

"""Public continuous and discrete cross-fitted CPS convenience classes."""

from __future__ import annotations

from sklearn.base import BaseEstimator

from .base import CrossConformalPredictiveSystem
from .distribution import DiscreteConformalDistribution


class ContinuousCrossConformalPredictiveSystem(CrossConformalPredictiveSystem):
    """Cross-fitted predictive system for continuous targets.

    Parameters
    ----------
    learner : estimator
        Unfitted location estimator implementing ``fit`` and ``predict``.
    dispersion_learner : estimator
        Unfitted estimator producing strictly positive conditional scales.
    """

    def __init__(self, learner, dispersion_learner):
        """Configure a continuous cross-fitted conformal predictive system."""
        super().__init__(
            learner=learner,
            dispersion_learner=dispersion_learner,
            discrete=False,
            minimum=None,
        )


class DiscreteCrossConformalPredictiveSystem(CrossConformalPredictiveSystem):
    """Cross-fitted predictive system for ordered integer outcomes.

    Parameters
    ----------
    learner : estimator
        Unfitted location estimator implementing ``fit`` and ``predict``.
    dispersion_learner : estimator
        Unfitted estimator producing strictly positive conditional scales.
    minimum : int or None, default=0
        Lower boundary of the integer support. Use ``0`` for counts, ``1`` for
        strictly positive outcomes, another integer for a known lower bound,
        or ``None`` when negative integers are valid.
    """

    def __init__(
        self,
        learner: BaseEstimator,
        dispersion_learner: BaseEstimator,
        minimum: int | None = 0,
    ):
        """Configure an integer-support cross-fitted conformal predictive system."""
        super().__init__(
            learner=learner,
            dispersion_learner=dispersion_learner,
            discrete=True,
            minimum=minimum,
        )

    def predict_distribution(self, X) -> DiscreteConformalDistribution:
        """Return one discrete predictive distribution per row of ``X``.

        The returned object exposes :meth:`cdf`, :meth:`ppf`, :meth:`pmf`,
        :meth:`interval` and :meth:`evaluate`. See
        :meth:`CrossConformalPredictiveSystem.predict_distribution` for the
        complete input, output, and error contract.
        """
        return super().predict_distribution(X)
