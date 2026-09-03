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
    cv : int or cross-validation splitter, default=5
        Cross-fitting strategy for location and dispersion predictions.
    n_jobs : int or None, default=None
        Parallel jobs passed to scikit-learn cross-validation.
    """

    def __init__(self, learner, dispersion_learner, cv=5, n_jobs=None):
        super().__init__(
            learner=learner,
            dispersion_learner=dispersion_learner,
            cv=cv,
            n_jobs=n_jobs,
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
    cv : int or cross-validation splitter, default=5
        Cross-fitting strategy for location and dispersion predictions.
    n_jobs : int or None, default=None
        Parallel jobs passed to scikit-learn cross-validation.
    minimum : int or None, default=0
        Lower boundary of the integer support. Use ``0`` for counts, ``1`` for
        strictly positive outcomes, another integer for a known lower bound,
        or ``None`` when negative integers are valid.
    """

    def __init__(
        self,
        learner: BaseEstimator,
        dispersion_learner: BaseEstimator,
        cv=5,
        n_jobs=None,
        minimum: int | None = 0,
    ):
        super().__init__(
            learner=learner,
            dispersion_learner=dispersion_learner,
            cv=cv,
            n_jobs=n_jobs,
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
