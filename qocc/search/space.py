"""Search space definition for candidate pipeline generation.

Generates compilable pipeline specifications by varying parameters such as
optimization level, routing method, and gate decomposition choices.

Supports three strategies:
  - ``"grid"``  — exhaustive Cartesian product (default)
  - ``"random"`` — random sampling from the parameter space
  - ``"bayesian"`` — surrogate-model-guided adaptive search (uses scipy/numpy)
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass, field
from typing import Any

from qocc.core.circuit_handle import PipelineSpec


@dataclass
class SearchSpaceConfig:
    """Defines the search space for closed-loop compilation.

    Attributes:
        adapter: Adapter name.
        optimization_levels: Levels to try.
        seeds: Seeds to try.
        routing_methods: Routing strategies (adapter-specific).
        extra_params: Additional parameter grids ``{key: [val1, val2, ...]}``.
        strategy: Search strategy — ``"grid"``, ``"random"``, or ``"bayesian"``.
        max_candidates: Maximum candidates to generate (used by random/bayesian).
        bayesian_init_points: Initial random samples before surrogate kicks in.
        bayesian_explore_weight: Exploration vs exploitation (higher = more exploration).
    """

    adapter: str = "qiskit"
    optimization_levels: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    seeds: list[int] = field(default_factory=lambda: [42])
    routing_methods: list[str] = field(default_factory=lambda: ["stochastic", "sabre"])
    extra_params: dict[str, list[Any]] = field(default_factory=dict)
    strategy: str = "grid"
    max_candidates: int = 50
    bayesian_init_points: int = 5
    bayesian_explore_weight: float = 1.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "optimization_levels": self.optimization_levels,
            "seeds": self.seeds,
            "routing_methods": self.routing_methods,
            "extra_params": self.extra_params,
            "strategy": self.strategy,
            "max_candidates": self.max_candidates,
            "bayesian_init_points": self.bayesian_init_points,
            "bayesian_explore_weight": self.bayesian_explore_weight,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SearchSpaceConfig:
        return cls(
            adapter=d.get("adapter", "qiskit"),
            optimization_levels=d.get("optimization_levels", [0, 1, 2, 3]),
            seeds=d.get("seeds", [42]),
            routing_methods=d.get("routing_methods", ["stochastic", "sabre"]),
            extra_params=d.get("extra_params", {}),
            strategy=d.get("strategy", "grid"),
            max_candidates=d.get("max_candidates", 50),
            bayesian_init_points=d.get("bayesian_init_points", 5),
            bayesian_explore_weight=d.get("bayesian_explore_weight", 1.5),
        )


@dataclass
class Candidate:
    """A single candidate pipeline with its outputs and scores.

    Attributes:
        pipeline: The pipeline specification.
        candidate_id: Unique identifier / hash.
        metrics: Computed metrics snapshot dict.
        surrogate_score: Cheap proxy score (lower is better).
        validated: Whether expensive validation has been run.
        validation_result: Results from expensive validation.
        contract_results: Contract evaluation results.
    """

    pipeline: PipelineSpec
    candidate_id: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    surrogate_score: float = float("inf")
    validated: bool = False
    validation_result: dict[str, Any] = field(default_factory=dict)
    contract_results: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.candidate_id:
            self.candidate_id = self.pipeline.stable_hash()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "pipeline": self.pipeline.to_dict(),
            "metrics": self.metrics,
            "surrogate_score": self.surrogate_score,
            "validated": self.validated,
            "validation_result": self.validation_result,
            "contract_results": self.contract_results,
        }


def generate_candidates(config: SearchSpaceConfig) -> list[Candidate]:
    """Enumerate candidate pipelines from the search space.

    Strategy:
      - ``"grid"``     — Cartesian product of all parameter axes.
      - ``"random"``   — Random sampling up to ``max_candidates``.
      - ``"bayesian"`` — Batch of initial random + placeholders for adaptive rounds.
    """
    strategy = config.strategy.lower()
    if strategy == "random":
        return _generate_random(config)
    elif strategy == "bayesian":
        return _generate_bayesian_init(config)
    else:
        return _generate_grid(config)


def _generate_grid(config: SearchSpaceConfig) -> list[Candidate]:
    """Exhaustive Cartesian product generation."""
    candidates: list[Candidate] = []

    axes: dict[str, list[Any]] = {
        "optimization_level": config.optimization_levels,
        "seed": config.seeds,
    }
    if config.routing_methods:
        axes["routing_method"] = config.routing_methods
    axes.update(config.extra_params)

    keys = list(axes.keys())
    values = [axes[k] for k in keys]

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        opt_level = params.pop("optimization_level")
        seed = params.pop("seed", 42)
        params["seed"] = seed

        pipeline = PipelineSpec(
            adapter=config.adapter,
            optimization_level=opt_level,
            parameters=params,
        )
        candidates.append(Candidate(pipeline=pipeline))

        if len(candidates) >= config.max_candidates:
            break

    return candidates


def _generate_random(config: SearchSpaceConfig) -> list[Candidate]:
    """Random sampling from the parameter space."""
    import numpy as np

    rng = np.random.RandomState(config.seeds[0] if config.seeds else 42)
    candidates: list[Candidate] = []
    seen: set[str] = set()

    axes: dict[str, list[Any]] = {
        "optimization_level": config.optimization_levels,
        "seed": config.seeds,
    }
    if config.routing_methods:
        axes["routing_method"] = config.routing_methods
    axes.update(config.extra_params)

    keys = list(axes.keys())
    max_attempts = config.max_candidates * 3

    for _ in range(max_attempts):
        if len(candidates) >= config.max_candidates:
            break

        params: dict[str, Any] = {}
        for k in keys:
            vals = axes[k]
            params[k] = vals[rng.randint(0, len(vals))]

        opt_level = params.pop("optimization_level")
        seed = params.pop("seed", 42)
        params["seed"] = seed

        pipeline = PipelineSpec(
            adapter=config.adapter,
            optimization_level=opt_level,
            parameters=params,
        )
        cid = pipeline.stable_hash()[:16]
        if cid not in seen:
            seen.add(cid)
            candidates.append(Candidate(pipeline=pipeline))

    return candidates


def _generate_bayesian_init(config: SearchSpaceConfig) -> list[Candidate]:
    """Generate initial candidates for Bayesian optimization.

    Creates ``bayesian_init_points`` random candidates as the initial
    exploratory batch.  The adaptive loop is driven by
    ``BayesianSearchOptimizer`` after scores for these are known.
    """
    old_max = config.max_candidates
    config_copy = SearchSpaceConfig(
        adapter=config.adapter,
        optimization_levels=config.optimization_levels,
        seeds=config.seeds,
        routing_methods=config.routing_methods,
        extra_params=config.extra_params,
        strategy="random",
        max_candidates=config.bayesian_init_points,
    )
    return _generate_random(config_copy)


class BayesianSearchOptimizer:
    """Surrogate-model-guided adaptive search.

    Uses a simple Gaussian-process-like UCB (Upper Confidence Bound)
    acquisition function built on numpy alone (no sklearn required).

    Usage::

        opt = BayesianSearchOptimizer(config)
        init_candidates = opt.initial_candidates()
        # compile & score init_candidates ...
        for round in range(n_rounds):
            next_batch = opt.suggest(scored_candidates, batch_size=4)
            # compile & score next_batch ...
    """

    def __init__(self, config: SearchSpaceConfig) -> None:
        self.config = config
        self._axes: dict[str, list[Any]] = {
            "optimization_level": config.optimization_levels,
            "seed": config.seeds,
        }
        if config.routing_methods:
            self._axes["routing_method"] = config.routing_methods
        self._axes.update(config.extra_params)

        self._keys = list(self._axes.keys())
        # Map categorical values to integer indices
        self._encoders: dict[str, dict[Any, int]] = {}
        for k, vals in self._axes.items():
            self._encoders[k] = {v: i for i, v in enumerate(vals)}

        self._observed_X: list[list[float]] = []
        self._observed_Y: list[float] = []
        self.explore_weight = config.bayesian_explore_weight

    def _encode(self, params: dict[str, Any]) -> list[float]:
        """Encode a parameter dict to a numeric vector."""
        vec = []
        for k in self._keys:
            val = params.get(k, 0)
            enc = self._encoders[k]
            idx = enc.get(val, 0)
            # Normalize to [0, 1]
            vec.append(idx / max(len(enc) - 1, 1))
        return vec

    def _decode(self, vec: list[float]) -> dict[str, Any]:
        """Decode a numeric vector back to parameters."""
        params: dict[str, Any] = {}
        for i, k in enumerate(self._keys):
            vals = self._axes[k]
            idx = round(vec[i] * max(len(vals) - 1, 1))
            idx = max(0, min(idx, len(vals) - 1))
            params[k] = vals[idx]
        return params

    def initial_candidates(self) -> list[Candidate]:
        """Generate the initial random exploratory batch."""
        return _generate_bayesian_init(self.config)

    def observe(self, candidates: list[Candidate]) -> None:
        """Record observed (compiled + scored) candidates."""
        for c in candidates:
            if c.surrogate_score < float("inf"):
                params = c.pipeline.to_dict().get("parameters", {})
                params["optimization_level"] = c.pipeline.optimization_level
                self._observed_X.append(self._encode(params))
                self._observed_Y.append(c.surrogate_score)

    def suggest(self, batch_size: int = 4) -> list[Candidate]:
        """Suggest the next batch of candidates using UCB acquisition.

        Uses a simple RBF-kernel distance-based uncertainty estimator:
        - Mean estimate = weighted average of nearest observed scores
        - Uncertainty = inverse of distance to nearest observed point
        - UCB = -mean + explore_weight * uncertainty  (minimising score)
        """
        import numpy as np

        if not self._observed_X:
            return _generate_bayesian_init(self.config)

        X_obs = np.array(self._observed_X)
        Y_obs = np.array(self._observed_Y)

        # Normalise Y for stable acquisition
        y_mean = Y_obs.mean()
        y_std = Y_obs.std() + 1e-8
        Y_norm = (Y_obs - y_mean) / y_std

        # Generate a large pool of random candidates
        rng = np.random.RandomState(len(self._observed_X))
        pool_size = max(batch_size * 20, 200)
        pool_vecs: list[list[float]] = []
        for _ in range(pool_size):
            vec = [rng.uniform(0, 1) for _ in self._keys]
            pool_vecs.append(vec)

        X_pool = np.array(pool_vecs)

        # Compute UCB for each pool candidate
        ucb_scores = []
        length_scale = 0.3
        for xp in X_pool:
            # RBF kernel distances
            dists = np.sqrt(((X_obs - xp) ** 2).sum(axis=1))
            weights = np.exp(-dists**2 / (2 * length_scale**2))
            w_sum = weights.sum() + 1e-8

            # Predicted mean (weighted average of observed)
            mu = (weights * Y_norm).sum() / w_sum

            # Uncertainty: inversely proportional to total weight
            sigma = 1.0 / (w_sum + 1e-8)

            # UCB: minimise score → lower is better → negate mean, add uncertainty
            ucb = -mu + self.explore_weight * sigma
            ucb_scores.append(ucb)

        ucb_arr = np.array(ucb_scores)
        # Select top-batch_size by UCB
        top_idx = np.argsort(ucb_arr)[-batch_size:]

        candidates: list[Candidate] = []
        for idx in top_idx:
            params = self._decode(pool_vecs[idx])
            opt_level = params.pop("optimization_level", 1)
            seed = params.pop("seed", 42)
            params["seed"] = seed
            pipeline = PipelineSpec(
                adapter=self.config.adapter,
                optimization_level=int(opt_level) if isinstance(opt_level, (int, float)) else opt_level,
                parameters=params,
            )
            candidates.append(Candidate(pipeline=pipeline))

        return candidates
