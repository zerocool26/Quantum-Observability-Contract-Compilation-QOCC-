"""Search space definition for candidate pipeline generation.

Generates compilable pipeline specifications by varying parameters such as
optimization level, routing method, and gate decomposition choices.
"""

from __future__ import annotations

import hashlib
import itertools
import json
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
    """

    adapter: str = "qiskit"
    optimization_levels: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    seeds: list[int] = field(default_factory=lambda: [42])
    routing_methods: list[str] = field(default_factory=lambda: ["stochastic", "sabre"])
    extra_params: dict[str, list[Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "optimization_levels": self.optimization_levels,
            "seeds": self.seeds,
            "routing_methods": self.routing_methods,
            "extra_params": self.extra_params,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SearchSpaceConfig:
        return cls(
            adapter=d.get("adapter", "qiskit"),
            optimization_levels=d.get("optimization_levels", [0, 1, 2, 3]),
            seeds=d.get("seeds", [42]),
            routing_methods=d.get("routing_methods", ["stochastic", "sabre"]),
            extra_params=d.get("extra_params", {}),
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

    Generates the Cartesian product of all parameter axes.
    """
    candidates: list[Candidate] = []

    # Build parameter axes
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

    return candidates
