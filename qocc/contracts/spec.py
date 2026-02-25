"""Contract specification and result data structures.

A contract is a machine-checkable requirement with:
- spec: what is preserved and how tested
- evaluator: method used (exact / sampling / stabilizer)
- tolerances: numeric thresholds
- confidence: confidence intervals / significance
- resource_budget: max shots / time / memory
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContractSpec:
    """Specification for a semantic or cost contract.

    Attributes:
        name: Human-readable identifier.
        type: One of ``"observable"``, ``"distribution"``, ``"clifford"``, ``"cost"``.
        spec: Details of what to check.
        tolerances: Numeric tolerance thresholds.
        confidence: Confidence configuration (alpha, CI level, etc.).
        resource_budget: Max shots / time / memory.
        evaluator: Preferred evaluation method.
    """

    name: str
    type: str  # "observable" | "distribution" | "clifford" | "cost"
    spec: dict[str, Any] = field(default_factory=dict)
    tolerances: dict[str, float] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)
    resource_budget: dict[str, Any] = field(default_factory=dict)
    evaluator: str = "auto"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "spec": self.spec,
            "tolerances": self.tolerances,
            "confidence": self.confidence,
            "resource_budget": self.resource_budget,
            "evaluator": self.evaluator,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ContractSpec:
        return cls(
            name=d["name"],
            type=d["type"],
            spec=d.get("spec", {}),
            tolerances=d.get("tolerances", {}),
            confidence=d.get("confidence", {}),
            resource_budget=d.get("resource_budget", {}),
            evaluator=d.get("evaluator", "auto"),
        )


@dataclass
class ContractResult:
    """Result of evaluating a contract.

    Attributes:
        name: Contract name.
        passed: Whether the contract was satisfied.
        details: Extra data (point estimates, CIs, test stats, etc.).
    """

    name: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "details": self.details,
        }


@dataclass
class CostSpec:
    """Cost/optimization objective specification.

    Attributes:
        objectives: List of objective dicts ``{"metric": ..., "direction": "minimize"}``.
        constraints: Hard constraints ``{"metric": ..., "max": ..., "min": ...}``.
        weights: Weights for scalarised multi-objective.
    """

    objectives: list[dict[str, Any]] = field(default_factory=list)
    constraints: list[dict[str, Any]] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objectives": self.objectives,
            "constraints": self.constraints,
            "weights": self.weights,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CostSpec:
        return cls(
            objectives=d.get("objectives", []),
            constraints=d.get("constraints", []),
            weights=d.get("weights", {}),
        )


@dataclass
class CostResult:
    """Result of cost evaluation."""

    score: float
    breakdown: dict[str, float] = field(default_factory=dict)
    feasible: bool = True
    constraint_violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "breakdown": self.breakdown,
            "feasible": self.feasible,
            "constraint_violations": self.constraint_violations,
        }
