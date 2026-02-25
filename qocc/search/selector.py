"""Candidate selection — choose the best candidate under contracts.

Selects the best candidate that satisfies all semantic contracts
and minimises the cost contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qocc.search.space import Candidate


@dataclass
class SelectionResult:
    """Result of candidate selection.

    Attributes:
        selected: The chosen candidate (or None if infeasible).
        all_candidates: Full candidate table.
        feasible: Whether any candidate satisfied all contracts.
        reason: Explanation of selection.
    """

    selected: Candidate | None = None
    all_candidates: list[Candidate] = field(default_factory=list)
    feasible: bool = True
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected.to_dict() if self.selected else None,
            "num_candidates": len(self.all_candidates),
            "feasible": self.feasible,
            "reason": self.reason,
        }


def select_best(
    candidates: list[Candidate],
    require_validated: bool = True,
) -> SelectionResult:
    """Select the best candidate.

    Strategy:
    1. Filter to candidates that passed all contracts.
    2. Among those, pick the one with lowest surrogate score.
    3. If none pass, return infeasible with the best-effort nearest.

    Parameters:
        candidates: Scored and (optionally) validated candidates.
        require_validated: If True, only consider validated candidates.
    """
    pool = candidates
    if require_validated:
        pool = [c for c in candidates if c.validated]

    # Check contract pass
    passing: list[Candidate] = []
    for c in pool:
        all_pass = all(r.get("passed", False) for r in c.contract_results) if c.contract_results else True
        if all_pass:
            passing.append(c)

    if passing:
        best = min(passing, key=lambda c: c.surrogate_score)
        return SelectionResult(
            selected=best,
            all_candidates=candidates,
            feasible=True,
            reason=f"Selected candidate {best.candidate_id} with score {best.surrogate_score:.4f}",
        )

    # Infeasible — return best-effort
    if pool:
        best_effort = min(pool, key=lambda c: c.surrogate_score)
        failures = []
        for r in best_effort.contract_results:
            if not r.get("passed", True):
                failures.append(r.get("name", "unknown"))
        return SelectionResult(
            selected=best_effort,
            all_candidates=candidates,
            feasible=False,
            reason=f"No candidate satisfies all contracts. Best-effort: {best_effort.candidate_id}. "
                   f"Failed contracts: {failures}",
        )

    return SelectionResult(
        all_candidates=candidates,
        feasible=False,
        reason="No candidates available for selection.",
    )
