"""Candidate selection — choose the best candidate under contracts.

Selects the best candidate that satisfies all semantic contracts
and minimises the cost contract.  Supports single-objective and
multi-objective Pareto selection.
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
        pareto_frontier: Pareto-optimal candidates (multi-objective mode).
    """

    selected: Candidate | None = None
    all_candidates: list[Candidate] = field(default_factory=list)
    feasible: bool = True
    reason: str = ""
    pareto_frontier: list[Candidate] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected.to_dict() if self.selected else None,
            "num_candidates": len(self.all_candidates),
            "feasible": self.feasible,
            "reason": self.reason,
            "pareto_frontier": [c.to_dict() for c in self.pareto_frontier],
        }


def _is_dominated(a: dict[str, float], b: dict[str, float], objectives: list[str]) -> bool:
    """Check if candidate *a* is dominated by candidate *b*.

    Both dicts map objective metric names to values.  All objectives
    are minimised (lower is better).

    *b* dominates *a* iff b[k] <= a[k] for all k AND b[k] < a[k] for at least one k.
    """
    all_leq = True
    any_lt = False
    for obj in objectives:
        va = a.get(obj, float("inf"))
        vb = b.get(obj, float("inf"))
        if vb > va:
            all_leq = False
            break
        if vb < va:
            any_lt = True
    return all_leq and any_lt


def compute_pareto_frontier(
    candidates: list[Candidate],
    objectives: list[str] | None = None,
) -> list[Candidate]:
    """Compute the Pareto-optimal frontier over *objectives*.

    Parameters:
        candidates: Pool of candidates with populated ``metrics``.
        objectives: Metric keys to minimise.  Defaults to
                    ``["depth", "gates_2q", "proxy_error_score"]``.

    Returns:
        List of non-dominated candidates.
    """
    if objectives is None:
        objectives = ["depth", "gates_2q", "proxy_error_score"]

    obj_vals = []
    for c in candidates:
        obj_vals.append({k: float(c.metrics.get(k, float("inf"))) for k in objectives})

    frontier: list[Candidate] = []
    for i, c in enumerate(candidates):
        dominated = False
        for j, other in enumerate(candidates):
            if i == j:
                continue
            if _is_dominated(obj_vals[i], obj_vals[j], objectives):
                dominated = True
                break
        if not dominated:
            frontier.append(c)
    return frontier


def select_best(
    candidates: list[Candidate],
    require_validated: bool = True,
    mode: str = "single",
    objectives: list[str] | None = None,
) -> SelectionResult:
    """Select the best candidate.

    Modes:
        ``"single"``  — minimise surrogate score (default).
        ``"pareto"``  — compute Pareto frontier, select the candidate with
                        lowest surrogate score from the frontier.

    Parameters:
        candidates: Scored and (optionally) validated candidates.
        require_validated: If True, only consider validated candidates.
        mode: ``"single"`` or ``"pareto"``.
        objectives: Metric keys for Pareto mode.
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

    frontier: list[Candidate] = []

    if mode == "pareto" and passing:
        frontier = compute_pareto_frontier(passing, objectives)
        best = min(frontier, key=lambda c: c.surrogate_score) if frontier else None
        if best:
            return SelectionResult(
                selected=best,
                all_candidates=candidates,
                feasible=True,
                reason=(
                    f"Pareto selection: {len(frontier)} non-dominated candidates. "
                    f"Selected {best.candidate_id} with score {best.surrogate_score:.4f}"
                ),
                pareto_frontier=frontier,
            )

    if passing:
        best = min(passing, key=lambda c: c.surrogate_score)
        return SelectionResult(
            selected=best,
            all_candidates=candidates,
            feasible=True,
            reason=f"Selected candidate {best.candidate_id} with score {best.surrogate_score:.4f}",
            pareto_frontier=frontier,
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
            pareto_frontier=frontier,
        )

    return SelectionResult(
        all_candidates=candidates,
        feasible=False,
        reason="No candidates available for selection.",
    )
