"""Expensive validation for top-k candidates.

Validates candidates using simulation (exact for small, sampling for medium).
"""

from __future__ import annotations

from typing import Any

from qocc import DEFAULT_SEED
from qocc.adapters.base import SimulationSpec
from qocc.search.space import Candidate


def validate_candidates(
    candidates: list[Candidate],
    adapter: Any,
    circuit_handles: dict[str, Any],
    top_k: int = 5,
    sim_spec: SimulationSpec | None = None,
) -> list[Candidate]:
    """Run expensive validation on the top-k candidates.

    Parameters:
        candidates: Sorted list (best first).
        adapter: Adapter instance for simulation.
        circuit_handles: ``{candidate_id: CircuitHandle}`` for compiled circuits.
        top_k: Number of candidates to validate.
        sim_spec: Simulation specification.

    Returns:
        List of validated candidates.
    """
    if sim_spec is None:
        sim_spec = SimulationSpec(shots=1024, seed=DEFAULT_SEED)

    validated: list[Candidate] = []

    for candidate in candidates[:top_k]:
        handle = circuit_handles.get(candidate.candidate_id)
        if handle is None:
            candidate.validated = False
            candidate.validation_result = {"error": "No compiled circuit available"}
            validated.append(candidate)
            continue

        try:
            result = adapter.simulate(handle, sim_spec)
            candidate.validated = True
            candidate.validation_result = {
                "counts": result.counts,
                "shots": result.shots,
                "seed": result.seed,
            }
        except Exception as exc:
            candidate.validated = False
            candidate.validation_result = {"error": str(exc)}

        validated.append(candidate)

    return validated
