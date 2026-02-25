"""Exact contract evaluation.

For small circuits, evaluate contracts by exact statevector comparison.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qocc.contracts.spec import ContractResult, ContractSpec


def evaluate_exact_equivalence(
    spec: ContractSpec,
    statevector_before: np.ndarray | list[complex],
    statevector_after: np.ndarray | list[complex],
) -> ContractResult:
    """Check exact equivalence of two statevectors up to global phase.

    Uses fidelity: F = |<ψ|φ>|².  Pass if F >= 1 - tolerance.
    """
    sv1 = np.asarray(statevector_before, dtype=complex).flatten()
    sv2 = np.asarray(statevector_after, dtype=complex).flatten()

    if sv1.shape != sv2.shape:
        return ContractResult(
            name=spec.name,
            passed=False,
            details={"error": "Statevector dimensions differ", "dim1": len(sv1), "dim2": len(sv2)},
        )

    # Fidelity = |<ψ|φ>|²
    overlap = np.vdot(sv1, sv2)
    fidelity = float(abs(overlap) ** 2)
    tolerance = spec.tolerances.get("fidelity", 1e-10)

    passed = fidelity >= (1.0 - tolerance)

    return ContractResult(
        name=spec.name,
        passed=passed,
        details={
            "fidelity": fidelity,
            "tolerance": tolerance,
            "method": "exact_statevector",
        },
    )


def evaluate_unitary_equivalence(
    spec: ContractSpec,
    unitary_before: np.ndarray,
    unitary_after: np.ndarray,
) -> ContractResult:
    """Check unitary matrix equivalence up to global phase.

    Uses: U1† @ U2 should be proportional to identity.
    """
    U1 = np.asarray(unitary_before, dtype=complex)
    U2 = np.asarray(unitary_after, dtype=complex)

    if U1.shape != U2.shape:
        return ContractResult(
            name=spec.name,
            passed=False,
            details={"error": "Unitary dimensions differ"},
        )

    product = U1.conj().T @ U2
    n = product.shape[0]

    # Check if proportional to identity
    # Extract phase from top-left
    phase = product[0, 0]
    if abs(phase) < 1e-15:
        return ContractResult(name=spec.name, passed=False, details={"error": "Zero phase"})

    dephased = product / phase
    identity = np.eye(n, dtype=complex)
    diff = float(np.max(np.abs(dephased - identity)))

    tolerance = spec.tolerances.get("max_diff", 1e-10)
    passed = diff <= tolerance

    return ContractResult(
        name=spec.name,
        passed=passed,
        details={
            "max_diff": diff,
            "tolerance": tolerance,
            "method": "unitary_equivalence",
            "global_phase": float(np.angle(phase)),
        },
    )
