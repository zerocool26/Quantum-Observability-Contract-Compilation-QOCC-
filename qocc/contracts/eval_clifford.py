"""Clifford/stabilizer contract evaluation.

For Clifford circuits, verify exact equivalence efficiently using
stabilizer tableau invariants.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qocc.contracts.spec import ContractResult, ContractSpec


def is_clifford_circuit(circuit_handle: Any) -> bool:
    """Detect whether a circuit is Clifford (heuristic).

    Checks if all gates in the circuit are from the Clifford gate set:
    {H, S, CNOT, CZ, X, Y, Z, SWAP, Sdg}.
    """
    CLIFFORD_GATES = {
        "h", "s", "sdg", "x", "y", "z",
        "cx", "cnot", "cz", "swap",
        "id", "barrier", "measure", "reset",
        # Qiskit names
        "sx", "sxdg",
    }

    fmt = circuit_handle.source_format

    if fmt == "qiskit":
        qc = circuit_handle.native_circuit
        for instruction in qc.data:
            name = instruction.operation.name.lower()
            if name not in CLIFFORD_GATES:
                return False
        return True
    elif fmt == "cirq":
        # Heuristic: check gate names
        native = circuit_handle.native_circuit
        for moment in native:
            for op in moment:
                gate_name = str(op.gate).lower() if op.gate else ""
                if not any(c in gate_name for c in ["h", "s", "x", "y", "z", "cx", "cnot", "cz", "swap"]):
                    return False
        return True

    return False


def evaluate_clifford_contract(
    spec: ContractSpec,
    circuit_before: Any,
    circuit_after: Any,
) -> ContractResult:
    """Evaluate Clifford equivalence contract.

    If both circuits are Clifford, attempt exact stabilizer comparison.
    Otherwise, fall back to a note that sampling should be used.
    """
    before_clifford = is_clifford_circuit(circuit_before)
    after_clifford = is_clifford_circuit(circuit_after)

    if not before_clifford or not after_clifford:
        return ContractResult(
            name=spec.name,
            passed=False,
            details={
                "method": "clifford_check",
                "before_is_clifford": before_clifford,
                "after_is_clifford": after_clifford,
                "note": "Non-Clifford circuit detected. Use distribution/observable contract instead.",
            },
        )

    # Try Qiskit's Clifford class if available
    try:
        from qiskit.quantum_info import Clifford  # type: ignore[import-untyped]

        cliff_before = Clifford(circuit_before.native_circuit)
        cliff_after = Clifford(circuit_after.native_circuit)

        equivalent = cliff_before == cliff_after

        return ContractResult(
            name=spec.name,
            passed=equivalent,
            details={
                "method": "stabilizer_tableau",
                "equivalent": equivalent,
                "before_is_clifford": True,
                "after_is_clifford": True,
            },
        )
    except (ImportError, Exception) as exc:
        # Fall back to hash comparison
        hash_before = circuit_before.stable_hash()
        hash_after = circuit_after.stable_hash()
        equivalent = hash_before == hash_after

        return ContractResult(
            name=spec.name,
            passed=equivalent,
            details={
                "method": "hash_comparison",
                "equivalent": equivalent,
                "hash_before": hash_before,
                "hash_after": hash_after,
                "fallback_reason": str(exc),
            },
        )
