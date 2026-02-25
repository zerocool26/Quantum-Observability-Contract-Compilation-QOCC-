"""Compute circuit metrics from a CircuitHandle.

Supports both Qiskit and Cirq circuits with graceful fallbacks.
All results are returned as an immutable ``MetricsSnapshot``.
"""

from __future__ import annotations

from typing import Any

from qocc.adapters.base import MetricsSnapshot
from qocc.core.circuit_handle import CircuitHandle


def compute_metrics(
    circuit: CircuitHandle,
    coupling_map: list[tuple[int, int]] | None = None,
    duration_model: dict[str, float] | None = None,
    error_model: dict[str, float] | None = None,
) -> MetricsSnapshot:
    """Compute the standard QOCC metric set for *circuit*.

    Parameters:
        circuit: The circuit to analyse.
        coupling_map: Optional hardware connectivity for topology violation count.
        duration_model: ``{gate_name: duration_ns}`` for duration estimation.
        error_model: ``{gate_name: error_rate}`` for proxy error scoring.
    """
    fmt = circuit.source_format

    if fmt == "qiskit":
        data = _metrics_qiskit(circuit, coupling_map, duration_model, error_model)
    elif fmt == "cirq":
        data = _metrics_cirq(circuit, coupling_map, duration_model, error_model)
    else:
        data = _metrics_generic(circuit)

    return MetricsSnapshot(data)


# ======================================================================
# Qiskit
# ======================================================================

def _metrics_qiskit(
    circuit: CircuitHandle,
    coupling_map: list[tuple[int, int]] | None,
    duration_model: dict[str, float] | None,
    error_model: dict[str, float] | None,
) -> dict[str, Any]:
    qc = circuit.native_circuit
    gate_counts: dict[str, int] = {}
    gates_1q = 0
    gates_2q = 0
    total_gates = 0

    for instruction in qc.data:
        op = instruction.operation
        name = op.name
        nq = op.num_qubits
        if name in ("barrier", "measure", "reset"):
            continue
        total_gates += 1
        gate_counts[name] = gate_counts.get(name, 0) + 1
        if nq == 1:
            gates_1q += 1
        elif nq >= 2:
            gates_2q += 1

    depth = qc.depth()
    width = qc.num_qubits

    # 2Q depth
    depth_2q = _two_qubit_depth_qiskit(qc)

    # Topology violations
    topo_violations = None
    if coupling_map is not None:
        topo_violations = _topology_violations_qiskit(qc, coupling_map)

    # Duration estimate
    duration_est = _duration_estimate(gate_counts, duration_model)

    # Proxy error
    proxy_error = _proxy_error(gate_counts, depth, error_model)

    return {
        "width": width,
        "total_gates": total_gates,
        "gates_1q": gates_1q,
        "gates_2q": gates_2q,
        "depth": depth,
        "depth_2q": depth_2q,
        "gate_histogram": gate_counts,
        "topology_violations": topo_violations,
        "duration_estimate": duration_est,
        "proxy_error_score": proxy_error,
    }


def _two_qubit_depth_qiskit(qc: Any) -> int:
    """Compute 2-qubit gate depth for a Qiskit circuit."""
    try:
        from qiskit.converters import circuit_to_dag  # type: ignore[import-untyped]
        dag = circuit_to_dag(qc)
        depth = dag.depth(lambda x: x.operation.num_qubits >= 2)
        return depth
    except Exception:
        return 0


def _topology_violations_qiskit(
    qc: Any, coupling_map: list[tuple[int, int]]
) -> int:
    """Count 2Q gates on non-adjacent qubits."""
    edges = set()
    for a, b in coupling_map:
        edges.add((a, b))
        edges.add((b, a))

    violations = 0
    for instruction in qc.data:
        op = instruction.operation
        if op.num_qubits >= 2 and op.name not in ("barrier",):
            qubits = instruction.qubits
            indices = [qc.find_bit(q).index for q in qubits]
            for i in range(len(indices) - 1):
                if (indices[i], indices[i + 1]) not in edges:
                    violations += 1
    return violations


# ======================================================================
# Cirq
# ======================================================================

def _metrics_cirq(
    circuit: CircuitHandle,
    coupling_map: list[tuple[int, int]] | None,
    duration_model: dict[str, float] | None,
    error_model: dict[str, float] | None,
) -> dict[str, Any]:
    try:
        import cirq  # type: ignore[import-untyped]
    except ImportError:
        return _metrics_generic(circuit)

    native = circuit.native_circuit
    qubits = sorted(native.all_qubits())
    width = len(qubits)

    gate_counts: dict[str, int] = {}
    gates_1q = 0
    gates_2q = 0
    total_gates = 0

    for moment in native:
        for op in moment:
            if cirq.is_measurement(op):
                continue
            name = str(op.gate) if op.gate else type(op).__name__
            nq = len(op.qubits)
            total_gates += 1
            gate_counts[name] = gate_counts.get(name, 0) + 1
            if nq == 1:
                gates_1q += 1
            elif nq >= 2:
                gates_2q += 1

    depth = len(native)

    # 2Q depth
    depth_2q = sum(
        1
        for m in native
        if any(len(op.qubits) >= 2 and not cirq.is_measurement(op) for op in m)
    )

    duration_est = _duration_estimate(gate_counts, duration_model)
    proxy_error = _proxy_error(gate_counts, depth, error_model)

    return {
        "width": width,
        "total_gates": total_gates,
        "gates_1q": gates_1q,
        "gates_2q": gates_2q,
        "depth": depth,
        "depth_2q": depth_2q,
        "gate_histogram": gate_counts,
        "topology_violations": None,
        "duration_estimate": duration_est,
        "proxy_error_score": proxy_error,
    }


# ======================================================================
# Generic fallback
# ======================================================================

def _metrics_generic(circuit: CircuitHandle) -> dict[str, Any]:
    """Minimal metrics from QASM string analysis."""
    return {
        "width": circuit.num_qubits,
        "total_gates": None,
        "gates_1q": None,
        "gates_2q": None,
        "depth": None,
        "depth_2q": None,
        "gate_histogram": {},
        "topology_violations": None,
        "duration_estimate": None,
        "proxy_error_score": None,
    }


# ======================================================================
# Shared helpers
# ======================================================================

def _duration_estimate(
    gate_counts: dict[str, int],
    duration_model: dict[str, float] | None,
) -> float | None:
    """duration = Σ count(gate_type) * duration(gate_type)."""
    if duration_model is None:
        return None
    total = 0.0
    for gate, count in gate_counts.items():
        dur = duration_model.get(gate, duration_model.get("default", 0.0))
        total += count * dur
    return total


def _proxy_error(
    gate_counts: dict[str, int],
    depth: int,
    error_model: dict[str, float] | None,
    decoherence_weight: float = 0.001,
) -> float | None:
    """proxy_error = Σ count(gate) * p_error(gate) + depth * decoherence_weight."""
    if error_model is None:
        return None
    total = 0.0
    for gate, count in gate_counts.items():
        err = error_model.get(gate, error_model.get("default", 0.001))
        total += count * err
    total += depth * decoherence_weight
    return total
