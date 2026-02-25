"""Cirq adapter for QOCC.

Implements the ``BaseAdapter`` interface using Google Cirq.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from qocc.adapters.base import (
    BaseAdapter,
    CompileResult,
    MetricsSnapshot,
    SimulationResult,
    SimulationSpec,
    register_adapter,
)
from qocc.core.circuit_handle import BackendInfo, CircuitHandle, PassLogEntry, PipelineSpec


class CirqAdapter(BaseAdapter):
    """Adapter for Google Cirq."""

    def __init__(self) -> None:
        self._cirq = _import_cirq()

    def name(self) -> str:
        return "cirq"

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(self, source: str | Any) -> CircuitHandle:
        cirq = self._cirq

        if isinstance(source, cirq.Circuit):
            circuit = source
        elif isinstance(source, str):
            path = Path(source)
            if path.exists() and path.suffix in (".qasm", ".qasm2"):
                qasm_str = path.read_text(encoding="utf-8")
                circuit = cirq.contrib.qasm_import.circuit_from_qasm(qasm_str)
            elif path.exists() and path.suffix == ".json":
                json_str = path.read_text(encoding="utf-8")
                circuit = cirq.read_json(json_text=json_str)
            else:
                # Assume raw QASM string
                try:
                    circuit = cirq.contrib.qasm_import.circuit_from_qasm(source)
                except Exception:
                    circuit = cirq.read_json(json_text=source)
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        num_qubits = len(circuit.all_qubits())
        qasm_str = _to_qasm(cirq, circuit)

        return CircuitHandle(
            name="cirq_circuit",
            num_qubits=num_qubits,
            native_circuit=circuit,
            source_format="cirq",
            qasm3=qasm_str,
        )

    # ------------------------------------------------------------------
    # Normalize
    # ------------------------------------------------------------------

    def normalize(self, circuit: CircuitHandle) -> CircuitHandle:
        from qocc.core.canonicalize import normalize_circuit
        return normalize_circuit(circuit)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self, circuit: CircuitHandle, fmt: str = "qasm3") -> str:
        cirq = self._cirq
        native = circuit.native_circuit
        if fmt in ("qasm2", "qasm3", "qasm"):
            return _to_qasm(cirq, native)
        elif fmt == "json":
            return cirq.to_json(native)
        else:
            raise ValueError(f"Unsupported export format: {fmt}")

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------

    def compile(self, circuit: CircuitHandle, pipeline: PipelineSpec, emitter: Any = None) -> CompileResult:
        cirq = self._cirq
        native = circuit.native_circuit

        t_total = time.perf_counter()

        # Cirq compilation: optimise using built-in optimizers
        opt_level = pipeline.optimization_level
        compiled = native.copy()

        optimizers: list[Any] = []
        if opt_level >= 1:
            optimizers.append(cirq.DropEmptyMoments())
            optimizers.append(cirq.DropNegligible())
        if opt_level >= 2:
            optimizers.append(cirq.MergeSingleQubitGates())
        if opt_level >= 3:
            try:
                optimizers.append(cirq.EjectZ())
                optimizers.append(cirq.EjectPhasedPaulis())
            except AttributeError:
                pass

        pass_log: list[PassLogEntry] = []
        for i, opt in enumerate(optimizers):
            pass_name = type(opt).__name__

            # Emit a child span per pass if emitter provided
            span_ctx = emitter.span(
                f"pass/{pass_name}",
                attributes={"pass_name": pass_name, "order": i},
            ) if emitter else None

            if span_ctx:
                span_ctx.__enter__()

            t_pass = time.perf_counter()
            try:
                opt(compiled)  # type: ignore[operator]
                dt_pass = (time.perf_counter() - t_pass) * 1000.0
                pass_log.append(PassLogEntry(
                    pass_name=pass_name,
                    order=i,
                    duration_ms=dt_pass,
                ))
                if span_ctx:
                    span_ctx.__exit__(None, None, None)
            except Exception as exc:
                dt_pass = (time.perf_counter() - t_pass) * 1000.0
                pass_log.append(PassLogEntry(
                    pass_name=pass_name,
                    order=i,
                    duration_ms=dt_pass,
                    errors=[str(exc)],
                ))
                if span_ctx:
                    span_ctx.__exit__(type(exc), exc, exc.__traceback__)

        total_elapsed = (time.perf_counter() - t_total) * 1000.0
        pass_log.append(PassLogEntry(
            pass_name="cirq.optimize.total",
            order=len(pass_log),
            duration_ms=total_elapsed,
            parameters={"optimization_level": opt_level},
        ))

        qasm_str = _to_qasm(cirq, compiled)
        out = CircuitHandle(
            name=f"{circuit.name}_compiled",
            num_qubits=len(compiled.all_qubits()),
            native_circuit=compiled,
            source_format="cirq",
            qasm3=qasm_str,
        )
        return CompileResult(circuit=out, pass_log=pass_log)

    # ------------------------------------------------------------------
    # Simulate
    # ------------------------------------------------------------------

    def simulate(self, circuit: CircuitHandle, spec: SimulationSpec) -> SimulationResult:
        cirq = self._cirq
        native = circuit.native_circuit
        import numpy as np

        # ── Statevector-only mode (shots == 0) ───────────────
        if spec.shots == 0:
            rng = np.random.default_rng(spec.seed) if spec.seed is not None else np.random.default_rng()
            sim = cirq.Simulator(seed=rng)
            result = sim.simulate(native)
            sv = result.final_state_vector
            return SimulationResult(
                counts={},
                shots=0,
                seed=spec.seed,
                metadata={"statevector": sv.tolist()},
            )

        # ── Shot-based simulation ────────────────────────────
        # Add measurements if not present
        qubits = sorted(native.all_qubits())
        measured = native.copy()
        if not any(cirq.is_measurement(op) for moment in measured for op in moment):
            measured.append(cirq.measure(*qubits, key="result"))

        rng = np.random.default_rng(spec.seed) if spec.seed is not None else np.random.default_rng()
        sim = cirq.DensityMatrixSimulator(seed=rng) if spec.method == "density_matrix" else cirq.Simulator(seed=rng)

        result = sim.run(measured, repetitions=spec.shots)
        # Convert to counts dict
        counts: dict[str, int] = {}
        for _, row in result.measurements.items():
            for bits in row:
                key = "".join(str(int(b)) for b in bits)
                counts[key] = counts.get(key, 0) + 1
            break  # take first measurement key

        return SimulationResult(
            counts=counts,
            shots=spec.shots,
            seed=spec.seed,
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self, circuit: CircuitHandle) -> MetricsSnapshot:
        from qocc.metrics.compute import compute_metrics
        return compute_metrics(circuit)

    # ------------------------------------------------------------------
    # Hash
    # ------------------------------------------------------------------

    def hash(self, circuit: CircuitHandle) -> str:
        normalized = self.normalize(circuit)
        return normalized.stable_hash()

    # ------------------------------------------------------------------
    # Backend info
    # ------------------------------------------------------------------

    def describe_backend(self) -> BackendInfo:
        cirq = self._cirq
        return BackendInfo(name="cirq", version=cirq.__version__)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _import_cirq() -> Any:
    """Lazy import of Cirq."""
    try:
        import cirq  # type: ignore[import-untyped]
        return cirq
    except ImportError as exc:
        raise ImportError(
            "Cirq is required for the Cirq adapter. "
            "Install with: pip install 'qocc[cirq]'"
        ) from exc


def _to_qasm(cirq: Any, circuit: Any) -> str:
    """Export a Cirq circuit to QASM."""
    try:
        return circuit.to_qasm()
    except Exception:
        return str(circuit)


# Auto-register
register_adapter("cirq", CirqAdapter)
