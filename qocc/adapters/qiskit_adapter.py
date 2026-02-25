"""Qiskit adapter for QOCC.

Implements the ``BaseAdapter`` interface using Qiskit ≥ 1.0.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from qocc import DEFAULT_SEED
from qocc.adapters.base import (
    BaseAdapter,
    CompileResult,
    MetricsSnapshot,
    SimulationResult,
    SimulationSpec,
    register_adapter,
)
from qocc.core.circuit_handle import BackendInfo, CircuitHandle, PassLogEntry, PipelineSpec


class QiskitAdapter(BaseAdapter):
    """Adapter for IBM Qiskit."""

    def __init__(self) -> None:
        self._qiskit = _import_qiskit()

    def name(self) -> str:
        return "qiskit"

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(self, source: str | Any) -> CircuitHandle:
        qk = self._qiskit
        QuantumCircuit = qk["QuantumCircuit"]

        if isinstance(source, QuantumCircuit):
            qc = source
        elif isinstance(source, str):
            path = Path(source)
            if path.exists() and path.suffix in (".qasm", ".qasm3"):
                qasm_str = path.read_text(encoding="utf-8")
                qc = _from_qasm(qk, qasm_str)
            else:
                # assume raw QASM string
                qc = _from_qasm(qk, source)
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        qasm3_str = _to_qasm3(qk, qc)
        return CircuitHandle(
            name=qc.name or "unnamed",
            num_qubits=qc.num_qubits,
            native_circuit=qc,
            source_format="qiskit",
            qasm3=qasm3_str,
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
        qk = self._qiskit
        qc = circuit.native_circuit
        if fmt == "qasm3":
            return _to_qasm3(qk, qc)
        elif fmt == "qasm2":
            from qiskit.qasm2 import dumps as qasm2_dumps  # type: ignore[import-untyped]
            return qasm2_dumps(qc)
        else:
            raise ValueError(f"Unsupported export format: {fmt}")

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------

    def compile(self, circuit: CircuitHandle, pipeline: PipelineSpec, emitter: Any = None) -> CompileResult:
        qk = self._qiskit
        transpile = qk["transpile"]
        qc = circuit.native_circuit

        pass_log: list[PassLogEntry] = []

        # ── Pre-compilation metrics snapshot ─────────────────
        t_total_start = time.perf_counter()

        opt_level = pipeline.optimization_level
        seed = pipeline.parameters.get("seed", DEFAULT_SEED)
        extra_params = {k: v for k, v in pipeline.parameters.items() if k != "seed"}

        # Attempt to use staged transpilation for per-pass logging
        try:
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager  # type: ignore
            from qiskit.transpiler import PassManager  # type: ignore
            import tracemalloc as _tm

            pm = generate_preset_pass_manager(
                optimization_level=opt_level,
                seed_transpiler=seed,
                **extra_params,
            )

            # Run transpile stages individually for granular logging
            intermediate = qc
            stage_names = ["init", "layout", "routing", "translation", "optimization", "scheduling"]

            for i, stage_name in enumerate(stage_names):
                stage_pm = getattr(pm, stage_name, None)
                if stage_pm is None:
                    continue

                # Emit a child span per stage if emitter provided
                span_ctx = emitter.span(
                    f"pass/qiskit.{stage_name}",
                    attributes={
                        "stage": stage_name,
                        "optimization_level": opt_level,
                        "order": i,
                    },
                ) if emitter else None

                if span_ctx:
                    span_ctx.__enter__()

                # Track memory usage
                mem_before = 0
                try:
                    _tm.start()
                except RuntimeError:
                    pass  # already tracing

                t_stage = time.perf_counter()
                try:
                    snapshot_before = _tm.take_snapshot()
                    intermediate = stage_pm.run(intermediate)
                    snapshot_after = _tm.take_snapshot()
                    dt_stage = (time.perf_counter() - t_stage) * 1000.0

                    # Compute memory delta
                    mem_bytes: int | None = None
                    try:
                        stats = snapshot_after.compare_to(snapshot_before, "lineno")
                        mem_bytes = sum(s.size_diff for s in stats if s.size_diff > 0)
                    except Exception:
                        pass

                    pass_log.append(PassLogEntry(
                        pass_name=f"qiskit.{stage_name}",
                        parameters={"optimization_level": opt_level},
                        order=i,
                        duration_ms=dt_stage,
                        memory_bytes=mem_bytes,
                    ))
                    if span_ctx:
                        span_ctx.__exit__(None, None, None)
                except Exception as exc:
                    dt_stage = (time.perf_counter() - t_stage) * 1000.0
                    pass_log.append(PassLogEntry(
                        pass_name=f"qiskit.{stage_name}",
                        parameters={"optimization_level": opt_level},
                        order=i,
                        duration_ms=dt_stage,
                        errors=[str(exc)],
                    ))
                    if span_ctx:
                        span_ctx.__exit__(type(exc), exc, exc.__traceback__)
                finally:
                    try:
                        _tm.stop()
                    except RuntimeError:
                        pass

            compiled = intermediate

        except (ImportError, AttributeError, Exception):
            # Fall back to monolithic transpile
            t0 = time.perf_counter()
            compiled = transpile(
                qc,
                optimization_level=opt_level,
                seed_transpiler=seed,
                **extra_params,
            )
            elapsed = (time.perf_counter() - t0) * 1000

            pass_log = [
                PassLogEntry(
                    pass_name="qiskit.transpile",
                    parameters={
                        "optimization_level": opt_level,
                        **pipeline.parameters,
                    },
                    order=0,
                    duration_ms=elapsed,
                )
            ]

        total_elapsed = (time.perf_counter() - t_total_start) * 1000.0

        # Add a summary entry
        pass_log.append(PassLogEntry(
            pass_name="qiskit.transpile.total",
            parameters={"optimization_level": opt_level, **pipeline.parameters},
            order=len(pass_log),
            duration_ms=total_elapsed,
        ))

        qasm3_str = _to_qasm3(qk, compiled)
        out = CircuitHandle(
            name=f"{circuit.name}_compiled",
            num_qubits=compiled.num_qubits,
            native_circuit=compiled,
            source_format="qiskit",
            qasm3=qasm3_str,
        )
        return CompileResult(circuit=out, pass_log=pass_log)

    # ------------------------------------------------------------------
    # Simulate
    # ------------------------------------------------------------------

    def simulate(self, circuit: CircuitHandle, spec: SimulationSpec) -> SimulationResult:
        from qiskit.quantum_info import Statevector  # type: ignore[import-untyped]

        # ── Statevector-only mode (shots == 0) ───────────────
        if spec.shots == 0 or spec.method == "statevector" and spec.shots == 0:
            qc = circuit.native_circuit
            sv = Statevector.from_instruction(qc)
            return SimulationResult(
                counts={},
                shots=0,
                seed=spec.seed,
                metadata={"statevector": sv.data.tolist()},
            )

        # ── Shot-based simulation ────────────────────────────
        try:
            from qiskit_aer import AerSimulator  # type: ignore[import-untyped]
        except ImportError:
            # Fall back to Qiskit's built-in statevector sampler
            qc = circuit.native_circuit
            sv = Statevector.from_instruction(qc)
            counts = sv.sample_counts(spec.shots, seed=spec.seed)
            return SimulationResult(
                counts={k: int(v) for k, v in counts.items()},
                shots=spec.shots,
                seed=spec.seed,
            )

        backend = AerSimulator(method=spec.method)
        qc = circuit.native_circuit.copy()
        qc.measure_all()
        from qiskit import transpile  # type: ignore[import-untyped]

        qc = transpile(qc, backend)
        job = backend.run(qc, shots=spec.shots, seed_simulator=spec.seed)
        result = job.result()
        counts = result.get_counts()
        return SimulationResult(
            counts={k: int(v) for k, v in counts.items()},
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
        import qiskit  # type: ignore[import-untyped]
        return BackendInfo(name="qiskit", version=qiskit.__version__)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _import_qiskit() -> dict[str, Any]:
    """Lazy import of Qiskit components."""
    try:
        from qiskit import QuantumCircuit, transpile  # type: ignore[import-untyped]
        return {"QuantumCircuit": QuantumCircuit, "transpile": transpile}
    except ImportError as exc:
        raise ImportError(
            "Qiskit is required for the Qiskit adapter. "
            "Install with: pip install 'qocc[qiskit]'"
        ) from exc


def _from_qasm(qk: dict[str, Any], qasm_str: str) -> Any:
    """Parse a QASM string into a Qiskit QuantumCircuit."""
    QuantumCircuit = qk["QuantumCircuit"]

    # Try QASM3 first, fall back to QASM2
    try:
        from qiskit.qasm3 import loads as qasm3_loads  # type: ignore[import-untyped]
        return qasm3_loads(qasm_str)
    except Exception:
        pass
    try:
        return QuantumCircuit.from_qasm_str(qasm_str)
    except Exception:
        pass
    # Last resort — try qasm2 module
    try:
        from qiskit.qasm2 import loads as qasm2_loads  # type: ignore[import-untyped]
        return qasm2_loads(qasm_str)
    except Exception as exc:
        raise ValueError(f"Cannot parse QASM string with Qiskit: {exc}") from exc


def _to_qasm3(qk: dict[str, Any], qc: Any) -> str:
    """Export a QuantumCircuit to OpenQASM 3."""
    try:
        from qiskit.qasm3 import dumps as qasm3_dumps  # type: ignore[import-untyped]
        return qasm3_dumps(qc)
    except Exception:
        # Fallback to QASM2 if QASM3 export fails
        try:
            return qc.qasm()
        except Exception:
            return repr(qc)


# Auto-register
register_adapter("qiskit", QiskitAdapter)
