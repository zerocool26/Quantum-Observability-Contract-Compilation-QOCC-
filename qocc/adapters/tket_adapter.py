"""pytket adapter for QOCC.

Implements the ``BaseAdapter`` interface using pytket.
"""

from __future__ import annotations

import hashlib
import json
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


class TketAdapter(BaseAdapter):
    """Adapter for pytket circuits and compilation passes."""

    def __init__(self) -> None:
        self._tk = _import_tket()

    def name(self) -> str:
        return "tket"

    def ingest(self, source: str | Any) -> CircuitHandle:
        """Ingest from .qasm path, pytket.Circuit object, JSON file, or raw string."""
        tk = self._tk
        CircuitCls = tk["Circuit"]

        circuit: Any
        if isinstance(source, CircuitCls):
            circuit = source
        elif isinstance(source, str):
            p = Path(source)
            if p.exists():
                text = p.read_text(encoding="utf-8")
                if p.suffix.lower() == ".qasm":
                    circuit = _circuit_from_qasm(tk, text)
                elif p.suffix.lower() == ".json":
                    circuit = _circuit_from_json(tk, text)
                else:
                    raise ValueError(f"Unsupported tket input file type: {p.suffix}")
            else:
                stripped = source.strip()
                if stripped.startswith("{"):
                    circuit = _circuit_from_json(tk, source)
                else:
                    circuit = _circuit_from_qasm(tk, source)
        else:
            raise TypeError(f"Unsupported source type for tket ingest: {type(source)}")

        qasm = _circuit_to_qasm(tk, circuit)
        return CircuitHandle(
            name=getattr(circuit, "name", "tket_circuit") or "tket_circuit",
            num_qubits=_num_qubits(circuit),
            native_circuit=circuit,
            source_format="tket",
            qasm3=qasm,
            metadata={"framework": "pytket"},
        )

    def normalize(self, circuit: CircuitHandle) -> CircuitHandle:
        """Canonical normalization via RemoveRedundancies + CommuteThroughMultis."""
        tk = self._tk
        native = _copy_circuit(circuit.native_circuit)
        seq = tk["SequencePass"]([
            tk["RemoveRedundancies"](),
            tk["CommuteThroughMultis"](),
        ])
        seq.apply(native)

        qasm = _circuit_to_qasm(tk, native)
        return CircuitHandle(
            name=f"{circuit.name}_normalized",
            num_qubits=_num_qubits(native),
            native_circuit=native,
            source_format="tket",
            qasm3=qasm,
            metadata={**circuit.metadata, "framework": "pytket", "normalized": True},
            _normalized=True,
        )

    def export(self, circuit: CircuitHandle, fmt: str = "qasm3") -> str:
        tk = self._tk
        if fmt in ("qasm", "qasm2", "qasm3"):
            return _circuit_to_qasm(tk, circuit.native_circuit)
        if fmt == "json":
            d = circuit.native_circuit.to_dict()
            return json.dumps(d, sort_keys=True)
        raise ValueError(f"Unsupported export format for tket: {fmt}")

    def compile(
        self,
        circuit: CircuitHandle,
        pipeline: PipelineSpec,
        emitter: Any | None = None,
    ) -> CompileResult:
        """Compile with a pass sequence, emitting one span per pass."""
        tk = self._tk
        native = _copy_circuit(circuit.native_circuit)
        pass_sequence = _resolve_pass_sequence(tk, pipeline)

        pass_log: list[PassLogEntry] = []
        for order, pass_obj in enumerate(pass_sequence):
            pass_name = type(pass_obj).__name__
            before = _extract_metrics(native)
            t0 = time.perf_counter()

            span_ctx = (
                emitter.span(
                    f"pass/tket.{pass_name}",
                    attributes={"pass_name": pass_name, "order": order},
                )
                if emitter
                else None
            )

            if span_ctx:
                span_ctx.__enter__()

            try:
                pass_obj.apply(native)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                after = _extract_metrics(native)
                delta = _metrics_delta(before, after)
                pass_log.append(
                    PassLogEntry(
                        pass_name=f"tket.{pass_name}",
                        order=order,
                        duration_ms=dt_ms,
                        parameters={"metrics_delta": delta},
                    )
                )
                if span_ctx:
                    span_ctx.__exit__(None, None, None)
            except Exception as exc:
                dt_ms = (time.perf_counter() - t0) * 1000.0
                pass_log.append(
                    PassLogEntry(
                        pass_name=f"tket.{pass_name}",
                        order=order,
                        duration_ms=dt_ms,
                        errors=[str(exc)],
                    )
                )
                if span_ctx:
                    span_ctx.__exit__(type(exc), exc, exc.__traceback__)

        qasm = _circuit_to_qasm(tk, native)
        out = CircuitHandle(
            name=f"{circuit.name}_compiled",
            num_qubits=_num_qubits(native),
            native_circuit=native,
            source_format="tket",
            qasm3=qasm,
            metadata={**circuit.metadata, "framework": "pytket"},
        )
        return CompileResult(circuit=out, pass_log=pass_log)

    def simulate(self, circuit: CircuitHandle, spec: SimulationSpec) -> SimulationResult:
        """Simulate via pytket extension backends when available."""
        backend, backend_name = _select_sim_backend(spec.method)
        if backend is None:
            raise ImportError(
                "pytket simulation extensions are required for tket simulate(). "
                "Install with: pip install 'qocc[tket]' and one of pytket-qulacs/pytket-projectq."
            )

        native = circuit.native_circuit
        if spec.shots == 0:
            if hasattr(backend, "get_state"):
                state = backend.get_state(native)
                return SimulationResult(
                    counts={},
                    shots=0,
                    seed=spec.seed,
                    metadata={"statevector": list(state), "backend": backend_name},
                )
            raise NotImplementedError(f"Selected backend {backend_name} does not provide get_state().")

        if hasattr(backend, "get_counts"):
            counts = backend.get_counts(native, n_shots=spec.shots, seed=spec.seed)
        elif hasattr(backend, "run_circuit"):
            run_result = backend.run_circuit(native, n_shots=spec.shots, seed=spec.seed)
            if isinstance(run_result, dict):
                counts = run_result
            elif hasattr(run_result, "get_counts"):
                counts = run_result.get_counts()
            else:
                raise ValueError("Unsupported simulation result object returned by tket backend.")
        else:
            raise NotImplementedError(f"Selected backend {backend_name} cannot sample counts.")

        normalized_counts: dict[str, int] = {}
        for key, value in counts.items():
            bitstring = "".join(str(int(bit)) for bit in key) if not isinstance(key, str) else key
            normalized_counts[bitstring] = int(value)

        return SimulationResult(
            counts=normalized_counts,
            shots=spec.shots,
            seed=spec.seed,
            metadata={"backend": backend_name},
        )

    def get_metrics(self, circuit: CircuitHandle) -> MetricsSnapshot:
        return MetricsSnapshot(_extract_metrics(circuit.native_circuit))

    def hash(self, circuit: CircuitHandle) -> str:
        """Deterministic hash from canonical JSON serialization."""
        payload = json.dumps(circuit.native_circuit.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def describe_backend(self) -> BackendInfo:
        tk = self._tk
        version = tk.get("version", "unknown")
        extension_name = _active_extension_name()
        default_passes = ["RemoveRedundancies", "CommuteThroughMultis"]
        pass_set_hash = hashlib.sha256("|".join(default_passes).encode("utf-8")).hexdigest()[:16]
        return BackendInfo(
            name="tket",
            version=str(version),
            extra={
                "active_extension": extension_name,
                "pass_set_hash": pass_set_hash,
            },
        )


def _import_tket() -> dict[str, Any]:
    try:
        import pytket  # type: ignore[import-untyped]
        from pytket import Circuit  # type: ignore[import-untyped]
        from pytket.passes import CommuteThroughMultis, RemoveRedundancies, SequencePass  # type: ignore[import-untyped]
        return {
            "module": pytket,
            "version": getattr(pytket, "__version__", "unknown"),
            "Circuit": Circuit,
            "SequencePass": SequencePass,
            "RemoveRedundancies": RemoveRedundancies,
            "CommuteThroughMultis": CommuteThroughMultis,
        }
    except ImportError as exc:
        raise ImportError(
            "pytket is required for the tket adapter. Install with: pip install 'qocc[tket]'"
        ) from exc


def _circuit_from_qasm(tk: dict[str, Any], qasm: str) -> Any:
    CircuitCls = tk["Circuit"]
    if hasattr(CircuitCls, "from_qasm"):
        return CircuitCls.from_qasm(qasm)
    try:
        from pytket.qasm import circuit_from_qasm_str  # type: ignore[import-untyped]

        return circuit_from_qasm_str(qasm)
    except ImportError as exc:
        raise ImportError(
            "QASM ingest for tket requires pytket.qasm support."
        ) from exc


def _circuit_from_json(tk: dict[str, Any], payload: str) -> Any:
    data = json.loads(payload)
    CircuitCls = tk["Circuit"]
    if hasattr(CircuitCls, "from_dict"):
        return CircuitCls.from_dict(data)
    raise ValueError("This pytket build does not support Circuit.from_dict().")


def _circuit_to_qasm(tk: dict[str, Any], circuit: Any) -> str:
    if hasattr(circuit, "to_qasm"):
        return str(circuit.to_qasm())
    try:
        from pytket.qasm import circuit_to_qasm_str  # type: ignore[import-untyped]

        return str(circuit_to_qasm_str(circuit))
    except Exception:
        return json.dumps(circuit.to_dict(), sort_keys=True)


def _num_qubits(circuit: Any) -> int:
    if hasattr(circuit, "n_qubits"):
        return int(circuit.n_qubits)
    if hasattr(circuit, "qubits"):
        return len(list(circuit.qubits))
    return 0


def _copy_circuit(circuit: Any) -> Any:
    if hasattr(circuit, "copy"):
        return circuit.copy()
    if hasattr(circuit, "to_dict") and hasattr(type(circuit), "from_dict"):
        return type(circuit).from_dict(circuit.to_dict())
    raise TypeError("Cannot copy tket circuit object.")


def _extract_metrics(circuit: Any) -> dict[str, Any]:
    gate_histogram: dict[str, int] = {}
    gates_2q = 0

    commands = list(circuit.get_commands()) if hasattr(circuit, "get_commands") else []
    for cmd in commands:
        op = getattr(cmd, "op", None)
        op_name = getattr(op, "type", None)
        if op_name is None and op is not None:
            op_name = getattr(op, "name", None)
        op_str = str(op_name) if op_name is not None else str(op)
        gate_histogram[op_str] = gate_histogram.get(op_str, 0) + 1

        qubit_args = getattr(cmd, "qubits", None)
        if qubit_args is None:
            qubit_args = getattr(cmd, "args", [])
        if len(list(qubit_args)) >= 2:
            gates_2q += 1

    depth = int(circuit.depth()) if hasattr(circuit, "depth") else 0
    two_qubit_depth = int(circuit.depth_2q()) if hasattr(circuit, "depth_2q") else None

    return {
        "width": _num_qubits(circuit),
        "total_gates": len(commands),
        "gates_2q": gates_2q,
        "depth": depth,
        "two_qubit_depth": two_qubit_depth,
        "gate_histogram": gate_histogram,
    }


def _metrics_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for key in ("width", "total_gates", "gates_2q", "depth", "two_qubit_depth"):
        b = before.get(key)
        a = after.get(key)
        if isinstance(b, int) and isinstance(a, int):
            delta[key] = a - b
    return delta


def _resolve_pass_sequence(tk: dict[str, Any], pipeline: PipelineSpec) -> list[Any]:
    mapping: dict[str, Any] = {
        "RemoveRedundancies": tk["RemoveRedundancies"],
        "CommuteThroughMultis": tk["CommuteThroughMultis"],
    }

    requested = pipeline.parameters.get("pass_sequence")
    if isinstance(requested, list) and requested:
        pass_names = [str(x) for x in requested]
    else:
        pass_names = ["RemoveRedundancies", "CommuteThroughMultis"]

    out: list[Any] = []
    for name in pass_names:
        cls = mapping.get(name)
        if cls is not None:
            out.append(cls())
    if not out:
        out = [tk["RemoveRedundancies"](), tk["CommuteThroughMultis"]()]
    return out


def _select_sim_backend(method: str) -> tuple[Any | None, str]:
    lower = method.lower()
    if lower in ("statevector", "shots", "qulacs"):
        try:
            from pytket.extensions.qulacs import QulacsBackend  # type: ignore[import-untyped]

            return QulacsBackend(), "qulacs"
        except Exception:
            pass

    try:
        from pytket.extensions.projectq import ProjectQBackend  # type: ignore[import-untyped]

        return ProjectQBackend(), "projectq"
    except Exception:
        return None, "none"


def _active_extension_name() -> str:
    backend, name = _select_sim_backend("statevector")
    return name


register_adapter("tket", TketAdapter)
