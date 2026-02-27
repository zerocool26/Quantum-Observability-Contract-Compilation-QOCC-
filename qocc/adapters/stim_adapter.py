"""Stim adapter for QOCC.

Implements the ``BaseAdapter`` interface for QEC-oriented workflows.
"""

from __future__ import annotations

import hashlib
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


class StimAdapter(BaseAdapter):
    """Adapter for Stim circuits and QEC sampling/decoding."""

    def __init__(self) -> None:
        self._stim = _import_stim()

    def name(self) -> str:
        return "stim"

    def ingest(self, source: str | Any) -> CircuitHandle:
        stim = self._stim
        CircuitCls = stim["Circuit"]

        if isinstance(source, CircuitCls):
            circuit = source
        elif isinstance(source, str):
            p = Path(source)
            if p.exists():
                if p.suffix.lower() != ".stim":
                    raise ValueError(f"Unsupported Stim input file type: {p.suffix}")
                text = p.read_text(encoding="utf-8")
                circuit = CircuitCls(text)
            else:
                circuit = CircuitCls(source)
        else:
            raise TypeError(f"Unsupported source type for stim ingest: {type(source)}")

        stim_text = str(circuit)
        return CircuitHandle(
            name="stim_circuit",
            num_qubits=_estimate_qubits(circuit),
            native_circuit=circuit,
            source_format="stim",
            qasm3=None,
            metadata={"framework": "stim", "stim_text": stim_text},
        )

    def normalize(self, circuit: CircuitHandle) -> CircuitHandle:
        # Stim circuits are treated as already canonical in text/instruction order.
        native = _copy_circuit(circuit.native_circuit)
        return CircuitHandle(
            name=f"{circuit.name}_normalized",
            num_qubits=_estimate_qubits(native),
            native_circuit=native,
            source_format="stim",
            qasm3=None,
            metadata={**circuit.metadata, "normalized": True},
            _normalized=True,
        )

    def export(self, circuit: CircuitHandle, fmt: str = "stim") -> str:
        if fmt not in ("stim", "text"):
            raise ValueError(f"Unsupported export format for stim: {fmt}")
        return str(circuit.native_circuit)

    def compile(
        self,
        circuit: CircuitHandle,
        pipeline: PipelineSpec,
        emitter: Any | None = None,
    ) -> CompileResult:
        """Compile by generating detector error model and mapping metadata."""
        native = _copy_circuit(circuit.native_circuit)
        pass_log: list[PassLogEntry] = []

        dem_text = ""
        t0 = time.perf_counter()
        dem_span = emitter.span("pass/stim.detector_error_model", attributes={"stage": "dem_generation"}) if emitter else None
        if dem_span:
            dem_span.__enter__()
        try:
            if hasattr(native, "detector_error_model"):
                dem = native.detector_error_model()
                dem_text = str(dem)
            dt = (time.perf_counter() - t0) * 1000.0
            pass_log.append(
                PassLogEntry(
                    pass_name="stim.detector_error_model",
                    order=0,
                    duration_ms=dt,
                    parameters={"dem_chars": len(dem_text)},
                )
            )
            if dem_span:
                dem_span.__exit__(None, None, None)
        except Exception as exc:
            dt = (time.perf_counter() - t0) * 1000.0
            pass_log.append(
                PassLogEntry(
                    pass_name="stim.detector_error_model",
                    order=0,
                    duration_ms=dt,
                    errors=[str(exc)],
                )
            )
            if dem_span:
                dem_span.__exit__(type(exc), exc, exc.__traceback__)

        mapping = pipeline.parameters.get("logical_physical_mapping", {})
        map_span = emitter.span("pass/stim.logical_physical_mapping", attributes={"mapping_size": len(mapping)}) if emitter else None
        if map_span:
            map_span.__enter__()
            map_span.__exit__(None, None, None)
        pass_log.append(
            PassLogEntry(
                pass_name="stim.logical_physical_mapping",
                order=1,
                parameters={"mapping": mapping},
            )
        )

        out = CircuitHandle(
            name=f"{circuit.name}_compiled",
            num_qubits=_estimate_qubits(native),
            native_circuit=native,
            source_format="stim",
            qasm3=None,
            metadata={
                **circuit.metadata,
                "dem": dem_text,
                "decoder_stats": {},
                "logical_error_rates": {},
            },
        )
        return CompileResult(circuit=out, pass_log=pass_log)

    def simulate(self, circuit: CircuitHandle, spec: SimulationSpec) -> SimulationResult:
        stim = self._stim
        native = circuit.native_circuit

        if spec.shots == 0:
            if "TableauSimulator" in stim:
                _ = stim["TableauSimulator"]()
            return SimulationResult(
                counts={},
                shots=0,
                seed=spec.seed,
                metadata={"simulator": "tableau", "logical_error_rate": 0.0},
            )

        sample_rows = _sample_rows(native, spec.shots)
        counts = _rows_to_counts(sample_rows)
        syndrome_dist = _syndrome_weight_distribution(sample_rows)
        logical_errors = sum(1 for row in sample_rows if (sum(int(v) for v in row) % 2) == 1)
        logical_error_rate = (logical_errors / spec.shots) if spec.shots > 0 else 0.0

        decoder_stats: dict[str, Any] = {
            "decoder_rounds": int(spec.extra.get("decoder_rounds", 1)),
            "matching_graph_edges": None,
            "matching_graph_nodes": None,
            "logical_errors": logical_errors,
        }

        decoded_errors = logical_errors
        try:
            import pymatching  # type: ignore[import-untyped]

            dem = native.detector_error_model() if hasattr(native, "detector_error_model") else None
            if dem is not None and hasattr(pymatching, "Matching"):
                matching = pymatching.Matching.from_detector_error_model(str(dem))
                decoder_stats["matching_graph_edges"] = int(getattr(matching, "num_edges", 0))
                decoder_stats["matching_graph_nodes"] = int(getattr(matching, "num_nodes", 0))
                if hasattr(matching, "decode_batch"):
                    _ = matching.decode_batch(sample_rows)
        except Exception:
            pass

        sinter_stats: dict[str, Any] | None = None
        if spec.method.lower() == "sinter":
            try:
                import sinter  # type: ignore[import-untyped]

                if hasattr(sinter, "collect"):
                    collected = sinter.collect(num_shots=spec.shots)
                    sinter_stats = {
                        "shots": int(getattr(collected, "shots", spec.shots)),
                        "errors": int(getattr(collected, "errors", logical_errors)),
                        "discards": int(getattr(collected, "discards", 0)),
                        "seconds": float(getattr(collected, "seconds", 0.0)),
                        "custom_counts": getattr(collected, "custom_counts", {}),
                    }
            except Exception:
                sinter_stats = None

        metadata = {
            "logical_error_rate": float(logical_error_rate),
            "logical_error_rates": {
                "logical_error_rate": float(logical_error_rate),
                "shots": int(spec.shots),
                "logical_errors": int(decoded_errors),
            },
            "syndrome_weight_distribution": syndrome_dist,
            "decoder_stats": decoder_stats,
        }
        if sinter_stats is not None:
            metadata["sinter"] = sinter_stats

        return SimulationResult(
            counts=counts,
            shots=spec.shots,
            seed=spec.seed,
            metadata=metadata,
        )

    def get_metrics(self, circuit: CircuitHandle) -> MetricsSnapshot:
        instructions = list(_iter_instructions(circuit.native_circuit))
        gate_hist: dict[str, int] = {}
        gates_2q = 0
        for inst in instructions:
            name = _instruction_name(inst)
            gate_hist[name] = gate_hist.get(name, 0) + 1
            if _instruction_arity(inst) >= 2:
                gates_2q += 1
        total = len(instructions)
        return MetricsSnapshot(
            {
                "width": _estimate_qubits(circuit.native_circuit),
                "total_gates": total,
                "gates_1q": total - gates_2q,
                "gates_2q": gates_2q,
                "depth": total,
                "depth_2q": gates_2q,
                "gate_histogram": gate_hist,
                "topology_violations": None,
                "duration_estimate": None,
                "proxy_error_score": None,
            }
        )

    def hash(self, circuit: CircuitHandle) -> str:
        payload = str(circuit.native_circuit).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def describe_backend(self) -> BackendInfo:
        versions: dict[str, str] = {"stim": str(self._stim.get("version", "unknown"))}
        for pkg in ("pymatching", "sinter"):
            try:
                mod = __import__(pkg)
                versions[pkg] = str(getattr(mod, "__version__", "installed"))
            except Exception:
                versions[pkg] = "not_installed"
        return BackendInfo(name="stim", version=versions["stim"], extra=versions)


def _import_stim() -> dict[str, Any]:
    try:
        import stim  # type: ignore[import-untyped]

        return {
            "module": stim,
            "version": getattr(stim, "__version__", "unknown"),
            "Circuit": stim.Circuit,
            "TableauSimulator": getattr(stim, "TableauSimulator", None),
        }
    except ImportError as exc:
        raise ImportError("stim is required for the stim adapter. Install with: pip install 'qocc[stim]'") from exc


def _copy_circuit(circuit: Any) -> Any:
    if hasattr(circuit, "copy"):
        return circuit.copy()
    return type(circuit)(str(circuit))


def _iter_instructions(circuit: Any) -> list[Any]:
    try:
        return list(circuit)
    except Exception:
        return []


def _instruction_name(inst: Any) -> str:
    if hasattr(inst, "name"):
        return str(inst.name)
    if hasattr(inst, "gate"):
        return str(inst.gate)
    text = str(inst)
    return text.split()[0] if text else "UNKNOWN"


def _instruction_arity(inst: Any) -> int:
    if hasattr(inst, "targets_copy"):
        try:
            return len(list(inst.targets_copy()))
        except Exception:
            pass
    if hasattr(inst, "targets"):
        try:
            return len(list(inst.targets))
        except Exception:
            pass
    if hasattr(inst, "qubits"):
        try:
            return len(list(inst.qubits))
        except Exception:
            pass
    return 1


def _estimate_qubits(circuit: Any) -> int:
    for attr in ("num_qubits", "n_qubits"):
        if hasattr(circuit, attr):
            try:
                return int(getattr(circuit, attr))
            except Exception:
                pass

    max_index = -1
    for inst in _iter_instructions(circuit):
        for t in _targets(inst):
            try:
                max_index = max(max_index, int(t))
            except Exception:
                continue
    return max_index + 1 if max_index >= 0 else 0


def _targets(inst: Any) -> list[Any]:
    if hasattr(inst, "targets_copy"):
        try:
            return list(inst.targets_copy())
        except Exception:
            return []
    if hasattr(inst, "targets"):
        try:
            return list(inst.targets)
        except Exception:
            return []
    return []


def _sample_rows(circuit: Any, shots: int) -> list[list[int]]:
    if hasattr(circuit, "compile_sampler"):
        sampler = circuit.compile_sampler()
        sampled = sampler.sample(shots=shots)
        rows = []
        for row in sampled:
            rows.append([int(v) for v in row])
        return rows
    return [[0] for _ in range(shots)]


def _rows_to_counts(rows: list[list[int]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = "".join(str(int(v)) for v in row)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _syndrome_weight_distribution(rows: list[list[int]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        weight = str(sum(int(v) for v in row))
        out[weight] = out.get(weight, 0) + 1
    return out


register_adapter("stim", StimAdapter)
