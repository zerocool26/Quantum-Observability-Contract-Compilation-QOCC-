"""IBM Quantum Runtime adapter for QOCC.

Adds hardware execution support on top of the Qiskit adapter via
``qiskit_ibm_runtime`` primitives.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from qocc import DEFAULT_SEED
from qocc.adapters.base import ExecutionResult, register_adapter
from qocc.adapters.qiskit_adapter import QiskitAdapter, _import_qiskit
from qocc.core.circuit_handle import CircuitHandle


class IBMAdapter(QiskitAdapter):
    """Qiskit Runtime adapter with real-hardware ``execute()`` support."""

    def __init__(self) -> None:
        self._qiskit = _import_qiskit()
        self._runtime = _import_ibm_runtime()

    def name(self) -> str:
        return "ibm"

    def execute(
        self,
        circuit: CircuitHandle,
        backend_spec: dict[str, Any],
        shots: int = 1024,
        emitter: Any | None = None,
    ) -> ExecutionResult:
        """Submit a circuit to IBM Runtime and return hardware execution data."""
        service = _build_service(self._runtime, backend_spec)
        backend_name = str(backend_spec.get("backend_name") or backend_spec.get("backend") or "")
        if not backend_name:
            raise ValueError("backend_spec must include 'backend_name' (or 'backend') for IBM execution")

        backend = _get_backend(service, backend_name)
        backend_version = str(getattr(backend, "backend_version", "unknown"))
        basis_gates, coupling_map_hash = _backend_topology_metadata(backend)

        native = circuit.native_circuit
        if native is None:
            if not circuit.qasm3:
                raise ValueError("CircuitHandle has no native circuit or qasm3 content for IBM execution")
            native = self.ingest(circuit.qasm3).native_circuit

        transpile_seed = int(backend_spec.get("seed", DEFAULT_SEED))
        transpile_opt = int(backend_spec.get("optimization_level", 1))
        transpile_kwargs = dict(backend_spec.get("transpile", {}))

        t_submit_start = time.perf_counter()
        with _span_or_noop(
            emitter,
            "compile/transpile_hardware",
            {
                "provider": "ibm",
                "backend_name": backend_name,
                "optimization_level": transpile_opt,
            },
        ) as transpile_span:
            transpiled = self._qiskit["transpile"](
                native,
                backend=backend,
                optimization_level=transpile_opt,
                seed_transpiler=transpile_seed,
                **transpile_kwargs,
            )
            if transpile_span is not None:
                transpile_span.set_attribute("depth_before", _safe_depth(native))
                transpile_span.set_attribute("depth_after", _safe_depth(transpiled))
                transpile_span.set_attribute("size_before", _safe_size(native))
                transpile_span.set_attribute("size_after", _safe_size(transpiled))

        primitive_name = str(backend_spec.get("primitive", "sampler")).lower()
        session_id = None
        error_budget = backend_spec.get("error_budget")

        with _span_or_noop(
            emitter,
            "job_submit",
            {
                "provider": "ibm",
                "backend_name": backend_name,
                "backend_version": backend_version,
                "basis_gates": basis_gates,
                "coupling_map_hash": coupling_map_hash,
            },
        ) as submit_span:
            job = _submit_runtime_job(
                runtime=self._runtime,
                primitive_name=primitive_name,
                backend=backend,
                transpiled=transpiled,
                shots=shots,
                backend_spec=backend_spec,
            )
            job_id = str(getattr(job, "job_id", lambda: "")() if callable(getattr(job, "job_id", None)) else getattr(job, "job_id", ""))
            if submit_span is not None:
                submit_span.set_attribute("job_id", job_id)
                if error_budget is not None:
                    submit_span.set_attribute("error_budget", error_budget)

        poll_interval_s = float(backend_spec.get("poll_interval_s", 5.0))
        timeout_s = backend_spec.get("timeout_s")
        timeout_s = float(timeout_s) if timeout_s is not None else None

        queue_start = time.perf_counter()
        with _span_or_noop(
            emitter,
            "queue_wait",
            {
                "job_id": job_id,
                "poll_interval_s": poll_interval_s,
            },
        ) as queue_span:
            terminal_status = _wait_for_job(
                job=job,
                queue_span=queue_span,
                poll_interval_s=poll_interval_s,
                timeout_s=timeout_s,
            )
        queue_time_s = time.perf_counter() - queue_start

        with _span_or_noop(
            emitter,
            "job_complete",
            {
                "job_id": job_id,
                "status": terminal_status,
            },
        ):
            pass

        with _span_or_noop(
            emitter,
            "result_fetch",
            {
                "job_id": job_id,
            },
        ) as fetch_span:
            t_fetch = time.perf_counter()
            runtime_result = job.result()
            run_time_s = time.perf_counter() - t_fetch
            if fetch_span is not None:
                fetch_span.set_attribute("wall_time_s", time.perf_counter() - t_submit_start)

        counts = _extract_counts(runtime_result, shots)
        raw_result = _to_jsonable(runtime_result)
        metadata = {
            "provider": "ibm",
            "primitive": primitive_name,
            "backend_version": backend_version,
            "basis_gates": basis_gates,
            "coupling_map_hash": coupling_map_hash,
            "session_id": session_id,
            "error_budget": error_budget,
            "terminal_status": terminal_status,
            "raw_result": raw_result,
        }

        return ExecutionResult(
            job_id=job_id,
            backend_name=backend_name,
            shots=shots,
            counts=counts,
            metadata=metadata,
            queue_time_s=queue_time_s,
            run_time_s=run_time_s,
            error_mitigation_applied=bool(backend_spec.get("error_mitigation", False)),
        )


def _import_ibm_runtime() -> dict[str, Any]:
    try:
        import qiskit_ibm_runtime as qir  # type: ignore[import-untyped]

        return {
            "module": qir,
            "QiskitRuntimeService": qir.QiskitRuntimeService,
            "SamplerV2": getattr(qir, "SamplerV2", None),
            "EstimatorV2": getattr(qir, "EstimatorV2", None),
            "Sampler": getattr(qir, "Sampler", None),
            "Estimator": getattr(qir, "Estimator", None),
        }
    except ImportError as exc:
        raise ImportError(
            "qiskit-ibm-runtime is required for IBM hardware execution. "
            "Install with: pip install 'qocc[ibm]'"
        ) from exc


def _build_service(runtime: dict[str, Any], backend_spec: dict[str, Any]) -> Any:
    svc_cls = runtime["QiskitRuntimeService"]
    kwargs: dict[str, Any] = {}
    for key in ("channel", "token", "instance", "url"):
        if key in backend_spec and backend_spec[key] is not None:
            kwargs[key] = backend_spec[key]
    return svc_cls(**kwargs)


def _get_backend(service: Any, backend_name: str) -> Any:
    if hasattr(service, "backend"):
        return service.backend(backend_name)
    if hasattr(service, "get_backend"):
        return service.get_backend(backend_name)
    raise RuntimeError("IBM Runtime service object does not expose backend lookup")


def _backend_topology_metadata(backend: Any) -> tuple[list[str], str]:
    cfg = None
    if hasattr(backend, "configuration") and callable(getattr(backend, "configuration")):
        cfg = backend.configuration()
    basis_gates = list(getattr(cfg, "basis_gates", []) or [])
    coupling_map = getattr(cfg, "coupling_map", None)
    cmap_payload = json.dumps(coupling_map, sort_keys=True, default=str)
    coupling_map_hash = hashlib.sha256(cmap_payload.encode("utf-8")).hexdigest()
    return basis_gates, coupling_map_hash


def _submit_runtime_job(
    runtime: dict[str, Any],
    primitive_name: str,
    backend: Any,
    transpiled: Any,
    shots: int,
    backend_spec: dict[str, Any],
) -> Any:
    if primitive_name == "estimator":
        estimator_cls = runtime.get("EstimatorV2") or runtime.get("Estimator")
        if estimator_cls is None:
            raise RuntimeError("Estimator primitive is unavailable in installed qiskit_ibm_runtime")
        estimator = _instantiate_primitive(estimator_cls, backend)
        observables = backend_spec.get("observables") or ["Z"]
        if hasattr(estimator, "run"):
            try:
                return estimator.run([(transpiled, observables)], shots=shots)
            except TypeError:
                return estimator.run([(transpiled, observables)])
        raise RuntimeError("Estimator primitive does not expose run()")

    sampler_cls = runtime.get("SamplerV2") or runtime.get("Sampler")
    if sampler_cls is None:
        raise RuntimeError("Sampler primitive is unavailable in installed qiskit_ibm_runtime")
    sampler = _instantiate_primitive(sampler_cls, backend)
    if hasattr(sampler, "run"):
        try:
            return sampler.run([transpiled], shots=shots)
        except TypeError:
            return sampler.run([transpiled])
    raise RuntimeError("Sampler primitive does not expose run()")


def _instantiate_primitive(primitive_cls: Any, backend: Any) -> Any:
    for kwargs in ({"backend": backend}, {"mode": backend}, {}):
        try:
            return primitive_cls(**kwargs)
        except TypeError:
            continue
    return primitive_cls()


def _wait_for_job(
    job: Any,
    queue_span: Any,
    poll_interval_s: float,
    timeout_s: float | None,
) -> str:
    t0 = time.perf_counter()
    terminal = {"done", "completed", "cancelled", "error", "failed"}
    while True:
        status = _job_status(job)
        elapsed = time.perf_counter() - t0
        if queue_span is not None:
            queue_span.add_event("job_polling", status=status, elapsed_s=elapsed)
        if status.lower() in terminal:
            return status
        if timeout_s is not None and elapsed >= timeout_s:
            raise TimeoutError(f"IBM runtime job polling timed out after {timeout_s:.1f}s")
        time.sleep(max(0.05, poll_interval_s))


def _job_status(job: Any) -> str:
    s = getattr(job, "status", None)
    if callable(s):
        try:
            s = s()
        except Exception:
            s = None
    if s is None and hasattr(job, "state"):
        st = getattr(job, "state")
        s = st() if callable(st) else st
    return str(s or "unknown")


def poll_ibm_job(
    backend_spec: dict[str, Any],
    job_id: str,
    shots: int = 1024,
) -> dict[str, Any]:
    """Poll a previously submitted IBM runtime job by ID."""
    if not job_id:
        return {"status": "failed", "done": True, "error": "Missing IBM job_id"}

    runtime = _import_ibm_runtime()
    service = _build_service(runtime, backend_spec)

    if hasattr(service, "job"):
        job = service.job(job_id)
    elif hasattr(service, "retrieve_job"):
        job = service.retrieve_job(job_id)
    else:
        return {"status": "failed", "done": True, "error": "IBM Runtime service cannot retrieve jobs"}

    status = _job_status(job)
    status_l = status.lower()
    terminal_ok = {"done", "completed"}
    terminal_fail = {"cancelled", "error", "failed"}
    if status_l not in terminal_ok | terminal_fail:
        return {"status": status, "done": False}

    if status_l in terminal_fail:
        return {"status": status, "done": True, "error": f"IBM job ended with status {status}"}

    result = job.result()
    counts = _extract_counts(result, shots)
    return {
        "status": status,
        "done": True,
        "result": {
            "counts": counts,
            "metadata": {
                "provider": "ibm",
                "backend_version": str(backend_spec.get("backend_version", "unknown")),
                "raw_result": _to_jsonable(result),
            },
        },
    }


def _extract_counts(result: Any, shots: int) -> dict[str, int]:
    if hasattr(result, "get_counts"):
        try:
            counts = result.get_counts()
            if isinstance(counts, dict):
                return {str(k): int(v) for k, v in counts.items()}
        except Exception:
            pass

    quasi_dists = getattr(result, "quasi_dists", None)
    if quasi_dists and isinstance(quasi_dists, list) and quasi_dists:
        first = quasi_dists[0]
        if isinstance(first, dict):
            out: dict[str, int] = {}
            for key, prob in first.items():
                try:
                    k = format(int(key), "b") if isinstance(key, int) else str(key)
                    out[k] = int(round(float(prob) * shots))
                except Exception:
                    continue
            if out:
                return out

    return {}


def _safe_depth(circuit: Any) -> int | None:
    fn = getattr(circuit, "depth", None)
    if callable(fn):
        try:
            return int(fn())
        except Exception:
            return None
    return None


def _safe_size(circuit: Any) -> int | None:
    fn = getattr(circuit, "size", None)
    if callable(fn):
        try:
            return int(fn())
        except Exception:
            return None
    return None


def _to_jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except Exception:
        pass

    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        try:
            return value.to_dict()
        except Exception:
            pass

    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    return repr(value)


class _span_or_noop:
    def __init__(self, emitter: Any | None, name: str, attributes: dict[str, Any] | None = None) -> None:
        self._emitter = emitter
        self._name = name
        self._attrs = attributes
        self._ctx = None
        self.span = None

    def __enter__(self) -> Any:
        if self._emitter is None:
            return None
        self._ctx = self._emitter.span(self._name, attributes=self._attrs or {})
        self.span = self._ctx.__enter__()
        return self.span

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._ctx is not None:
            self._ctx.__exit__(exc_type, exc, tb)


register_adapter("ibm", IBMAdapter)
