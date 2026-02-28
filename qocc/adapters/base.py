"""Base adapter interface.

Every backend adapter must subclass ``BaseAdapter`` and implement the
abstract methods below.  The adapter interface guarantees that QOCC can
ingest, normalise, compile, export, hash, and measure circuits uniformly.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qocc.trace.emitter import TraceEmitter

from qocc.core.circuit_handle import BackendInfo, CircuitHandle, PassLogEntry, PipelineSpec

logger = logging.getLogger("qocc.adapters")


class SimulationSpec:
    """Configuration for running a simulation."""

    def __init__(
        self,
        shots: int = 1024,
        seed: int | None = None,
        method: str = "statevector",
        **kwargs: Any,
    ) -> None:
        self.shots = shots
        self.seed = seed
        self.method = method
        self.extra = kwargs

    def to_dict(self) -> dict[str, Any]:
        return {
            "shots": self.shots,
            "seed": self.seed,
            "method": self.method,
            **self.extra,
        }


class SimulationResult:
    """Result of a simulation run."""

    def __init__(
        self,
        counts: dict[str, int],
        shots: int,
        seed: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.counts = counts
        self.shots = shots
        self.seed = seed
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": self.counts,
            "shots": self.shots,
            "seed": self.seed,
            "metadata": self.metadata,
        }


class CompileResult:
    """Result of compilation — a circuit + pass log."""

    def __init__(
        self,
        circuit: CircuitHandle,
        pass_log: list[PassLogEntry] | None = None,
    ) -> None:
        self.circuit = circuit
        self.pass_log = pass_log or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "circuit": self.circuit.to_dict(),
            "pass_log": [p.to_dict() for p in self.pass_log],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CompileResult:
        """Reconstruct a CompileResult from its serialised dict.

        The native circuit is not available — only QASM3 text is preserved,
        so the ``CircuitHandle.native_circuit`` is set to ``None``.
        """
        c = d.get("circuit", {})
        handle = CircuitHandle(
            name=c.get("name", "cached"),
            num_qubits=c.get("num_qubits", 0),
            native_circuit=None,
            source_format=c.get("source_format", "unknown"),
            qasm3=c.get("qasm3"),
            _normalized=c.get("normalized", False),
        )
        pass_log = [
            PassLogEntry(
                pass_name=p.get("pass_name", ""),
                parameters=p.get("parameters", {}),
                order=p.get("order", 0),
                duration_ms=p.get("duration_ms"),
                memory_bytes=p.get("memory_bytes"),
                warnings=p.get("warnings", []),
                errors=p.get("errors", []),
            )
            for p in d.get("pass_log", [])
        ]
        return cls(circuit=handle, pass_log=pass_log)


@dataclass
class ExecutionResult:
    """Result of a real hardware execution run."""

    job_id: str
    backend_name: str
    shots: int
    counts: dict[str, int]
    metadata: dict[str, Any] = field(default_factory=dict)
    queue_time_s: float | None = None
    run_time_s: float | None = None
    error_mitigation_applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "backend_name": self.backend_name,
            "shots": self.shots,
            "counts": self.counts,
            "metadata": self.metadata,
            "queue_time_s": self.queue_time_s,
            "run_time_s": self.run_time_s,
            "error_mitigation_applied": self.error_mitigation_applied,
        }


class MetricsSnapshot:
    """Immutable metrics for a circuit at a point in time."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = dict(data)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)


class BaseAdapter(abc.ABC):
    """Abstract base adapter all backend adapters must implement."""

    @abc.abstractmethod
    def name(self) -> str:
        """Return adapter name (e.g. ``'qiskit'``)."""

    @abc.abstractmethod
    def ingest(self, source: str | Any) -> CircuitHandle:
        """Load a circuit from a file path, QASM string, or native object.

        Parameters:
            source: File path (str ending in ``.qasm``), raw QASM string, or
                    a native circuit object.

        Returns:
            A ``CircuitHandle`` wrapping the circuit.
        """

    @abc.abstractmethod
    def normalize(self, circuit: CircuitHandle) -> CircuitHandle:
        """Canonicalize the circuit (ordering, naming, register mapping)."""

    @abc.abstractmethod
    def export(self, circuit: CircuitHandle, fmt: str = "qasm3") -> str:
        """Export the circuit to the given format string."""

    @abc.abstractmethod
    def compile(
        self,
        circuit: CircuitHandle,
        pipeline: PipelineSpec,
        emitter: Any | None = None,
    ) -> CompileResult:
        """Compile/transpile the circuit with the given pipeline spec.

        Parameters:
            circuit: Input circuit handle.
            pipeline: Pipeline specification.
            emitter: Optional ``TraceEmitter`` — when supplied, each
                     compilation pass/stage should be wrapped in a child span.
        """

    def simulate(self, circuit: CircuitHandle, spec: SimulationSpec) -> SimulationResult:
        """Run simulation. Optional in MVP — adapters may raise NotImplementedError."""
        raise NotImplementedError(f"{self.name()} adapter does not support simulation yet.")

    def execute(
        self,
        circuit: CircuitHandle,
        backend_spec: dict[str, Any],
        shots: int = 1024,
        emitter: "TraceEmitter" | None = None,
    ) -> ExecutionResult:
        """Submit to real hardware and return counts + job metadata.

        Adapters implementing real hardware execution must emit these span names:
        ``job_submit``, ``queue_wait``, ``job_complete``, ``result_fetch``.

        Recommended span attributes include ``job_id``, ``provider``,
        ``backend_version``, ``basis_gates``, and ``coupling_map_hash``.
        While polling asynchronous jobs, adapters should add ``job_polling``
        events at their configured interval and record total wall time.
        """
        raise NotImplementedError(f"{self.name()} adapter does not support hardware execute() yet.")

    @abc.abstractmethod
    def get_metrics(self, circuit: CircuitHandle) -> MetricsSnapshot:
        """Compute and return a metrics snapshot for the given circuit."""

    @abc.abstractmethod
    def hash(self, circuit: CircuitHandle) -> str:
        """Return a stable hash of the circuit (via normalization + serialization)."""

    @abc.abstractmethod
    def describe_backend(self) -> BackendInfo:
        """Return backend/version information."""


# ------------------------------------------------------------------
# Adapter registry
# ------------------------------------------------------------------

_REGISTRY: dict[str, type[BaseAdapter]] = {}


def register_adapter(name: str, cls: type[BaseAdapter]) -> None:
    """Register an adapter class under *name*."""
    _REGISTRY[name] = cls


_ep_adapters_discovered = False


def _discover_entry_point_adapters() -> None:
    """Auto-discover adapters registered via ``qocc.adapters`` entry-point group.

    Scans only once; subsequent calls are no-ops.
    """
    global _ep_adapters_discovered
    if _ep_adapters_discovered:
        return
    _ep_adapters_discovered = True
    import importlib.metadata

    try:
        eps = importlib.metadata.entry_points()
        # Python 3.12+ returns a SelectableGroups; older returns dict
        group = eps.select(group="qocc.adapters") if hasattr(eps, "select") else eps.get("qocc.adapters", [])
        for ep in group:
            try:
                cls = ep.load()
                if ep.name not in _REGISTRY:
                    _REGISTRY[ep.name] = cls
            except Exception:
                logger.debug("Failed to load adapter entry-point %s", ep.name, exc_info=True)
    except Exception:
        logger.debug("Entry-point discovery for qocc.adapters failed", exc_info=True)


def get_adapter(name: str) -> BaseAdapter:
    """Instantiate and return the adapter registered under *name*.

    Resolution order:
    1. Explicit ``register_adapter()`` calls.
    2. ``qocc.adapters`` entry-point group (auto-discovered once).
    3. Built-in lazy imports for ``"qiskit"``, ``"cirq"``, ``"tket"``, and ``"stim"``.
    """
    if name not in _REGISTRY:
        # Try entry-point discovery
        _discover_entry_point_adapters()

    if name not in _REGISTRY:
        # Built-in lazy import fallback
        if name == "qiskit":
            from qocc.adapters.qiskit_adapter import QiskitAdapter  # noqa: F811
            return QiskitAdapter()
        elif name == "cirq":
            from qocc.adapters.cirq_adapter import CirqAdapter  # noqa: F811
            return CirqAdapter()
        elif name == "tket":
            from qocc.adapters.tket_adapter import TketAdapter  # noqa: F811
            return TketAdapter()
        elif name == "stim":
            from qocc.adapters.stim_adapter import StimAdapter  # noqa: F811
            return StimAdapter()
        elif name == "ibm":
            from qocc.adapters.ibm_adapter import IBMAdapter  # noqa: F811
            return IBMAdapter()
        elif name == "cudaq":
            from qocc.adapters.cudaq_adapter import CudaqAdapter
            return CudaqAdapter()
        raise KeyError(f"No adapter registered for {name!r}. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]()
