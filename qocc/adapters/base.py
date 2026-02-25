"""Base adapter interface.

Every backend adapter must subclass ``BaseAdapter`` and implement the
abstract methods below.  The adapter interface guarantees that QOCC can
ingest, normalise, compile, export, hash, and measure circuits uniformly.
"""

from __future__ import annotations

import abc
from typing import Any

from qocc.core.circuit_handle import BackendInfo, CircuitHandle, PassLogEntry, PipelineSpec


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
    def compile(self, circuit: CircuitHandle, pipeline: PipelineSpec) -> CompileResult:
        """Compile/transpile the circuit with the given pipeline spec."""

    def simulate(self, circuit: CircuitHandle, spec: SimulationSpec) -> SimulationResult:
        """Run simulation. Optional in MVP — adapters may raise NotImplementedError."""
        raise NotImplementedError(f"{self.name()} adapter does not support simulation yet.")

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


def get_adapter(name: str) -> BaseAdapter:
    """Instantiate and return the adapter registered under *name*."""
    if name not in _REGISTRY:
        # try lazy import
        if name == "qiskit":
            from qocc.adapters.qiskit_adapter import QiskitAdapter  # noqa: F811
            return QiskitAdapter()
        elif name == "cirq":
            from qocc.adapters.cirq_adapter import CirqAdapter  # noqa: F811
            return CirqAdapter()
        raise KeyError(f"No adapter registered for {name!r}. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]()
