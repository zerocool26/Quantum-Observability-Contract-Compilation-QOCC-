"""IR-neutral circuit wrapper for QOCC.

CircuitHandle wraps any vendor circuit object and provides a uniform
interface for metadata, hashing, and serialization.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=False)
class CircuitHandle:
    """Vendor-agnostic wrapper around a quantum circuit.

    Attributes:
        name: Human-readable circuit name.
        num_qubits: Number of qubits in the circuit.
        native_circuit: The vendor-specific circuit object (Qiskit QuantumCircuit, Cirq Circuit, etc.).
        source_format: Original format string (e.g. ``"qiskit"``, ``"cirq"``, ``"qasm3"``).
        metadata: Arbitrary key-value metadata.
        qasm3: OpenQASM 3 representation (populated after export).
        _normalized: Whether the circuit has been canonicalized.
        _stable_hash_cache: Cached result of ``stable_hash()``.  Computed
            once on first call so that the hash stays immutable even if
            mutable fields are later modified (e.g. after ``deepcopy``).

    .. warning::

        Do **not** mutate ``qasm3`` or ``native_circuit`` after the first
        call to ``stable_hash()`` / ``__hash__()``; the cached value will
        not be updated.  ``normalize_circuit()`` deep-copies first.
    """

    name: str
    num_qubits: int
    native_circuit: Any
    source_format: str
    metadata: dict[str, Any] = field(default_factory=dict)
    qasm3: str | None = None
    _normalized: bool = False
    _stable_hash_cache: str | None = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict (no native circuit)."""
        return {
            "name": self.name,
            "num_qubits": self.num_qubits,
            "source_format": self.source_format,
            "metadata": self.metadata,
            "qasm3": self.qasm3,
            "normalized": self._normalized,
            "hash": self.stable_hash(),
        }

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------

    def stable_hash(self) -> str:
        """Produce a deterministic SHA-256 hash of the canonical representation.

        The result is cached after the first call.  If QASM3 is available
        use it; otherwise fall back to a ``repr()``-based hash (less
        stable across versions).
        """
        if self._stable_hash_cache is not None:
            return self._stable_hash_cache
        if self.qasm3 is not None:
            payload = self.qasm3.encode("utf-8")
        else:
            payload = repr(self.native_circuit).encode("utf-8")
        self._stable_hash_cache = hashlib.sha256(payload).hexdigest()
        return self._stable_hash_cache

    def __hash__(self) -> int:  # noqa: D105
        return int(self.stable_hash()[:16], 16)

    def __eq__(self, other: object) -> bool:  # noqa: D105
        if not isinstance(other, CircuitHandle):
            return NotImplemented
        return self.stable_hash() == other.stable_hash()

    def __repr__(self) -> str:
        return (
            f"CircuitHandle(name={self.name!r}, qubits={self.num_qubits}, "
            f"fmt={self.source_format!r}, hash={self.stable_hash()[:12]}…)"
        )


@dataclass
class PassLogEntry:
    """A single compilation-pass log entry."""

    pass_name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    order: int = 0
    duration_ms: float | None = None
    memory_bytes: int | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass_name": self.pass_name,
            "parameters": self.parameters,
            "order": self.order,
            "duration_ms": self.duration_ms,
            "memory_bytes": self.memory_bytes,
            "warnings": self.warnings,
            "errors": self.errors,
        }


@dataclass
class PipelineSpec:
    """Describes a compilation pipeline configuration.

    Attributes:
        adapter: Adapter name (``"qiskit"``, ``"cirq"``).
        optimization_level: Framework opt level.
        passes: Ordered list of pass names.
        parameters: Extra key-value parameters.
    """

    adapter: str
    optimization_level: int = 1
    passes: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    mitigation: MitigationSpec | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "adapter": self.adapter,
            "optimization_level": self.optimization_level,
            "passes": self.passes,
            "parameters": self.parameters,
        }
        if self.mitigation is not None:
            payload["mitigation"] = self.mitigation.to_dict()
        return payload

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PipelineSpec:
        mitigation_payload = d.get("mitigation")
        mitigation: MitigationSpec | None = None
        if isinstance(mitigation_payload, dict):
            mitigation = MitigationSpec.from_dict(mitigation_payload)
        return cls(
            adapter=d["adapter"],
            optimization_level=d.get("optimization_level", 1),
            passes=d.get("passes", []),
            parameters=d.get("parameters", {}),
            mitigation=mitigation,
        )

    def stable_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass
class MitigationSpec:
    """Error mitigation configuration for an optional pipeline stage."""

    method: str
    params: dict[str, Any] = field(default_factory=dict)
    overhead_budget: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "params": dict(self.params),
            "overhead_budget": dict(self.overhead_budget),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MitigationSpec":
        if "method" not in data:
            raise ValueError("MitigationSpec requires 'method'")
        return cls(
            method=str(data["method"]),
            params=dict(data.get("params", {}) or {}),
            overhead_budget=dict(data.get("overhead_budget", {}) or {}),
        )


@dataclass
class BackendInfo:
    """Describes an adapter backend."""

    name: str
    version: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "extra": self.extra}
