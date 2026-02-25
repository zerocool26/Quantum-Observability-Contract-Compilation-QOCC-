"""Tests for deterministic hashing."""

from __future__ import annotations

import pytest

from qocc.core.circuit_handle import CircuitHandle, PipelineSpec
from qocc.core.hashing import hash_bytes, hash_dict, hash_string


def test_hash_string_deterministic():
    """Same string always produces the same hash."""
    a = hash_string("hello quantum world")
    b = hash_string("hello quantum world")
    assert a == b
    assert len(a) == 64  # SHA-256 hex


def test_hash_bytes_deterministic():
    a = hash_bytes(b"\x00\x01\x02")
    b = hash_bytes(b"\x00\x01\x02")
    assert a == b


def test_hash_dict_deterministic():
    """Dict hashing should be order-independent (uses sorted keys)."""
    a = hash_dict({"b": 2, "a": 1})
    b = hash_dict({"a": 1, "b": 2})
    assert a == b


def test_hash_dict_different():
    a = hash_dict({"a": 1})
    b = hash_dict({"a": 2})
    assert a != b


def test_circuit_handle_hash_from_qasm():
    """CircuitHandle hash should be stable given the same QASM3 string."""
    qasm = "OPENQASM 3.0;\nqubit[2] q;\nh q[0];\ncx q[0], q[1];\n"
    h1 = CircuitHandle(name="test", num_qubits=2, native_circuit=None,
                       source_format="qasm3", qasm3=qasm)
    h2 = CircuitHandle(name="test", num_qubits=2, native_circuit=None,
                       source_format="qasm3", qasm3=qasm)
    assert h1.stable_hash() == h2.stable_hash()
    assert h1 == h2


def test_circuit_handle_hash_changes_with_content():
    """Different QASM should produce different hashes."""
    q1 = "OPENQASM 3.0;\nqubit[2] q;\nh q[0];\n"
    q2 = "OPENQASM 3.0;\nqubit[2] q;\nx q[0];\n"
    h1 = CircuitHandle(name="a", num_qubits=2, native_circuit=None,
                       source_format="qasm3", qasm3=q1)
    h2 = CircuitHandle(name="b", num_qubits=2, native_circuit=None,
                       source_format="qasm3", qasm3=q2)
    assert h1.stable_hash() != h2.stable_hash()


def test_pipeline_spec_hash_stable():
    """PipelineSpec hash should be deterministic."""
    p1 = PipelineSpec(adapter="qiskit", optimization_level=2, parameters={"seed": 42})
    p2 = PipelineSpec(adapter="qiskit", optimization_level=2, parameters={"seed": 42})
    assert p1.stable_hash() == p2.stable_hash()


def test_pipeline_spec_hash_changes():
    p1 = PipelineSpec(adapter="qiskit", optimization_level=1)
    p2 = PipelineSpec(adapter="qiskit", optimization_level=2)
    assert p1.stable_hash() != p2.stable_hash()
