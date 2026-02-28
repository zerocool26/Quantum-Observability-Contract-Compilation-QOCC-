import pytest
from qocc.contracts.spec import ContractSpec, ContractResult
from qocc.core.circuit_handle import CircuitHandle
from qocc.contracts.eval_qpt import evaluate_qpt_contract, simulate_unitary_approx, perform_rb_approx

def test_qpt_tomography_small_circuit():
    """Test QPT contraction for small circuit (<=5 qubits) using state tomography mock"""
    circuit = CircuitHandle(
        name="test",
        source_format="mock",
        native_circuit=None,
        qasm3="OPENQASM 3.0; qubit[2] q;",
        num_qubits=2,
        metadata={"metrics": {"n_gates": 0, "depth": 0}}
    )
    ideal = CircuitHandle(
        name="test_ideal",
        source_format="mock",
        native_circuit=None,
        qasm3="OPENQASM 3.0; qubit[2] q;",
        num_qubits=2,
        metadata={"metrics": {"n_gates": 0, "depth": 0}}
    )
    
    spec = ContractSpec(
        name="qpt_test",
        type="qpt",
        spec={"threshold": 0.9, "confidence": 0.95}
    )
    
    result = evaluate_qpt_contract(adapter=None, circuit=circuit, spec=spec, ideal_circuit=ideal)
    # Both are mocked to eye() for fidelity 1.0
    assert result.passed is True
    assert result.details["method"] == "tomography"
    assert result.details["estimated_fidelity"] == 1.0


def test_qpt_rb_large_circuit():
    """Test QPT contraction for large circuit (>5 qubits) using RB proxy"""
    circuit = CircuitHandle(
        name="test_large",
        source_format="mock",
        native_circuit=None,
        qasm3="OPENQASM 3.0; qubit[15] q;",
        num_qubits=15,
        metadata={"metrics": {"n_gates": 100, "depth": 50}}
    )
    
    spec = ContractSpec(
        name="qpt_test",
        type="qpt",
        spec={"threshold": 0.8, "confidence": 0.95}
    )
    
    # It decays as exp(-0.001 * 50) = 0.9512
    result = evaluate_qpt_contract(adapter=None, circuit=circuit, spec=spec)
    assert result.passed is True
    assert result.details["method"] == "rb"
    assert "estimated_fidelity" in result.details
    assert result.details["estimated_fidelity"] > 0.8
    assert result.details["ci_lower"] < result.details["ci_upper"]

def test_qpt_rb_fail_large_circuit():
    """Test QPT contract explicitly failing when depth is high"""
    circuit = CircuitHandle(
        name="test_large_fail",
        source_format="mock",
        native_circuit=None,
        qasm3="OPENQASM 3.0; qubit[15] q;",
        num_qubits=15,
        metadata={"metrics": {"n_gates": 2000, "depth": 1000}}
    )
    
    spec = ContractSpec(
        name="qpt_test_fail",
        type="qpt",
        spec={"threshold": 0.8, "confidence": 0.95}
    )
    
    # It decays as exp(-0.001 * 1000) ~ 0.36
    result = evaluate_qpt_contract(adapter=None, circuit=circuit, spec=spec)
    assert result.passed is False
    assert result.details["method"] == "rb"
    assert result.details["estimated_fidelity"] < 0.5
