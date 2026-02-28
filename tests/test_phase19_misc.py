import pytest
from qocc.core.format_bridge import convert
from qocc.metrics.topology import TopologyGraph
from qocc.core.circuit_handle import CircuitHandle
from qocc.contracts.spec import ContractSpec
from qocc.contracts.eval_topology import evaluate_topology_contract

def test_topology_graph_load(tmp_path):
    json_path = tmp_path / "top.json"
    json_path.write_text('{"nodes": [{"qubit": 0, "t1": 10}, {"qubit": 1}], "edges": [[0, 1]]}')
    
    graph = TopologyGraph.from_json(str(json_path))
    assert 0 in graph.nodes
    assert graph.nodes[0].t1 == 10
    assert (0, 1) in graph.edges

def test_topology_contract():
    spec = ContractSpec(
        name="top_test",
        type="topology_violations",
        spec={"max_swap_insertions": 2}
    )
    
    # Under budget
    res = evaluate_topology_contract(spec, {"topology_violations": 1})
    assert res.passed is True
    assert res.details["actual_violations"] == 1
    
    # Over budget
    res2 = evaluate_topology_contract(spec, {"topology_violations": 3})
    assert res2.passed is False

def test_format_bridge():
    handle = CircuitHandle(
        name="test",
        source_format="qasm3",
        num_qubits=2,
        native_circuit=None,
        qasm3="OPENQASM 3.0; qubit[2] q;"
    )
    
    # A normal test would use qiskit/cirq if installed, but let's test failure mode on missing adapter
    with pytest.raises(RuntimeError) as exc:
        convert(handle, "unknown_fmt")
        
    assert "Unknown target format" in str(exc.value)
