import json
from pathlib import Path
from qocc.core.cross_check import cross_check
from qocc.cli.main import cli
from click.testing import CliRunner

from unittest.mock import patch, MagicMock

class DummyAdapter(MagicMock):
    def ingest(self, source):
        return MagicMock(name="handle", num_qubits=2, metadata={"metrics": {"n_gates": 1}})
    
    def normalize_circuit(self, handle):
        return handle
        
    def simulate(self, handle, spec):
        return MagicMock(counts={"00": 500, "11": 500}, metadata={"statevector": [0.707, 0, 0, 0.707]})
        
    def get_metrics(self, handle):
        m = MagicMock()
        m.to_dict.return_value = {"n_gates": 1}
        return m

def test_cross_check_api(tmp_path):
    qasm_file = tmp_path / "circuit.qasm"
    qasm_file.write_text("OPENQASM 3.0; qubit[2] q; cx q[0], q[1];")
    
    spec = {
        "name": "dist_test",
        "type": "distribution",
        "spec": {"test": "tvd", "threshold": 0.05}
    }
    
    with patch("qocc.core.cross_check.get_adapter", return_value=DummyAdapter()):
        result = cross_check(
            str(qasm_file),
            ["mock_a", "mock_b"],
            [spec],
            1000,
            42
        )
    
    assert "adapters" in result
    assert "mock_a" in result["matrix"]
    results_mock_mock = result["matrix"]["mock_a"]["mock_b"]
    assert len(results_mock_mock) == 1
    assert results_mock_mock[0]["passed"] is True

def test_cross_check_cli(tmp_path):
    qasm_file = tmp_path / "circuit.qasm"
    qasm_file.write_text("OPENQASM 3.0; qubit[2] q; cx q[0], q[1];")
    
    contract_file = tmp_path / "contract.json"
    spec = {
        "name": "dist_test",
        "type": "distribution",
        "spec": {"test": "tvd", "threshold": 0.05}
    }
    contract_file.write_text(json.dumps([spec]))
    
    runner = CliRunner()
    with patch("qocc.core.cross_check.get_adapter", return_value=DummyAdapter()):
        result = runner.invoke(cli, [
            "cross-check",
            "--adapters", "mock_a,mock_b",
            "--input", str(qasm_file),
            "--contract", str(contract_file)
        ])
    
    assert result.exit_code == 0
    assert "Cross-Adapter Portability Matrix" in result.output
    assert "PASS" in result.output
    
    raw_result = Path("cross_check_result.json")
    assert raw_result.exists()
    try:
        raw_result.unlink()
    except Exception:
        pass
