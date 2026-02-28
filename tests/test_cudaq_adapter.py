import pytest
from qocc.adapters.cudaq_adapter import CudaqAdapter
from qocc.core.circuit_handle import CircuitHandle
from qocc.contracts.spec import ContractSpec, ContractResult

class MockCudaqAdapter(CudaqAdapter):
    def simulate(self, circuit: CircuitHandle, spec: dict) -> dict:
        # mock a simulation response for a test
        pass

def test_cudaq_adapter_initialization():
    adapter = CudaqAdapter()
    assert adapter is not None

def test_cudaq_adapter_ingest_qasm():
    adapter = CudaqAdapter()
    qasm_str = "OPENQASM 3.0;\nqubit[3] q;\n"
    handle = adapter.ingest(qasm_str)
    assert handle.num_qubits == 3

def test_cudaq_adapter_ingest_kernel():
    class MockKernel:
        name = "test_kernel"
        num_qubits = 2

    adapter = CudaqAdapter()
    kernel = MockKernel()
    handle = adapter.ingest(kernel)
    assert handle.name == "test_kernel"
    assert handle.num_qubits == 2

def test_cudaq_adapter_compile():
    adapter = CudaqAdapter()
    handle = CircuitHandle(
        name="test_compile",
        source_format="cudaq",
        native_circuit="OPENQASM 3.0;\n",
        num_qubits=2,
    )
    pipeline_spec = {"target": "tensornet"}
    compiled = adapter.compile(handle, pipeline_spec)
    assert compiled.metadata["cudaq_target"] == "tensornet"

def test_cudaq_adapter_get_metrics():
    adapter = CudaqAdapter()
    handle = CircuitHandle(
        name="test_metrics",
        source_format="cudaq",
        native_circuit="OPENQASM 3.0;\n",
        num_qubits=4,
    )
    metrics = adapter.get_metrics(handle)
    assert metrics.get("n_qubits") == 4
