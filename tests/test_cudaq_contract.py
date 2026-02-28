import pytest
from qocc.adapters.cudaq_adapter import CudaqAdapter
from qocc.contracts.spec import ContractSpec, ContractResult
from qocc.core.circuit_handle import CircuitHandle

def test_cudaq_memory_budget_contract():
    # As detailed in prompt2.md, we need to enforce a VRAM memory budget 
    adapter = CudaqAdapter()
    
    # Mock some data
    class MockCircuit:
        name = "mock_cudaq"
        num_qubits = 20
        native_circuit = "some qasm or cudaq.kernel"
        qasm3 = None
        metadata = {}
        
    handle = CircuitHandle(
        name="mock_cudaq",
        source_format="cudaq",
        num_qubits=20,
        native_circuit="some qasm or cudaq.kernel",
        metadata={}
    )
    
    # To evaluate a memory budget contract, often it's simulated and the metadata captures 'vram_usage_mb'
    # Then a specific evaluator or adapter checks it. For now, since prompt2.md asked to ensure
    # VRAM/GPU details can be extracted, we'll verify it returns the 'vram_usage_mb' after simulation.
    spec = ContractSpec(
        name="max_vram",
        type="cost", # cost memory contract
        resource_budget={"max_vram_mb": 512}
    )
    
    # Run simulate and capture VRAM. In reality, we don't have CUDA-Q installed for test, so simulate will raise RuntimeError
    # Oh wait, we caught HAS_CUDA_Q = False. Let's patch it.
    import qocc.adapters.cudaq_adapter as cudaq_mod
    cudaq_mod.HAS_CUDA_Q = True
    
    # We still need to patch cudaq
    import sys
    from unittest.mock import MagicMock
    sys.modules['cudaq'] = MagicMock()
    cudaq_mod.cudaq = sys.modules['cudaq']
    
    # Mock sample results
    cudaq_mod.cudaq.sample.return_value = {"00": 100}
    
    from qocc.adapters.base import SimulationSpec
    sim_res = adapter.simulate(handle, SimulationSpec(shots=100))
    
    assert "vram_usage_mb" in sim_res.metadata
    
    budget = spec.resource_budget.get("max_vram_mb")
    assert sim_res.metadata["vram_usage_mb"] <= budget

    # cleanup mock
    del sys.modules['cudaq']
    cudaq_mod.HAS_CUDA_Q = False
