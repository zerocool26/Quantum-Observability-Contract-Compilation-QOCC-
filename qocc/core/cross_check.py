import logging
from typing import Any, Dict, List
import json
from pathlib import Path

from qocc.adapters.base import SimulationSpec
from qocc.contracts.spec import ContractSpec, ContractResult
from qocc.core.artifacts import ArtifactStore
from qocc.adapters.base import get_adapter

logger = logging.getLogger(__name__)

def cross_check(
    circuit_source: Any,
    adapters: List[str],
    contract_specs: List[Dict[str, Any]],
    simulation_shots: int = 1000,
    simulation_seed: int = 42
) -> Dict[str, Any]:
    """
    Simulates the same circuit across multiple adapters and cross-checks their outputs.
    
    Returns a compatibility matrix mapping (adapter_i, adapter_j) -> List[ContractResult].
    """
    
    # Ingest and simulate across all requested adapters
    results: Dict[str, Any] = {}
    handles: Dict[str, Any] = {}
    metric_snapshots: Dict[str, Any] = {}
    
    # 1. Gather all simulated outputs
    for adapter_name in adapters:
        try:
            adapter = get_adapter(adapter_name)
            handle = adapter.ingest(circuit_source)
            handle = adapter.normalize_circuit(handle)
            sim = adapter.simulate(handle, SimulationSpec(shots=simulation_shots, seed=simulation_seed))
            results[adapter_name] = {"counts": sim.counts, "statevector": sim.metadata.get("statevector")}
            handles[adapter_name] = handle
            metric_snapshots[adapter_name] = adapter.get_metrics(handle).to_dict()
        except Exception as e:
            logger.error(f"Failed to simulate input with adapter {adapter_name}: {e}")
            results[adapter_name] = None
            
    # 2. Cross-check each pair
    # We evaluate adapter_a as the "baseline / ideal" and adapter_b as the "compiled / candidate"
    from qocc.contracts.eval_sampling import evaluate_distribution_contract, evaluate_observable_contract, _counts_to_observable_values
    from qocc.contracts.eval_exact import evaluate_exact_equivalence
    from qocc.contracts.eval_clifford import evaluate_clifford_contract
    from qocc.contracts.eval_qpt import evaluate_qpt_contract
    
    matrix: Dict[str, Dict[str, list[Dict[str, Any]]]] = {}
    
    for adapter_a in adapters:
        matrix[adapter_a] = {}
        for adapter_b in adapters:
            matrix[adapter_a][adapter_b] = []
            
            if results[adapter_a] is None or results[adapter_b] is None:
                matrix[adapter_a][adapter_b] = [{"error": "Missing simulation result"}]
                continue
                
            pair_results = []
            for spec_dict in contract_specs:
                try:
                    spec = ContractSpec.from_dict(spec_dict)
                    res_obj: ContractResult
                    
                    if spec.type == "distribution":
                        res_obj = evaluate_distribution_contract(spec, results[adapter_a]["counts"], results[adapter_b]["counts"])
                    elif spec.type == "observable":
                        vals_a = _counts_to_observable_values(results[adapter_a]["counts"])
                        vals_b = _counts_to_observable_values(results[adapter_b]["counts"])
                        res_obj = evaluate_observable_contract(spec, vals_a, vals_b)
                    elif spec.type == "exact":
                        res_obj = evaluate_exact_equivalence(spec, results[adapter_a]["statevector"], results[adapter_b]["statevector"])
                    elif spec.type == "clifford":
                        res_obj = evaluate_clifford_contract(
                            spec, handles[adapter_a], handles[adapter_b],
                            counts_before=results[adapter_a]["counts"], counts_after=results[adapter_b]["counts"]
                        )
                    else:
                        res_obj = ContractResult(
                            name=spec.name,
                            passed=False,
                            details={"error": f"Contract type {spec.type} not supported in cross-check yet"}
                        )
                        
                    pair_results.append(res_obj.to_dict())
                except Exception as e:
                    pair_results.append({"name": spec_dict.get("name", "unknown"), "passed": False, "details": {"error": str(e)}})
            
            matrix[adapter_a][adapter_b] = pair_results

    return {
        "adapters": adapters,
        "matrix": matrix,
        "metrics": metric_snapshots
    }
