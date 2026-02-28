from qocc.contracts.spec import ContractSpec, ContractResult
from qocc.core.circuit_handle import CircuitHandle
from typing import Any

def evaluate_topology_contract(
    spec: ContractSpec,
    compiled_metrics: dict[str, Any]
) -> ContractResult:
    """
    Evaluates whether a circuit complies with the topology rules setup in the contract.
    Passes if `topology_violations` (or additional SWAP gates added) is <= budget.
    """
    budget = spec.spec.get("max_swap_insertions", 0)
    
    # Ideally the compilation pipeline emits `topology_violations` or we compare swap gates added
    violations = compiled_metrics.get("topology_violations", 0)
    
    passed = violations <= budget
    
    return ContractResult(
        name=spec.name,
        passed=passed,
        details={
            "type": "topology_violations",
            "budget": budget,
            "actual_violations": violations,
        }
    )
