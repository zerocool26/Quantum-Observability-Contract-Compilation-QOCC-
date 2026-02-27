"""Contracts subpackage — semantic and cost contract evaluation."""

from __future__ import annotations

__all__ = [
    "ContractSpec",
    "ContractResult",
    "ContractType",
    "evaluate_distribution_contract",
    "evaluate_observable_contract",
    "evaluate_clifford_contract",
    "evaluate_exact_equivalence",
    "evaluate_qec_contract",
    "parse_contract_dsl",
    "resolve_contract_spec",
    "evaluate_contract_entry",
]

from qocc.contracts.spec import ContractResult, ContractSpec, ContractType
from qocc.contracts.eval_sampling import (
    evaluate_distribution_contract,
    evaluate_observable_contract,
)
from qocc.contracts.eval_clifford import evaluate_clifford_contract
from qocc.contracts.eval_exact import evaluate_exact_equivalence
from qocc.contracts.eval_qec import evaluate_qec_contract
from qocc.contracts.dsl import parse_contract_dsl
from qocc.contracts.parametric import resolve_contract_spec
from qocc.contracts.composition import evaluate_contract_entry
