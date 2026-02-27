"""Contract composition operators.

Supports:
- all_of([c1, c2, ...])
- any_of([c1, c2, ...])
- best_effort(contract)
- with_fallback(primary, fallback)
"""

from __future__ import annotations

from typing import Any, Callable

from qocc.contracts.spec import ContractResult

LeafEvaluator = Callable[[dict[str, Any]], ContractResult]


def iter_leaf_contract_dicts(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten composition entries to leaf contract dicts."""
    out: list[dict[str, Any]] = []
    for entry in entries:
        out.extend(_iter_leaf(entry))
    return out


def _iter_leaf(entry: dict[str, Any]) -> list[dict[str, Any]]:
    op = entry.get("op") if isinstance(entry, dict) else None
    if not op:
        return [entry]

    op = str(op)
    if op in ("all_of", "any_of"):
        contracts = entry.get("contracts", [])
        if not isinstance(contracts, list):
            return []
        out: list[dict[str, Any]] = []
        for c in contracts:
            if isinstance(c, dict):
                out.extend(_iter_leaf(c))
        return out

    if op == "best_effort":
        c = entry.get("contract")
        if c is None:
            contracts = entry.get("contracts", [])
            if isinstance(contracts, list) and contracts:
                c = contracts[0]
        return _iter_leaf(c) if isinstance(c, dict) else []

    if op == "with_fallback":
        primary = entry.get("primary")
        fallback = entry.get("fallback")
        if primary is None or fallback is None:
            contracts = entry.get("contracts", [])
            if isinstance(contracts, list) and len(contracts) >= 2:
                primary, fallback = contracts[0], contracts[1]
        out: list[dict[str, Any]] = []
        if isinstance(primary, dict):
            out.extend(_iter_leaf(primary))
        if isinstance(fallback, dict):
            out.extend(_iter_leaf(fallback))
        return out

    return []


def evaluate_contract_entry(entry: dict[str, Any], evaluate_leaf: LeafEvaluator) -> ContractResult:
    """Evaluate a contract entry that can be a leaf or composition envelope."""
    if not isinstance(entry, dict):
        return ContractResult(name="invalid", passed=False, details={"error": "Contract entry must be an object"})

    op = entry.get("op")
    if not op:
        return evaluate_leaf(entry)

    op = str(op)
    if op == "all_of":
        return _eval_all_of(entry, evaluate_leaf)
    if op == "any_of":
        return _eval_any_of(entry, evaluate_leaf)
    if op == "best_effort":
        return _eval_best_effort(entry, evaluate_leaf)
    if op == "with_fallback":
        return _eval_with_fallback(entry, evaluate_leaf)

    return ContractResult(
        name=str(entry.get("name") or "unknown_op"),
        passed=False,
        details={"type": "composition", "error": f"Unknown composition op: {op!r}"},
    )


def _eval_all_of(entry: dict[str, Any], evaluate_leaf: LeafEvaluator) -> ContractResult:
    contracts = entry.get("contracts", [])
    if not isinstance(contracts, list):
        return ContractResult(
            name=str(entry.get("name") or "all_of"),
            passed=False,
            details={"type": "composition", "op": "all_of", "error": "'contracts' must be an array"},
        )

    children = [evaluate_contract_entry(c, evaluate_leaf).to_dict() for c in contracts if isinstance(c, dict)]
    passed = all(bool(c.get("passed", False)) for c in children) if children else False
    return ContractResult(
        name=str(entry.get("name") or "all_of"),
        passed=passed,
        details={"type": "composition", "op": "all_of", "children": children},
    )


def _eval_any_of(entry: dict[str, Any], evaluate_leaf: LeafEvaluator) -> ContractResult:
    contracts = entry.get("contracts", [])
    if not isinstance(contracts, list):
        return ContractResult(
            name=str(entry.get("name") or "any_of"),
            passed=False,
            details={"type": "composition", "op": "any_of", "error": "'contracts' must be an array"},
        )

    children = [evaluate_contract_entry(c, evaluate_leaf).to_dict() for c in contracts if isinstance(c, dict)]
    passed = any(bool(c.get("passed", False)) for c in children) if children else False
    return ContractResult(
        name=str(entry.get("name") or "any_of"),
        passed=passed,
        details={"type": "composition", "op": "any_of", "children": children},
    )


def _eval_best_effort(entry: dict[str, Any], evaluate_leaf: LeafEvaluator) -> ContractResult:
    contract = entry.get("contract")
    if contract is None:
        contracts = entry.get("contracts", [])
        if isinstance(contracts, list) and contracts:
            contract = contracts[0]

    if not isinstance(contract, dict):
        return ContractResult(
            name=str(entry.get("name") or "best_effort"),
            passed=True,
            details={"type": "composition", "op": "best_effort", "error": "missing contract"},
        )

    child = evaluate_contract_entry(contract, evaluate_leaf).to_dict()
    return ContractResult(
        name=str(entry.get("name") or "best_effort"),
        passed=True,
        details={
            "type": "composition",
            "op": "best_effort",
            "effective_passed": bool(child.get("passed", False)),
            "child": child,
        },
    )


def _eval_with_fallback(entry: dict[str, Any], evaluate_leaf: LeafEvaluator) -> ContractResult:
    primary = entry.get("primary")
    fallback = entry.get("fallback")

    if primary is None or fallback is None:
        contracts = entry.get("contracts", [])
        if isinstance(contracts, list) and len(contracts) >= 2:
            primary, fallback = contracts[0], contracts[1]

    if not isinstance(primary, dict) or not isinstance(fallback, dict):
        return ContractResult(
            name=str(entry.get("name") or "with_fallback"),
            passed=False,
            details={
                "type": "composition",
                "op": "with_fallback",
                "error": "requires primary and fallback contract objects",
            },
        )

    p = evaluate_contract_entry(primary, evaluate_leaf).to_dict()
    err = str(p.get("details", {}).get("error", ""))
    use_fallback = "NotImplementedError" in err

    if use_fallback:
        f = evaluate_contract_entry(fallback, evaluate_leaf).to_dict()
        return ContractResult(
            name=str(entry.get("name") or "with_fallback"),
            passed=bool(f.get("passed", False)),
            details={
                "type": "composition",
                "op": "with_fallback",
                "used_fallback": True,
                "primary": p,
                "fallback": f,
            },
        )

    return ContractResult(
        name=str(entry.get("name") or "with_fallback"),
        passed=bool(p.get("passed", False)),
        details={
            "type": "composition",
            "op": "with_fallback",
            "used_fallback": False,
            "primary": p,
        },
    )
