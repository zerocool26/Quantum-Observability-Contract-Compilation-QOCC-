"""Parametric contract expression resolution."""

from __future__ import annotations

import ast
from copy import deepcopy
from typing import Any

from qocc.contracts.spec import ContractSpec


class ParametricResolutionError(ValueError):
    """Raised when a parametric expression cannot be resolved."""


def resolve_contract_spec(
    spec: ContractSpec,
    *,
    bundle: dict[str, Any],
    input_metrics: dict[str, Any],
    compiled_metrics: dict[str, Any],
) -> ContractSpec:
    """Resolve expression-valued fields in a ContractSpec at evaluation time."""
    context = _build_context(spec, bundle, input_metrics, compiled_metrics)

    out = ContractSpec.from_dict(spec.to_dict())
    out.tolerances = _resolve_dict(out.tolerances, context, namespace="tolerances")
    out.resource_budget = _resolve_dict(out.resource_budget, context, namespace="resource_budget")
    out.confidence = _resolve_dict(out.confidence, context, namespace="confidence")

    out.spec = deepcopy(out.spec)
    out.spec = _resolve_spec_payload(out.spec, context)
    return out


def _resolve_spec_payload(payload: dict[str, Any], context: dict[str, float]) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key, value in payload.items():
        if key in ("assertions", "contexts") and isinstance(value, list):
            new_rows = []
            for row in value:
                if not isinstance(row, dict):
                    new_rows.append(row)
                    continue
                row_new = dict(row)
                if "threshold" in row_new:
                    row_new["threshold"] = _resolve_value(row_new["threshold"], context, f"spec.{key}.threshold")
                if "context_value" in row_new:
                    row_new["context_value"] = _resolve_value(row_new["context_value"], context, f"spec.{key}.context_value")
                new_rows.append(row_new)
            resolved[key] = new_rows
            continue

        resolved[key] = _resolve_value(value, context, f"spec.{key}")
    return resolved


def _resolve_dict(values: dict[str, Any], context: dict[str, float], namespace: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in values.items():
        out[key] = _resolve_value(value, context, f"{namespace}.{key}")
        if isinstance(out[key], (int, float)):
            context[key] = float(out[key])
    return out


def _resolve_value(value: Any, context: dict[str, float], label: str) -> Any:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return value

    expr = value.strip()
    if not expr:
        return value

    try:
        return _safe_eval(expr, context)
    except Exception as exc:
        raise ParametricResolutionError(f"Failed to resolve {label}: {exc}") from exc


def _build_context(
    spec: ContractSpec,
    bundle: dict[str, Any],
    input_metrics: dict[str, Any],
    compiled_metrics: dict[str, Any],
) -> dict[str, float]:
    ctx: dict[str, float] = {}

    for key, value in input_metrics.items():
        if isinstance(value, (int, float)):
            ctx[f"input_{key}"] = float(value)

    for key, value in compiled_metrics.items():
        if isinstance(value, (int, float)):
            ctx[key] = float(value)
            ctx[f"compiled_{key}"] = float(value)

    baseline = bundle.get("baseline_metrics")
    if not isinstance(baseline, dict):
        baseline = bundle.get("metrics", {}).get("baseline", {})
    if not isinstance(baseline, dict):
        baseline = {}

    for key, value in baseline.items():
        if isinstance(value, (int, float)):
            ctx[f"baseline_{key}"] = float(value)

    # Allow symbolic references to numeric fields within the same contract.
    for src in (spec.spec, spec.tolerances, spec.resource_budget, spec.confidence):
        for key, value in src.items():
            if isinstance(value, (int, float)):
                ctx[key] = float(value)

    return ctx


def _safe_eval(expr: str, context: dict[str, float]) -> float:
    node = ast.parse(expr, mode="eval")
    return float(_eval_node(node.body, context))


def _eval_node(node: ast.AST, context: dict[str, float]) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Only numeric constants are allowed")

    if isinstance(node, ast.Name):
        if node.id not in context:
            raise ValueError(f"Unknown symbol '{node.id}'")
        return float(context[node.id])

    if isinstance(node, ast.UnaryOp):
        val = _eval_node(node.operand, context)
        if isinstance(node.op, ast.USub):
            return -val
        if isinstance(node.op, ast.UAdd):
            return val
        raise ValueError("Unsupported unary operator")

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, context)
        right = _eval_node(node.right, context)

        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left ** right
        if isinstance(node.op, ast.Mod):
            return left % right
        raise ValueError("Unsupported binary operator")

    raise ValueError("Unsupported expression")
