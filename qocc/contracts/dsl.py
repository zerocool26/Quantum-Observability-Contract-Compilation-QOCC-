"""Contract DSL parser.

Parses `.qocc` contract files into `ContractSpec` dictionaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from qocc.contracts.spec import ContractSpec


@dataclass
class ContractDSLParseError(ValueError):
    """Syntax error in contract DSL with source location."""

    message: str
    line: int
    column: int

    def __str__(self) -> str:
        return f"DSL parse error at line {self.line}, column {self.column}: {self.message}"


_HEADER_RE = re.compile(r"^contract\s+([A-Za-z_][\w\-]*)\s*:\s*$")
_FIELD_RE = re.compile(r"^([A-Za-z_][\w\-]*)\s*:\s*(.+?)\s*$")
_NUM_RE = r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
_CONSTRAINT_RE = re.compile(r"^([A-Za-z_][\w]*)\s*(<=|>=|==|=|<|>)\s*(.+?)$")
_SHOTS_RE = re.compile(r"^(\d+)\s*\.\.\s*(\d+)$")


def parse_contract_dsl(text: str) -> list[ContractSpec]:
    """Parse Contract DSL text into `ContractSpec` objects."""
    lines = text.splitlines()
    out: list[ContractSpec] = []

    idx = 0
    while idx < len(lines):
        raw = lines[idx]
        stripped = _strip_comment(raw).strip()

        if not stripped:
            idx += 1
            continue

        header = _HEADER_RE.match(stripped)
        if not header:
            raise ContractDSLParseError(
                "Expected 'contract <name>:'",
                line=idx + 1,
                column=1,
            )

        name = header.group(1)
        block_start = idx + 1
        idx += 1

        block_rows: list[tuple[int, str]] = []
        while idx < len(lines):
            line = lines[idx]
            line_wo_comment = _strip_comment(line)
            stripped_line = line_wo_comment.strip()

            if not stripped_line:
                idx += 1
                continue

            # New contract starts -> stop current block
            if _HEADER_RE.match(stripped_line):
                break

            indent = len(line_wo_comment) - len(line_wo_comment.lstrip(" \t"))
            if indent == 0:
                raise ContractDSLParseError(
                    "Contract fields must be indented under the contract header",
                    line=idx + 1,
                    column=1,
                )

            block_rows.append((idx + 1, stripped_line))
            idx += 1

        if not block_rows:
            raise ContractDSLParseError(
                "Contract block is empty",
                line=block_start + 1,
                column=1,
            )

        out.append(_parse_block(name=name, rows=block_rows))

    return out


def _parse_block(name: str, rows: list[tuple[int, str]]) -> ContractSpec:
    ctype: str | None = None
    tolerances: dict[str, float] = {}
    confidence: dict[str, float] = {}
    resource_budget: dict[str, Any] = {}
    spec: dict[str, Any] = {}
    evaluator = "auto"

    for line_no, row in rows:
        m = _FIELD_RE.match(row)
        if not m:
            raise ContractDSLParseError(
                "Expected '<field>: <value>'",
                line=line_no,
                column=1,
            )

        key = m.group(1).lower()
        value = m.group(2).strip()

        if key == "type":
            ctype = value
            continue

        if key == "confidence":
            try:
                confidence["level"] = float(value)
            except ValueError as exc:
                raise ContractDSLParseError(
                    "confidence must be numeric",
                    line=line_no,
                    column=row.index(value) + 1,
                ) from exc
            continue

        if key == "shots":
            shots_m = _SHOTS_RE.match(value)
            if not shots_m:
                raise ContractDSLParseError(
                    "shots must use range syntax: <min> .. <max>",
                    line=line_no,
                    column=row.index(value) + 1,
                )
            smin = int(shots_m.group(1))
            smax = int(shots_m.group(2))
            if smax < smin:
                raise ContractDSLParseError(
                    "shots upper bound must be >= lower bound",
                    line=line_no,
                    column=row.index(value) + 1,
                )
            resource_budget["min_shots"] = smin
            resource_budget["max_shots"] = smax
            continue

        if key in ("tolerance", "assert"):
            parsed = _parse_constraint_expr(value, line_no, row.index(value) + 1)
            _apply_constraint(parsed, tolerances, resource_budget, spec)
            continue

        if key == "evaluator":
            evaluator = value
            continue

        raise ContractDSLParseError(
            f"Unknown field '{key}'",
            line=line_no,
            column=1,
        )

    if not ctype:
        raise ContractDSLParseError(
            "Missing required field 'type'",
            line=rows[0][0],
            column=1,
        )

    return ContractSpec(
        name=name,
        type=ctype,
        spec=spec,
        tolerances=tolerances,
        confidence=confidence,
        resource_budget=resource_budget,
        evaluator=evaluator,
    )


def _parse_constraint_expr(value: str, line: int, column: int) -> dict[str, Any]:
    if "@" in value:
        expr_part, context_part = value.split("@", 1)
        context_part = context_part.strip()
    else:
        expr_part = value
        context_part = ""

    m = _CONSTRAINT_RE.match(expr_part.strip())
    if not m:
        raise ContractDSLParseError(
            "Invalid constraint expression; expected: <metric> <op> <value/expression>",
            line=line,
            column=column,
        )

    metric = m.group(1)
    op = m.group(2)
    rhs = m.group(3).strip()
    threshold: float | str
    try:
        threshold = float(rhs)
    except ValueError:
        threshold = rhs

    context_metric: str | None = None
    context_value: float | str | None = None
    if context_part:
        if "=" not in context_part:
            raise ContractDSLParseError(
                "Invalid context expression after '@'; expected <name> = <value/expression>",
                line=line,
                column=column,
            )
        c_name, c_expr = context_part.split("=", 1)
        context_metric = c_name.strip()
        c_rhs = c_expr.strip()
        try:
            context_value = float(c_rhs)
        except ValueError:
            context_value = c_rhs

    return {
        "metric": metric,
        "op": op,
        "threshold": threshold,
        "context_metric": context_metric,
        "context_value": context_value,
    }


def _apply_constraint(
    parsed: dict[str, Any],
    tolerances: dict[str, float],
    resource_budget: dict[str, Any],
    spec: dict[str, Any],
) -> None:
    metric = parsed["metric"]
    op = parsed["op"]
    threshold = parsed["threshold"]

    if parsed.get("context_metric"):
        contexts = spec.setdefault("contexts", [])
        contexts.append(
            {
                "metric": metric,
                "context_metric": parsed["context_metric"],
                "context_value": parsed["context_value"],
                "op": op,
                "threshold": threshold,
            }
        )

    # Common direct mappings for current evaluators.
    if metric == "tvd" and op in ("<=", "<"):
        tolerances["tvd"] = threshold
        return

    if metric == "logical_error_rate" and op in ("<=", "<"):
        tolerances["logical_error_rate_threshold"] = threshold
        return

    if metric == "code_distance" and op in (">=", ">"):
        tolerances["code_distance"] = threshold
        return

    if metric in ("syndrome_weight", "syndrome_weight_budget") and op in ("<=", "<"):
        tolerances["syndrome_weight_budget"] = threshold
        return

    if metric == "depth" and op in ("<=", "<"):
        resource_budget["max_depth"] = threshold
        return

    if metric in ("two_qubit_gates", "gates_2q") and op in ("<=", "<"):
        resource_budget["max_gates_2q"] = threshold
        return

    if metric == "total_gates" and op in ("<=", "<"):
        resource_budget["max_total_gates"] = threshold
        return

    if metric in ("duration", "duration_ns") and op in ("<=", "<"):
        resource_budget["max_duration_ns"] = threshold
        return

    if metric in ("proxy_error", "proxy_error_score") and op in ("<=", "<"):
        tolerances["max_proxy_error"] = threshold
        return

    extra = spec.setdefault("assertions", [])
    extra.append(parsed)


def _strip_comment(line: str) -> str:
    if "#" not in line:
        return line
    return line.split("#", 1)[0]
