"""Generate docs/api_reference.md from package docstrings/signatures."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any

MODULES = [
    "qocc.api",
    "qocc.core.circuit_handle",
    "qocc.adapters.base",
    "qocc.contracts.spec",
    "qocc.search.space",
]

OUTPUT = Path(__file__).resolve().parent / "api_reference.md"


def _public_members(module: Any) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for name, obj in inspect.getmembers(module):
        if name.startswith("_"):
            continue
        if inspect.isfunction(obj) and obj.__module__ == module.__name__:
            out.append((name, obj))
        elif inspect.isclass(obj) and obj.__module__ == module.__name__:
            out.append((name, obj))
    return out


def _emit_function(name: str, fn: Any) -> list[str]:
    sig = str(inspect.signature(fn))
    doc = inspect.getdoc(fn) or "No documentation provided."
    first = doc.splitlines()[0] if doc else ""
    return [f"### `{name}{sig}`", "", first, ""]


def _emit_class(name: str, cls: Any) -> list[str]:
    lines = [f"### `class {name}`", ""]
    doc = inspect.getdoc(cls) or "No documentation provided."
    lines.append(doc.splitlines()[0] if doc else "")
    lines.append("")

    methods = []
    for method_name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if method_name.startswith("_"):
            continue
        if method.__qualname__.split(".")[0] != cls.__name__:
            continue
        methods.append((method_name, method))

    if methods:
        lines.append("Methods:")
        lines.append("")
        for method_name, method in methods:
            msig = str(inspect.signature(method))
            mdoc = inspect.getdoc(method) or ""
            first = mdoc.splitlines()[0] if mdoc else ""
            lines.append(f"- `{method_name}{msig}` — {first}")
        lines.append("")

    return lines


def generate() -> str:
    lines = [
        "# API Reference",
        "",
        "This file is generated from docstrings and signatures.",
        "",
    ]

    for module_name in MODULES:
        module = importlib.import_module(module_name)
        lines.append(f"## `{module_name}`")
        lines.append("")
        module_doc = inspect.getdoc(module) or "No module documentation provided."
        lines.append(module_doc.splitlines()[0] if module_doc else "")
        lines.append("")

        for name, obj in _public_members(module):
            if inspect.isfunction(obj):
                lines.extend(_emit_function(name, obj))
            elif inspect.isclass(obj):
                lines.extend(_emit_class(name, obj))

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    OUTPUT.write_text(generate(), encoding="utf-8")
    print(f"Generated {OUTPUT}")


if __name__ == "__main__":
    main()
