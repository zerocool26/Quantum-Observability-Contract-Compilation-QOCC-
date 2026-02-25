"""Trace exporters — convert collected spans to various formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qocc.trace.span import Span


def export_jsonl(spans: list[Span], path: str | Path) -> Path:
    """Write spans as JSON Lines."""
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        for s in spans:
            f.write(json.dumps(s.to_dict(), default=str) + "\n")
    return p


def spans_to_dicts(spans: list[Span]) -> list[dict[str, Any]]:
    """Convert spans to list of dicts."""
    return [s.to_dict() for s in spans]
