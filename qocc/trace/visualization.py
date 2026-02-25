"""Trace timeline visualization — ASCII flame chart for terminal display.

Renders trace spans as a flamechart / waterfall diagram in the terminal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def render_timeline(
    spans: list[dict[str, Any]],
    width: int = 80,
    show_attributes: bool = False,
) -> str:
    """Render spans as an ASCII timeline / waterfall.

    Parameters:
        spans: List of span dicts (as from trace.jsonl).
        width: Terminal width for the chart.
        show_attributes: Show span attributes inline.

    Returns:
        Multi-line string with the rendered timeline.
    """
    if not spans:
        return "(no spans)"

    # Parse timestamps
    parsed = []
    for s in spans:
        start = s.get("start_time", 0)
        end = s.get("end_time", start)
        if isinstance(start, str):
            try:
                from datetime import datetime

                start = datetime.fromisoformat(start).timestamp()
                end = datetime.fromisoformat(end).timestamp() if isinstance(end, str) else end
            except Exception:
                start = float(start) if start else 0
                end = float(end) if end else start
        parsed.append({
            "name": s.get("name", "?"),
            "start": float(start),
            "end": float(end),
            "parent_id": s.get("parent_id"),
            "span_id": s.get("span_id", ""),
            "attributes": s.get("attributes", {}),
            "status": s.get("status", "ok"),
        })

    # Calculate time bounds
    t_min = min(p["start"] for p in parsed)
    t_max = max(p["end"] for p in parsed)
    t_range = t_max - t_min

    if t_range == 0:
        t_range = 1.0  # avoid division by zero

    # Chart dimensions
    label_width = min(30, max(len(p["name"]) for p in parsed) + 4)
    bar_width = width - label_width - 12  # 12 for duration column
    if bar_width < 10:
        bar_width = 10

    lines: list[str] = []

    # Header
    total_ms = t_range * 1000
    lines.append(f"{'Span':<{label_width}} {'Timeline':<{bar_width}}  Duration")
    lines.append("─" * (label_width + bar_width + 12))

    # Determine nesting depth by parent relationships
    depth_map: dict[str, int] = {}
    id_to_span: dict[str, dict[str, Any]] = {}
    for p in parsed:
        id_to_span[p["span_id"]] = p

    def get_depth(sp: dict[str, Any]) -> int:
        sid = sp["span_id"]
        if sid in depth_map:
            return depth_map[sid]
        pid = sp.get("parent_id")
        if pid and pid in id_to_span:
            d = get_depth(id_to_span[pid]) + 1
        else:
            d = 0
        depth_map[sid] = d
        return d

    for p in parsed:
        get_depth(p)

    # Sort by start time, then depth
    sorted_spans = sorted(parsed, key=lambda p: (p["start"], depth_map.get(p["span_id"], 0)))

    # Render each span
    for p in sorted_spans:
        depth = depth_map.get(p["span_id"], 0)
        indent = "  " * min(depth, 5)
        name = p["name"]
        truncated_name = indent + name
        if len(truncated_name) > label_width - 1:
            truncated_name = truncated_name[: label_width - 4] + "..."

        # Bar position
        rel_start = (p["start"] - t_min) / t_range
        rel_end = (p["end"] - t_min) / t_range
        bar_start = int(rel_start * bar_width)
        bar_end = max(bar_start + 1, int(rel_end * bar_width))
        bar_end = min(bar_end, bar_width)

        bar = " " * bar_start + "█" * (bar_end - bar_start) + " " * (bar_width - bar_end)

        # Duration
        dur_s = p["end"] - p["start"]
        if dur_s >= 1.0:
            dur_str = f"{dur_s:.2f}s"
        else:
            dur_str = f"{dur_s * 1000:.1f}ms"

        # Status indicator
        status_char = "✓" if p["status"] in ("ok", "OK", "unset") else "✗"

        lines.append(f"{truncated_name:<{label_width}} {bar}  {dur_str:>7} {status_char}")

        if show_attributes and p["attributes"]:
            attrs_str = json.dumps(p["attributes"], default=str)
            if len(attrs_str) > width - 4:
                attrs_str = attrs_str[: width - 7] + "..."
            lines.append(f"{'':>{label_width}}   {attrs_str}")

    # Footer with total duration
    lines.append("─" * (label_width + bar_width + 12))
    lines.append(f"{'Total':<{label_width}} {'':>{bar_width}}  {total_ms:.1f}ms")

    return "\n".join(lines)


def render_timeline_from_bundle(
    bundle_path: str,
    width: int = 80,
    show_attributes: bool = False,
) -> str:
    """Render a timeline from a trace bundle.

    Parameters:
        bundle_path: Path to bundle zip or directory.
        width: Terminal width.
        show_attributes: Show attributes inline.

    Returns:
        Rendered timeline string.
    """
    from qocc.core.artifacts import ArtifactStore

    bundle = ArtifactStore.load_bundle(bundle_path)
    spans = bundle.get("trace", [])
    return render_timeline(spans, width=width, show_attributes=show_attributes)


def render_metrics_comparison(
    metrics_before: dict[str, Any],
    metrics_after: dict[str, Any],
    width: int = 60,
) -> str:
    """Render a side-by-side metrics comparison with ASCII bars.

    Parameters:
        metrics_before: Input circuit metrics.
        metrics_after: Compiled circuit metrics.
        width: Terminal width.

    Returns:
        Multi-line comparison string.
    """
    lines: list[str] = []
    lines.append(f"{'Metric':<20} {'Input':>10} {'Compiled':>10} {'Change':>10} {'Visual'}")
    lines.append("─" * width)

    numeric_keys = [
        "width", "depth", "total_gates", "gates_1q", "gates_2q",
        "depth_2q", "duration_estimate", "proxy_error_score",
    ]

    for key in numeric_keys:
        va = metrics_before.get(key)
        vb = metrics_after.get(key)
        if va is None and vb is None:
            continue

        va_num = float(va) if va is not None else 0
        vb_num = float(vb) if vb is not None else 0
        max_val = max(abs(va_num), abs(vb_num), 1)

        # Percentage change
        if va_num != 0:
            pct = ((vb_num - va_num) / abs(va_num)) * 100
            pct_str = f"{pct:+.1f}%"
        else:
            pct_str = "—"

        # Mini bar
        bar_len = 15
        bar_a = int((va_num / max_val) * bar_len)
        bar_b = int((vb_num / max_val) * bar_len)
        visual = "▓" * bar_a + "░" * (bar_len - bar_a) + " → " + "▓" * bar_b + "░" * (bar_len - bar_b)

        # Format values
        va_str = f"{va}" if va is not None else "—"
        vb_str = f"{vb}" if vb is not None else "—"

        lines.append(f"{key:<20} {va_str:>10} {vb_str:>10} {pct_str:>10} {visual}")

    return "\n".join(lines)
