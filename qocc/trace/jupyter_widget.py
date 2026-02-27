"""Jupyter widget integration for QOCC trace/search visualization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qocc.core.artifacts import ArtifactStore


def show_bundle(bundle_path: str) -> Any:
    """Render an interactive bundle flame chart view in Jupyter.

    Returns an ipywidgets container.
    """
    widgets, go = _require_widget_deps()
    bundle = ArtifactStore.load_bundle(bundle_path)
    spans = bundle.get("trace", [])
    fig = _build_flame_figure(spans, go)

    header = widgets.HTML(
        value=(
            f"<b>Run:</b> {bundle.get('manifest', {}).get('run_id', 'unknown')} "
            f"<span style='opacity:.7'>({bundle.get('manifest', {}).get('adapter', 'unknown')})</span>"
        )
    )
    fig_widget = _to_figure_widget(fig, go)
    return widgets.VBox([header, fig_widget])


def compare_interactive(bundle_a: str, bundle_b: str) -> Any:
    """Render side-by-side metric sliders for two bundles in Jupyter."""
    widgets, _go = _require_widget_deps()

    a = ArtifactStore.load_bundle(bundle_a)
    b = ArtifactStore.load_bundle(bundle_b)
    ma = a.get("metrics", {}).get("compiled", {})
    mb = b.get("metrics", {}).get("compiled", {})

    numeric_keys = []
    for key in sorted(set(ma) | set(mb)):
        va = ma.get(key)
        vb = mb.get(key)
        if isinstance(va, (int, float)) or isinstance(vb, (int, float)):
            numeric_keys.append(key)

    rows: list[Any] = [widgets.HTML(value="<b>Compiled metric comparison</b>")]
    if not numeric_keys:
        rows.append(widgets.HTML(value="<i>No numeric compiled metrics available.</i>"))
        return widgets.VBox(rows)

    for key in numeric_keys:
        va = float(ma.get(key, 0.0) or 0.0)
        vb = float(mb.get(key, 0.0) or 0.0)
        vmax = max(1.0, va, vb)

        label = widgets.HTML(value=f"<div style='min-width:140px'><b>{key}</b></div>")
        sa = widgets.FloatSlider(value=va, min=0.0, max=vmax, step=max(vmax / 200.0, 0.001), description="A", disabled=True)
        sb = widgets.FloatSlider(value=vb, min=0.0, max=vmax, step=max(vmax / 200.0, 0.001), description="B", disabled=True)
        rows.append(widgets.HBox([label, sa, sb]))

    return widgets.VBox(rows)


def search_dashboard(search_result: dict[str, Any]) -> Any:
    """Render candidate Pareto-style scatter view with hover tooltips."""
    widgets, go = _require_widget_deps()

    candidates = search_result.get("top_rankings") or search_result.get("rankings") or []
    if not isinstance(candidates, list):
        candidates = []

    x_vals: list[float] = []
    y_vals: list[float] = []
    texts: list[str] = []

    for idx, c in enumerate(candidates):
        metrics = c.get("metrics", {}) if isinstance(c, dict) else {}
        depth = float(metrics.get("depth", 0.0) or 0.0)
        gates_2q = float(metrics.get("gates_2q", 0.0) or 0.0)
        score = c.get("surrogate_score", None) if isinstance(c, dict) else None
        cid = c.get("candidate_id", f"candidate_{idx + 1}") if isinstance(c, dict) else f"candidate_{idx + 1}"
        texts.append(
            json.dumps(
                {
                    "candidate_id": cid,
                    "depth": depth,
                    "gates_2q": gates_2q,
                    "surrogate_score": score,
                }
            )
        )
        x_vals.append(depth)
        y_vals.append(gates_2q)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            text=texts,
            hovertemplate="%{text}<extra></extra>",
            marker={"size": 10, "color": "#3b82f6"},
            name="Candidates",
        )
    )
    fig.update_layout(
        title="Search Candidate Dashboard",
        xaxis_title="Depth",
        yaxis_title="2Q Gates",
        template="plotly_white",
    )

    title = widgets.HTML(value="<b>Pareto Scatter (Depth vs 2Q Gates)</b>")
    return widgets.VBox([title, _to_figure_widget(fig, go)])


def _to_figure_widget(fig: Any, go: Any) -> Any:
    FigureWidget = getattr(go, "FigureWidget", None)
    if FigureWidget is None:
        return fig
    return FigureWidget(fig)


def _build_flame_figure(spans: list[dict[str, Any]], go: Any) -> Any:
    parsed: list[dict[str, Any]] = []
    for s in spans:
        start = _parse_time(s.get("start_time"))
        end = _parse_time(s.get("end_time", s.get("start_time")))
        if end < start:
            end = start
        parsed.append(
            {
                "name": s.get("name", "?"),
                "start": start,
                "duration": max(0.0, end - start),
                "attrs": s.get("attributes", {}),
            }
        )

    parsed.sort(key=lambda x: x["start"])
    labels = [p["name"] for p in parsed]
    starts = [p["start"] for p in parsed]
    durations = [p["duration"] for p in parsed]
    hover = [json.dumps(p["attrs"], default=str) for p in parsed]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=labels,
            x=durations,
            base=starts,
            orientation="h",
            marker={"color": "#3b82f6"},
            text=hover,
            hovertemplate="%{y}<br>%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Trace Flame Chart",
        xaxis_title="Time",
        yaxis_title="Span",
        template="plotly_white",
        barmode="overlay",
        height=max(320, 30 * max(1, len(labels))),
    )
    return fig


def _parse_time(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            from datetime import datetime

            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except Exception:
            try:
                return float(value)
            except Exception:
                return 0.0
    return 0.0


def _require_widget_deps() -> tuple[Any, Any]:
    try:
        import ipywidgets as widgets  # type: ignore[import-untyped]
        import plotly.graph_objects as go  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "Jupyter widget dependencies are not installed. Install with: pip install 'qocc[jupyter]'"
        ) from exc
    return widgets, go
