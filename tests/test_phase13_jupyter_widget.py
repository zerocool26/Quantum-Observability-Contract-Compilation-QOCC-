"""Phase 13 tests: Jupyter widget integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class _FakeWidget:
    def __init__(self, *children, **kwargs):
        if children and len(children) == 1 and isinstance(children[0], (list, tuple)):
            self.children = list(children[0])
        else:
            self.children = list(children) if children else list(kwargs.get("children", []))
        self.kwargs = kwargs


class _FakeWidgets:
    class VBox(_FakeWidget):
        pass

    class HBox(_FakeWidget):
        pass

    class HTML(_FakeWidget):
        pass

    class FloatSlider(_FakeWidget):
        pass


class _FakeFigure:
    def __init__(self):
        self.traces = []
        self.layout = {}

    def add_trace(self, trace):
        self.traces.append(trace)

    def update_layout(self, **kwargs):
        self.layout.update(kwargs)


class _FakeGo:
    class Figure(_FakeFigure):
        pass

    class FigureWidget(_FakeWidget):
        def __init__(self, fig):
            super().__init__(fig=fig)
            self.fig = fig

    class Bar(dict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    class Scatter(dict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)


def _write_bundle(root: Path, run_id: str = "run-j", adapter: str = "qiskit") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "created_at": "2026-02-26T00:00:00Z",
                "run_id": run_id,
                "adapter": adapter,
            }
        ),
        encoding="utf-8",
    )
    (root / "env.json").write_text(json.dumps({"os": "x", "python": "3.11"}), encoding="utf-8")
    (root / "seeds.json").write_text(json.dumps({"global_seed": 1, "rng_algorithm": "PCG64"}), encoding="utf-8")
    (root / "metrics.json").write_text(
        json.dumps({"compiled": {"depth": 5, "gates_2q": 3}}),
        encoding="utf-8",
    )
    (root / "contracts.json").write_text(json.dumps([]), encoding="utf-8")
    (root / "contract_results.json").write_text(json.dumps([]), encoding="utf-8")
    (root / "trace.jsonl").write_text(
        json.dumps(
            {
                "trace_id": "t",
                "span_id": "s",
                "name": "compile",
                "start_time": "2026-02-26T00:00:00Z",
                "end_time": "2026-02-26T00:00:01Z",
                "attributes": {"x": 1},
                "events": [],
                "links": [],
                "status": "OK",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def test_show_bundle_returns_widget_container(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from qocc.trace import jupyter_widget as jw

    bundle = _write_bundle(tmp_path / "bundleA")
    monkeypatch.setattr(jw, "_require_widget_deps", lambda: (_FakeWidgets, _FakeGo))

    view = jw.show_bundle(str(bundle))
    assert isinstance(view, _FakeWidgets.VBox)
    assert len(view.children) == 2


def test_compare_interactive_returns_metric_sliders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from qocc.trace import jupyter_widget as jw

    a = _write_bundle(tmp_path / "bundleB", run_id="a")
    b = _write_bundle(tmp_path / "bundleC", run_id="b")
    (b / "metrics.json").write_text(json.dumps({"compiled": {"depth": 9, "gates_2q": 4}}), encoding="utf-8")

    monkeypatch.setattr(jw, "_require_widget_deps", lambda: (_FakeWidgets, _FakeGo))
    view = jw.compare_interactive(str(a), str(b))

    assert isinstance(view, _FakeWidgets.VBox)
    assert len(view.children) >= 2


def test_search_dashboard_builds_scatter(monkeypatch: pytest.MonkeyPatch) -> None:
    from qocc.trace import jupyter_widget as jw

    monkeypatch.setattr(jw, "_require_widget_deps", lambda: (_FakeWidgets, _FakeGo))
    result = {
        "top_rankings": [
            {"candidate_id": "c1", "surrogate_score": 1.2, "metrics": {"depth": 5, "gates_2q": 2}},
            {"candidate_id": "c2", "surrogate_score": 0.9, "metrics": {"depth": 4, "gates_2q": 3}},
        ]
    }

    view = jw.search_dashboard(result)
    assert isinstance(view, _FakeWidgets.VBox)
    assert len(view.children) == 2


def test_widget_import_error_has_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    from qocc.trace import jupyter_widget as jw

    def _boom():
        raise ImportError("nope")

    monkeypatch.setattr(jw, "_require_widget_deps", _boom)

    with pytest.raises(ImportError):
        jw.show_bundle("missing")


def test_top_level_qocc_exports_widget_helpers() -> None:
    import qocc

    assert hasattr(qocc, "show_bundle")
    assert hasattr(qocc, "compare_interactive")
    assert hasattr(qocc, "search_dashboard")
