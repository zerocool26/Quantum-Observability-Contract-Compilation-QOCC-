"""Tests for regression analysis in bundle comparison."""

from __future__ import annotations

import pytest

from qocc.api import compare_bundles, _analyze_regression_causes


def _make_bundle(
    run_id: str = "test",
    metrics: dict | None = None,
    env: dict | None = None,
    seeds: dict | None = None,
    pipeline: dict | None = None,
) -> dict:
    manifest = {"run_id": run_id}
    if pipeline:
        manifest["pipeline"] = pipeline
    return {
        "manifest": manifest,
        "metrics": metrics or {"input": {}, "compiled": {}},
        "env": env or {"os": "Linux", "python": "3.12", "packages": {}},
        "seeds": seeds or {"global_seed": 42},
    }


def test_no_regression():
    """Identical bundles → no regression detected."""
    b = _make_bundle()
    result = _analyze_regression_causes(b, b, {}, {})
    assert result["severity"] == "none"
    assert result["summary"] == "No regressions detected."


def test_tool_version_regression():
    """Package version change flagged as likely cause."""
    b1 = _make_bundle(env={"os": "Linux", "python": "3.12", "packages": {"qiskit": "1.0"}})
    b2 = _make_bundle(env={"os": "Linux", "python": "3.12", "packages": {"qiskit": "2.0"}})

    result = _analyze_regression_causes(b1, b2, {}, {})
    causes = result["causes"]
    assert any(c["type"] == "tool_version_change" for c in causes)


def test_seed_change_regression():
    """Different seeds flagged as potential cause."""
    b1 = _make_bundle(seeds={"global_seed": 42})
    b2 = _make_bundle(seeds={"global_seed": 99})

    result = _analyze_regression_causes(b1, b2, {}, {})
    causes = result["causes"]
    assert any(c["type"] == "seed_change" for c in causes)


def test_metric_regression_severity():
    """Large metric regressions increase severity."""
    metrics_diff = {
        "compiled": {
            "depth": {"a": 10, "b": 100, "pct_change": 900.0},
        }
    }
    result = _analyze_regression_causes(
        _make_bundle(), _make_bundle(), metrics_diff, {},
    )
    assert result["severity"] in ("high", "critical")
    assert len(result["regressions"]) == 1


def test_pipeline_change_detected():
    """Different pipeline specs are flagged."""
    b1 = _make_bundle(pipeline={"optimization_level": 1})
    b2 = _make_bundle(pipeline={"optimization_level": 3})

    result = _analyze_regression_causes(b1, b2, {}, {})
    causes = result["causes"]
    assert any(c["type"] == "pipeline_spec_change" for c in causes)


def test_compare_bundles_includes_regression():
    """compare_bundles returns regression_analysis section."""
    b1 = _make_bundle(
        run_id="a",
        metrics={"input": {"depth": 5}, "compiled": {"depth": 10}},
        env={"os": "Linux", "python": "3.12", "packages": {"qiskit": "1.0"}},
    )
    b2 = _make_bundle(
        run_id="b",
        metrics={"input": {"depth": 5}, "compiled": {"depth": 50}},
        env={"os": "Linux", "python": "3.12", "packages": {"qiskit": "2.0"}},
    )

    report = compare_bundles(b1, b2)
    assert "regression_analysis" in report
    assert report["regression_analysis"]["severity"] != "none"


def test_env_change_detected():
    """OS or Python change flagged."""
    b1 = _make_bundle(env={"os": "Linux", "python": "3.11", "packages": {}})
    b2 = _make_bundle(env={"os": "Windows", "python": "3.12", "packages": {}})

    env_diff = {"os": {"a": "Linux", "b": "Windows"}, "python": {"a": "3.11", "b": "3.12"}}
    result = _analyze_regression_causes(b1, b2, {}, env_diff)
    assert any(c["type"] == "environment_change" for c in result["causes"])


def test_markdown_includes_regression():
    """Comparison markdown includes the regression analysis section."""
    b1 = _make_bundle(
        run_id="a",
        metrics={"input": {"depth": 5}, "compiled": {"depth": 10}},
        env={"os": "Linux", "python": "3.12", "packages": {"qiskit": "1.0"}},
    )
    b2 = _make_bundle(
        run_id="b",
        metrics={"input": {"depth": 5}, "compiled": {"depth": 50}},
        env={"os": "Linux", "python": "3.12", "packages": {"qiskit": "2.0"}},
    )

    report = compare_bundles(b1, b2)
    md = report.get("markdown", "")
    assert "Regression Analysis" in md
