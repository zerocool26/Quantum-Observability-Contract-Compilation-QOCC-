"""Tests for the search_compile API and search components."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from qocc.search.space import SearchSpaceConfig, Candidate, generate_candidates
from qocc.search.scorer import surrogate_score, rank_candidates
from qocc.search.selector import select_best, SelectionResult


def test_generate_candidates_default():
    """Default config generates 4 opt_levels × 1 seed × 2 routing = 8 candidates."""
    config = SearchSpaceConfig()
    candidates = generate_candidates(config)
    assert len(candidates) == 8


def test_generate_candidates_custom():
    """Custom config generates correct Cartesian product."""
    config = SearchSpaceConfig(
        adapter="qiskit",
        optimization_levels=[1, 2],
        seeds=[42, 99],
        routing_methods=["sabre"],
    )
    candidates = generate_candidates(config)
    assert len(candidates) == 4  # 2 levels × 2 seeds × 1 routing


def test_candidate_ids_unique():
    """Each candidate has a unique ID."""
    config = SearchSpaceConfig()
    candidates = generate_candidates(config)
    ids = [c.candidate_id for c in candidates]
    assert len(ids) == len(set(ids))


def test_surrogate_score_basic():
    """Score is computed from weighted metrics."""
    metrics = {"depth": 10, "gates_2q": 5, "duration": 1000, "proxy_error": 0.1}
    result = surrogate_score(metrics)
    assert result["score"] > 0
    assert "breakdown" in result


def test_surrogate_score_custom_weights():
    """Custom weights change the score."""
    metrics = {"depth": 10, "gates_2q": 5}
    w1 = {"depth": 1.0, "gates_2q": 0.0}
    w2 = {"depth": 0.0, "gates_2q": 1.0}

    r1 = surrogate_score(metrics, weights=w1)
    r2 = surrogate_score(metrics, weights=w2)

    assert r1["score"] == 10.0
    assert r2["score"] == 5.0


def test_rank_candidates():
    """Candidates are sorted by score ascending."""
    from qocc.core.circuit_handle import PipelineSpec

    c1 = Candidate(
        pipeline=PipelineSpec(adapter="test", optimization_level=1),
        metrics={"depth": 100},
    )
    c2 = Candidate(
        pipeline=PipelineSpec(adapter="test", optimization_level=2),
        metrics={"depth": 10},
    )

    ranked = rank_candidates([c1, c2], weights={"depth": 1.0})
    assert ranked[0].surrogate_score < ranked[1].surrogate_score


def test_select_best_with_passing():
    """Selection picks the candidate with lowest score among those passing contracts."""
    from qocc.core.circuit_handle import PipelineSpec

    c1 = Candidate(
        pipeline=PipelineSpec(adapter="test", optimization_level=1),
        candidate_id="c1",
        surrogate_score=50.0,
        validated=True,
        contract_results=[{"name": "test", "passed": True}],
    )
    c2 = Candidate(
        pipeline=PipelineSpec(adapter="test", optimization_level=2),
        candidate_id="c2",
        surrogate_score=10.0,
        validated=True,
        contract_results=[{"name": "test", "passed": True}],
    )

    result = select_best([c1, c2])
    assert result.selected is not None
    assert result.selected.candidate_id == "c2"
    assert result.feasible is True


def test_select_best_infeasible():
    """Infeasible selection when all contract results fail."""
    from qocc.core.circuit_handle import PipelineSpec

    c1 = Candidate(
        pipeline=PipelineSpec(adapter="test"),
        candidate_id="c1",
        surrogate_score=10.0,
        validated=True,
        contract_results=[{"name": "test", "passed": False}],
    )

    result = select_best([c1])
    assert result.feasible is False
    assert result.selected is not None  # best-effort


def test_select_best_no_candidates():
    """No candidates → infeasible."""
    result = select_best([])
    assert result.feasible is False
    assert result.selected is None


def test_selection_result_serialization():
    """SelectionResult serializes to dict."""
    result = SelectionResult(feasible=True, reason="test")
    d = result.to_dict()
    assert d["feasible"] is True
    assert d["reason"] == "test"


def test_search_space_config_roundtrip():
    """Config serializes and deserializes correctly."""
    config = SearchSpaceConfig(
        adapter="cirq",
        optimization_levels=[0, 1, 2],
        seeds=[42, 100],
        routing_methods=["greedy"],
    )
    d = config.to_dict()
    config2 = SearchSpaceConfig.from_dict(d)
    assert config2.adapter == "cirq"
    assert config2.optimization_levels == [0, 1, 2]
    assert config2.seeds == [42, 100]
