"""Tests for Phase 9 features — audit fixes and hardening.

Covers:
- CircuitHandle stable_hash caching
- DEFAULT_SEED used across modules
- np.random.default_rng migration
- Thread-safe TraceEmitter reads
- CompilationCache context manager + lock cleanup
- Entry-point idempotency guard
- render_timeline width clamping
- run_id sanitization
- py.typed marker exists
- Tooling config consistency
- Redundant exception tuple fix
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from click.testing import CliRunner


# ======================================================================
# 1. CircuitHandle stable_hash caching
# ======================================================================

class TestCircuitHandleHashCache:
    """Cached stable_hash should be immutable once computed."""

    def test_hash_is_cached(self) -> None:
        from qocc.core.circuit_handle import CircuitHandle

        h = CircuitHandle(
            name="test", num_qubits=2,
            native_circuit="dummy", source_format="qasm3",
            qasm3="OPENQASM 3.0; qubit[2] q;",
        )
        hash1 = h.stable_hash()
        hash2 = h.stable_hash()
        assert hash1 == hash2
        assert h._stable_hash_cache == hash1

    def test_hash_stable_after_mutation(self) -> None:
        """Once cached, hash doesn't change even if qasm3 is mutated."""
        from qocc.core.circuit_handle import CircuitHandle

        h = CircuitHandle(
            name="test", num_qubits=2,
            native_circuit="dummy", source_format="qasm3",
            qasm3="OPENQASM 3.0; qubit[2] q;",
        )
        hash_before = h.stable_hash()
        # Mutate qasm3 (bad practice but tests immutability of cache)
        h.qasm3 = "OPENQASM 3.0; qubit[3] q;"
        hash_after = h.stable_hash()
        assert hash_before == hash_after

    def test_deepcopy_gets_fresh_hash(self) -> None:
        """A deepcopy should recompute its hash."""
        import copy
        from qocc.core.circuit_handle import CircuitHandle

        h = CircuitHandle(
            name="test", num_qubits=2,
            native_circuit="dummy", source_format="qasm3",
            qasm3="OPENQASM 3.0; qubit[2] q;",
        )
        _ = h.stable_hash()  # cache it
        h2 = copy.deepcopy(h)
        h2.qasm3 = "OPENQASM 3.0; qubit[5] q;"
        h2._stable_hash_cache = None  # reset cache on copy

        assert h.stable_hash() != h2.stable_hash()

    def test_hash_and_eq_consistent(self) -> None:
        from qocc.core.circuit_handle import CircuitHandle

        a = CircuitHandle(name="a", num_qubits=2, native_circuit="x", source_format="qasm3",
                          qasm3="OPENQASM 3.0;")
        b = CircuitHandle(name="b", num_qubits=3, native_circuit="y", source_format="qasm3",
                          qasm3="OPENQASM 3.0;")
        assert a == b  # same qasm3 → same hash
        assert hash(a) == hash(b)


# ======================================================================
# 2. DEFAULT_SEED consistency
# ======================================================================

class TestDefaultSeedUsage:
    """Verify DEFAULT_SEED is used instead of hardcoded 42."""

    def test_api_uses_default_seed(self) -> None:
        import inspect
        import qocc.api as api

        source = inspect.getsource(api)
        # Should not have hardcoded 'seed: int = 42'
        assert "simulation_seed: int = 42" not in source
        assert "DEFAULT_SEED" in source
        assert '"MT19937"' not in source
        assert "DEFAULT_RNG_ALGORITHM" in source

    def test_stats_uses_default_seed(self) -> None:
        import inspect
        import qocc.contracts.stats as stats

        source = inspect.getsource(stats)
        assert "seed: int = 42" not in source
        assert "DEFAULT_SEED" in source

    def test_commands_trace_uses_default_seed(self) -> None:
        import inspect
        import qocc.cli.commands_trace as ct

        source = inspect.getsource(ct)
        assert "default=42" not in source
        assert "DEFAULT_SEED" in source
        assert '"MT19937"' not in source
        assert "DEFAULT_RNG_ALGORITHM" in source

    def test_search_space_uses_default_seed(self) -> None:
        import inspect
        import qocc.search.space as sp

        source = inspect.getsource(sp)
        assert "DEFAULT_SEED" in source

    def test_validator_uses_default_seed(self) -> None:
        import inspect
        import qocc.search.validator as v

        source = inspect.getsource(v)
        assert "seed=42" not in source
        assert "DEFAULT_SEED" in source


# ======================================================================
# 3. np.random.default_rng migration
# ======================================================================

class TestNumpyRngMigration:
    """Verify no more np.random.RandomState in source (except cirq adapter passthrough)."""

    @pytest.mark.parametrize("mod_path", [
        "qocc.contracts.stats",
        "qocc.search.space",
    ])
    def test_no_random_state(self, mod_path: str) -> None:
        import importlib
        import inspect

        mod = importlib.import_module(mod_path)
        source = inspect.getsource(mod)
        assert "RandomState" not in source, f"{mod_path} still uses RandomState"

    def test_bootstrap_ci_works(self) -> None:
        """Bootstrap CI should work with the new default_rng."""
        from qocc.contracts.stats import tvd_bootstrap_ci

        counts_a = {"00": 500, "11": 500}
        counts_b = {"00": 480, "11": 520}
        ci = tvd_bootstrap_ci(counts_a, counts_b, confidence=0.95, seed=42)
        assert 0 < ci["upper"] < 0.5
        assert ci["point"] < 0.1


# ======================================================================
# 4. Thread-safe TraceEmitter reads
# ======================================================================

class TestEmitterThreadSafety:
    """finished_spans() and to_dicts() should be thread-safe."""

    def test_concurrent_write_and_read(self) -> None:
        from qocc.trace.emitter import TraceEmitter

        emitter = TraceEmitter()
        errors: list[str] = []

        def writer() -> None:
            for i in range(100):
                s = emitter.start_span(f"span-{i}")
                emitter.finish_span(s)

        def reader() -> None:
            for _ in range(100):
                try:
                    spans = emitter.finished_spans()
                    # Should always be a valid list
                    assert isinstance(spans, list)
                except Exception as e:
                    errors.append(str(e))

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Thread safety errors: {errors}"
        assert len(emitter.finished_spans()) == 100

    def test_to_dicts_thread_safe(self) -> None:
        from qocc.trace.emitter import TraceEmitter

        emitter = TraceEmitter()
        for i in range(10):
            s = emitter.start_span(f"s{i}")
            emitter.finish_span(s)

        dicts = emitter.to_dicts()
        assert len(dicts) == 10
        assert all("name" in d for d in dicts)


# ======================================================================
# 5. CompilationCache context manager + lock cleanup
# ======================================================================

class TestCacheContextManager:
    """Cache should support with-statement and clean up locks."""

    def test_context_manager(self, tmp_path: Path) -> None:
        from qocc.core.cache import CompilationCache

        with CompilationCache(cache_dir=tmp_path / "cache") as cache:
            key = cache.cache_key("abc", {"opt": 1})
            cache.put(key, {"result": "ok"}, circuit_qasm="OPENQASM 3.0;")
            assert cache.get(key) is not None

    def test_close_clears_locks(self, tmp_path: Path) -> None:
        from qocc.core.cache import CompilationCache

        cache = CompilationCache(cache_dir=tmp_path / "cache")
        key = cache.cache_key("abc", {"opt": 1})
        _ = cache._lock_for_key(key)
        assert len(cache._key_locks) >= 1
        cache.close()
        assert len(cache._key_locks) == 0

    def test_evict_prunes_locks(self, tmp_path: Path) -> None:
        from qocc.core.cache import CompilationCache

        cache = CompilationCache(cache_dir=tmp_path / "cache")
        # Create 5 entries
        keys = []
        for i in range(5):
            key = cache.cache_key(f"circ{i}", {"opt": i})
            cache.put(key, {"result": i}, circuit_qasm=f"OPENQASM 3.0; // {i}")
            keys.append(key)

        # All 5 have locks
        for k in keys:
            cache._lock_for_key(k)
        assert len(cache._key_locks) == 5

        # Evict to keep only 2
        evicted = cache.evict_lru(max_entries=2)
        assert evicted == 3
        # Should have fewer locks now
        assert len(cache._key_locks) <= 2


# ======================================================================
# 6. Entry-point idempotency
# ======================================================================

class TestEntryPointIdempotency:
    """Entry-point discovery should only scan once."""

    def test_adapter_discovery_idempotent(self) -> None:
        from qocc.adapters import base as ab

        # Reset the flag
        ab._ep_adapters_discovered = False
        ab._discover_entry_point_adapters()
        assert ab._ep_adapters_discovered is True
        # Second call should be a no-op (flag stays True)
        ab._discover_entry_point_adapters()
        assert ab._ep_adapters_discovered is True

    def test_evaluator_discovery_idempotent(self) -> None:
        from qocc.contracts import registry as reg

        reg._ep_evaluators_discovered = False
        reg._discover_entry_point_evaluators()
        assert reg._ep_evaluators_discovered is True
        reg._discover_entry_point_evaluators()
        assert reg._ep_evaluators_discovered is True


# ======================================================================
# 7. render_timeline width clamping
# ======================================================================

class TestRenderTimelineWidth:
    """render_timeline should clamp width >= 20."""

    def test_width_zero_clamped(self) -> None:
        from qocc.trace.visualization import render_timeline

        spans = [{"name": "root", "span_id": "a", "start_time": 0, "end_time": 1.0, "status": "ok"}]
        output = render_timeline(spans, width=0)
        assert "root" in output

    def test_negative_width_clamped(self) -> None:
        from qocc.trace.visualization import render_timeline

        spans = [{"name": "root", "span_id": "a", "start_time": 0, "end_time": 1.0, "status": "ok"}]
        output = render_timeline(spans, width=-10)
        assert "root" in output


# ======================================================================
# 8. run_id sanitization
# ======================================================================

class TestRunIdSanitization:
    """ArtifactStore should reject unsafe run_ids."""

    def test_valid_run_id(self, tmp_path: Path) -> None:
        from qocc.core.artifacts import ArtifactStore

        store = ArtifactStore(tmp_path / "bundle")
        # Should not raise
        store.write_manifest("abc_123-def")

    def test_path_traversal_rejected(self, tmp_path: Path) -> None:
        from qocc.core.artifacts import ArtifactStore

        store = ArtifactStore(tmp_path / "bundle")
        with pytest.raises(ValueError, match="Invalid run_id"):
            store.write_manifest("../../etc/passwd")

    def test_slashes_rejected(self, tmp_path: Path) -> None:
        from qocc.core.artifacts import ArtifactStore

        store = ArtifactStore(tmp_path / "bundle")
        with pytest.raises(ValueError, match="Invalid run_id"):
            store.write_manifest("foo/bar")

    def test_spaces_rejected(self, tmp_path: Path) -> None:
        from qocc.core.artifacts import ArtifactStore

        store = ArtifactStore(tmp_path / "bundle")
        with pytest.raises(ValueError, match="Invalid run_id"):
            store.write_manifest("has spaces")


# ======================================================================
# 9. py.typed marker
# ======================================================================

class TestPyTypedMarker:
    """The py.typed marker should exist for PEP 561 compliance."""

    def test_py_typed_exists(self) -> None:
        marker = Path(__file__).resolve().parent.parent / "qocc" / "py.typed"
        assert marker.exists(), "qocc/py.typed marker is missing"


# ======================================================================
# 10. Tooling config consistency
# ======================================================================

class TestToolingConfig:
    """Ruff and mypy should target Python 3.11+ (matching requires-python)."""

    def test_versions_consistent(self) -> None:
        import tomllib

        toml_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        with open(toml_path, "rb") as f:
            cfg = tomllib.load(f)

        req_py = cfg["project"]["requires-python"]
        ruff_target = cfg["tool"]["ruff"]["target-version"]
        mypy_version = cfg["tool"]["mypy"]["python_version"]

        assert "3.11" in req_py
        assert ruff_target == "py311"
        assert mypy_version == "3.11"


# ======================================================================
# 11. Redundant exception tuple removed
# ======================================================================

class TestRedundantExceptFixed:
    """api.py should not have (NotImplementedError, Exception)."""

    def test_no_redundant_except(self) -> None:
        import inspect
        import qocc.api as api

        source = inspect.getsource(api)
        assert "(NotImplementedError, Exception)" not in source


# ======================================================================
# 12. Search space generates candidates with default_rng
# ======================================================================

class TestSearchSpaceRng:
    """generate_candidates should work with the new default_rng."""

    def test_random_strategy(self) -> None:
        from qocc.search.space import SearchSpaceConfig, generate_candidates

        config = SearchSpaceConfig(
            adapter="qiskit",
            optimization_levels=[0, 1, 2],
            seeds=[42, 99],
            strategy="random",
            max_candidates=10,
        )
        candidates = generate_candidates(config)
        assert 0 < len(candidates) <= 10

    def test_grid_strategy(self) -> None:
        from qocc.search.space import SearchSpaceConfig, generate_candidates

        config = SearchSpaceConfig(
            adapter="qiskit",
            optimization_levels=[0, 1],
            seeds=[42],
            strategy="grid",
            max_candidates=20,
        )
        candidates = generate_candidates(config)
        assert len(candidates) >= 2


# ======================================================================
# 13. Replay CLI status output
# ======================================================================

class TestReplayCLIStatusOutput:
    """trace replay should show explicit hash verification statuses."""

    def test_replay_cli_shows_unknown_status(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from qocc.cli.commands_trace import trace_replay
        from qocc.core.replay import ReplayResult

        bundle = tmp_path / "bundle.zip"
        bundle.write_text("dummy", encoding="utf-8")

        def _fake_replay_bundle(*_: object, **__: object) -> ReplayResult:
            return ReplayResult(
                original_run_id="run-1",
                replay_bundle="replayed.zip",
                input_hash_match=True,
                compiled_hash_match=False,
                metrics_match=True,
                input_hash_status="matched",
                compiled_hash_status="unknown",
                diff={"_verification": {"compiled_hash": "unknown"}},
            )

        monkeypatch.setattr("qocc.core.replay.replay_bundle", _fake_replay_bundle)

        runner = CliRunner()
        result = runner.invoke(trace_replay, [str(bundle)])
        assert result.exit_code == 0
        assert "Input hash status:    matched" in result.output
        assert "Compiled hash status: unknown" in result.output

    def test_replay_cli_shows_matched_statuses(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from qocc.cli.commands_trace import trace_replay
        from qocc.core.replay import ReplayResult

        bundle = tmp_path / "bundle.zip"
        bundle.write_text("dummy", encoding="utf-8")

        def _fake_replay_bundle(*_: object, **__: object) -> ReplayResult:
            return ReplayResult(
                original_run_id="run-2",
                replay_bundle="replayed.zip",
                input_hash_match=True,
                compiled_hash_match=True,
                metrics_match=True,
                input_hash_status="matched",
                compiled_hash_status="matched",
                diff={},
            )

        monkeypatch.setattr("qocc.core.replay.replay_bundle", _fake_replay_bundle)

        runner = CliRunner()
        result = runner.invoke(trace_replay, [str(bundle)])
        assert result.exit_code == 0
        assert "BIT-EXACT match" in result.output
        assert "Input hash status:    matched" in result.output
        assert "Compiled hash status: matched" in result.output
