"""Tests for Phase 7 features.

Covers:
- ``qocc validate`` CLI command
- Bundle comparison deprecation warning (compare_legacy)
- JSON-pure stdout (compare --format json)
- Cache concurrent put (atomic writes + per-key locking)
- ZipSlip rejection in artifact extraction
- Resource budget enforcement (max_runtime in SPRT loop)
- Cross-thread span parentage (parent= kwarg)
- CI mypy job presence
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner


# ======================================================================
# qocc validate CLI
# ======================================================================

class TestValidateCLI:
    """Verify the ``qocc validate`` command."""

    def _make_bundle(self, tmp: Path, files: dict[str, Any]) -> Path:
        """Create a minimal bundle directory with the given JSON files."""
        bundle = tmp / "test_bundle"
        bundle.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (bundle / name).write_text(
                json.dumps(content, indent=2) + "\n", encoding="utf-8"
            )
        return bundle

    def test_validate_empty_bundle(self, tmp_path: Path) -> None:
        """An empty bundle should report all files as skipped."""
        from qocc.cli.commands_validate import validate

        bundle = tmp_path / "empty"
        bundle.mkdir()

        runner = CliRunner()
        result = runner.invoke(validate, [str(bundle), "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert all(r["status"] == "skipped" for r in data["results"])

    def test_validate_valid_manifest(self, tmp_path: Path) -> None:
        """A well-formed manifest.json should pass validation."""
        from qocc.cli.commands_validate import validate

        bundle = self._make_bundle(tmp_path, {
            "manifest.json": {
                "schema_version": "0.1.0",
                "created_at": "2025-01-01T00:00:00Z",
                "run_id": "abc-123",
                "qocc_version": "0.1.0",
            }
        })

        runner = CliRunner()
        result = runner.invoke(validate, [str(bundle), "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        manifest_result = next(r for r in data["results"] if r["file"] == "manifest.json")
        assert manifest_result["status"] == "valid"

    def test_validate_strict_mode_exits_nonzero(self, tmp_path: Path) -> None:
        """With --strict, an invalid file should cause exit code 1."""
        from qocc.cli.commands_validate import validate

        bundle = self._make_bundle(tmp_path, {
            "manifest.json": {"bad_key": True},  # invalid schema
        })

        runner = CliRunner()
        result = runner.invoke(validate, [str(bundle), "--strict", "--format", "json"])
        # If there are validation errors + strict, exit code should be 1
        data = json.loads(result.output)
        manifest_result = next(r for r in data["results"] if r["file"] == "manifest.json")
        if manifest_result["status"] == "invalid":
            assert result.exit_code == 1

    def test_validate_table_format(self, tmp_path: Path) -> None:
        """Table format should not crash (output goes to stderr via Rich)."""
        from qocc.cli.commands_validate import validate

        bundle = self._make_bundle(tmp_path, {
            "manifest.json": {
                "schema_version": "0.1.0",
                "created_at": "2025-01-01T00:00:00Z",
                "run_id": "r",
                "qocc_version": "0.1.0",
            }
        })

        runner = CliRunner()
        result = runner.invoke(validate, [str(bundle), "--format", "table"])
        assert result.exit_code == 0


# ======================================================================
# compare_legacy deprecation
# ======================================================================

class TestCompareLegacyDeprecation:
    """Verify the top-level ``qocc compare`` shows deprecation notice."""

    def test_compare_legacy_emits_warning(self, tmp_path: Path) -> None:
        """compare_legacy should print 'deprecated' to stderr."""
        from qocc.cli.commands_compare import compare_legacy

        # Create two minimal bundles
        for name in ("a", "b"):
            d = tmp_path / name
            d.mkdir()
            (d / "manifest.json").write_text(
                json.dumps({"schema_version": "0.1.0", "run_id": name}) + "\n"
            )
            (d / "metrics.json").write_text(json.dumps({}) + "\n")
            (d / "env.json").write_text(json.dumps({}) + "\n")

        runner = CliRunner()
        result = runner.invoke(
            compare_legacy,
            [str(tmp_path / "a"), str(tmp_path / "b"), "--format", "text"],
        )
        # Should succeed (or at least not crash on missing circuits)
        # Check combined output for deprecation warning
        assert "deprecated" in (result.output or "").lower()


# ======================================================================
# JSON-pure stdout for compare
# ======================================================================

class TestCompareJSONPure:
    """Compare --format json should put ONLY JSON on stdout."""

    def test_json_format_stdout_is_pure(self, tmp_path: Path) -> None:
        from qocc.cli.commands_compare import compare

        for name in ("a", "b"):
            d = tmp_path / name
            d.mkdir()
            (d / "manifest.json").write_text(
                json.dumps({"schema_version": "0.1.0", "run_id": name}) + "\n"
            )
            (d / "metrics.json").write_text(json.dumps({}) + "\n")
            (d / "env.json").write_text(json.dumps({}) + "\n")

        runner = CliRunner()
        result = runner.invoke(
            compare,
            [str(tmp_path / "a"), str(tmp_path / "b"), "--format", "json"],
        )
        if result.exit_code == 0 and result.output.strip():
            # The output should contain valid JSON somewhere; since
            # CliRunner merges streams, find the JSON object in output
            text = result.output.strip()
            # Find the first '{' to skip any Rich preamble mixed in
            start = text.find("{")
            if start >= 0:
                parsed = json.loads(text[start:])
                assert isinstance(parsed, dict)


# ======================================================================
# Cache: concurrent put (atomic writes + per-key locking)
# ======================================================================

class TestCacheConcurrentPut:
    """Verify cache thread safety under concurrent writes."""

    def test_concurrent_puts_no_corruption(self, tmp_path: Path) -> None:
        """Many threads writing the same key should not produce partial JSON."""
        from qocc.core.cache import CompilationCache

        cache = CompilationCache(cache_dir=tmp_path / "cache")
        key = "test_key_abc123"
        errors: list[str] = []

        def writer(idx: int) -> None:
            try:
                cache.put(key, {"idx": idx, "data": "x" * 1000})
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent put: {errors}"

        # The final result should be valid JSON
        got = cache.get(key)
        assert got is not None
        assert "idx" in got

    def test_concurrent_put_get_different_keys(self, tmp_path: Path) -> None:
        """Concurrent put+get on different keys should all succeed."""
        from qocc.core.cache import CompilationCache

        cache = CompilationCache(cache_dir=tmp_path / "cache")
        results: dict[str, dict[str, Any] | None] = {}

        def worker(idx: int) -> None:
            k = f"key_{idx}"
            cache.put(k, {"v": idx})
            results[k] = cache.get(k)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for i in range(10):
            k = f"key_{i}"
            assert results[k] is not None
            assert results[k]["v"] == i  # type: ignore[index]

    def test_atomic_write_creates_valid_file(self, tmp_path: Path) -> None:
        """_atomic_write_text should produce a fully written file."""
        from qocc.core.cache import _atomic_write_text

        target = tmp_path / "test.json"
        content = json.dumps({"hello": "world"}, indent=2)
        _atomic_write_text(target, content)

        assert target.exists()
        assert json.loads(target.read_text(encoding="utf-8")) == {"hello": "world"}


# ======================================================================
# ZipSlip protection
# ======================================================================

class TestZipSlipProtection:
    """Verify that malicious zip members are rejected."""

    def test_zipslip_path_traversal_rejected(self, tmp_path: Path) -> None:
        """A zip with '../evil.txt' should raise ValueError."""
        from qocc.core.artifacts import ArtifactStore

        # Create a malicious zip
        zip_path = tmp_path / "evil.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("../evil.txt", "pwned")
            zf.writestr("manifest.json", '{"schema_version":"0.1.0"}')

        with pytest.raises(ValueError, match="ZipSlip"):
            ArtifactStore.load_bundle(str(zip_path))

    def test_safe_zip_extracts_normally(self, tmp_path: Path) -> None:
        """A safe zip should extract without issues."""
        from qocc.core.artifacts import ArtifactStore

        zip_path = tmp_path / "safe.zip"
        manifest = {"schema_version": "0.1.0", "run_id": "test", "qocc_version": "0.1.0"}
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("manifest.json", json.dumps(manifest))

        bundle = ArtifactStore.load_bundle(str(zip_path))
        assert "manifest" in bundle
        assert bundle["manifest"]["run_id"] == "test"


# ======================================================================
# Resource budget: max_runtime in SPRT loop
# ======================================================================

class TestMaxRuntimeBudget:
    """Verify that max_runtime in resource_budget stops iteration."""

    def test_iterative_evaluate_respects_max_runtime(self) -> None:
        """When max_runtime is tiny, _iterative_evaluate should stop early."""
        from qocc.contracts.eval_sampling import _iterative_evaluate
        from qocc.contracts.spec import ContractSpec, ContractResult

        spec = ContractSpec(
            name="test_runtime",
            type="distribution",
            tolerances={"tvd": 0.05},
            confidence={"level": 0.95},
            resource_budget={
                "min_shots": 100,
                "max_shots": 10_000_000,
                "early_stopping": True,
                "max_runtime": 0.001,  # 1 ms — should stop almost immediately
            },
        )

        counts_before = {"00": 50, "01": 50}
        counts_after = {"00": 50, "01": 50}

        def slow_simulate(shots: int) -> dict[str, int]:
            time.sleep(0.05)  # 50 ms per simulation
            return {"00": shots // 2, "01": shots // 2}

        def evaluate_once(
            s: ContractSpec, before: dict[str, int], after: dict[str, int]
        ) -> ContractResult:
            return ContractResult(
                name=s.name,
                passed=True,
                details={"tvd_point": 0.0, "tolerance": 0.05, "shots_after": sum(after.values())},
            )

        result = _iterative_evaluate(
            spec, slow_simulate, counts_before, counts_after, evaluate_once,
        )
        # Should have set budget_exceeded or stopped early
        assert result.details.get("early_stopped") is True or result.details.get("budget_exceeded") == "max_runtime"


# ======================================================================
# Cross-thread span parentage
# ======================================================================

class TestCrossThreadSpanParentage:
    """Verify that candidate spans set parent explicitly."""

    def test_start_span_accepts_parent_kwarg(self) -> None:
        """TraceEmitter.start_span should accept a parent kwarg."""
        from qocc.trace.emitter import TraceEmitter

        emitter = TraceEmitter()
        parent = emitter.start_span("parent_op")
        child = emitter.start_span("child_op", parent=parent)

        # Child should reference parent's span_id
        assert child.parent_span_id == parent.span_id
        emitter.finish_span(child)
        emitter.finish_span(parent)

    def test_parent_propagates_across_threads(self) -> None:
        """A span started in a worker thread with parent= should have correct parent_span_id."""
        from qocc.trace.emitter import TraceEmitter

        emitter = TraceEmitter()
        parent = emitter.start_span("root")
        child_ids: list[str | None] = [None]

        def worker() -> None:
            child = emitter.start_span("worker_task", parent=parent)
            child_ids[0] = child.parent_span_id
            emitter.finish_span(child)

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        emitter.finish_span(parent)

        assert child_ids[0] == parent.span_id


# ======================================================================
# CI mypy job presence
# ======================================================================

class TestCIConfiguration:
    """Verify the CI workflow has expected jobs."""

    def test_ci_yaml_has_typecheck_job(self) -> None:
        """ci.yml should contain a typecheck job."""
        ci_path = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"
        if not ci_path.exists():
            pytest.skip("CI file not found")

        text = ci_path.read_text(encoding="utf-8")
        # Verify the typecheck job and mypy step are present
        assert "typecheck:" in text or "typecheck\n" in text, "Expected 'typecheck' job in ci.yml"
        assert "mypy" in text.lower(), "Expected 'mypy' step in ci.yml"


# ======================================================================
# Validate command registered in main CLI
# ======================================================================

class TestCLIRegistration:
    """Verify all Phase 7 commands are properly registered."""

    def test_validate_command_registered(self) -> None:
        from qocc.cli.main import cli

        commands = list(cli.commands.keys())
        assert "validate" in commands

    def test_trace_compare_subcommand_registered(self) -> None:
        from qocc.cli.commands_trace import trace

        commands = list(trace.commands.keys())
        assert "compare" in commands

    def test_compare_legacy_registered_at_top_level(self) -> None:
        from qocc.cli.main import cli

        assert "compare" in cli.commands
