"""Bundle replay — reconstruct and re-run a compilation from a stored bundle.

Provides bit-exact reproducibility verification by extracting the pipeline
spec, seeds, and input circuit from a trace bundle, then re-running the
compilation and comparing the resulting hashes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ReplayResult:
    """Result of replaying a bundle.

    Attributes:
        original_run_id: The run ID from the original bundle.
        replay_bundle: Path to the replay output bundle.
        input_hash_match: Whether the input circuit hashes match.
        compiled_hash_match: Whether the compiled circuit hashes match.
        metrics_match: Whether compiled metrics are identical.
        original_hash: Original compiled circuit hash.
        replay_hash: Replayed compiled circuit hash.
        diff: Summary of differences (empty if bit-exact).
    """

    original_run_id: str
    replay_bundle: str | None = None
    input_hash_match: bool = False
    compiled_hash_match: bool = False
    metrics_match: bool = False
    original_hash: str = ""
    replay_hash: str = ""
    diff: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_run_id": self.original_run_id,
            "replay_bundle": self.replay_bundle,
            "input_hash_match": self.input_hash_match,
            "compiled_hash_match": self.compiled_hash_match,
            "metrics_match": self.metrics_match,
            "original_hash": self.original_hash,
            "replay_hash": self.replay_hash,
            "diff": self.diff,
        }

    @property
    def bit_exact(self) -> bool:
        return self.input_hash_match and self.compiled_hash_match and self.metrics_match


def replay_bundle(
    bundle_path: str,
    output: str | None = None,
) -> ReplayResult:
    """Replay a trace bundle — re-run the same compilation and verify.

    Parameters:
        bundle_path: Path to the original bundle (zip or directory).
        output: Path for the replay output bundle (optional).

    Returns:
        ReplayResult with match/mismatch details.
    """
    from qocc.core.artifacts import ArtifactStore
    from qocc.api import run_trace

    # Load original bundle
    bundle = ArtifactStore.load_bundle(bundle_path)
    manifest = bundle.get("manifest", {})
    seeds = bundle.get("seeds", {})
    metrics_orig = bundle.get("metrics", {})

    original_run_id = manifest.get("run_id", "unknown")
    adapter_name = manifest.get("adapter", "qiskit")
    pipeline_conf = manifest.get("pipeline", {})

    # Find input circuit
    root = bundle.get("_root")
    input_circuit: str | None = None
    if root:
        root_path = Path(root)
        for candidate in ["circuits/input.qasm", "circuits/selected.qasm"]:
            p = root_path / candidate
            if p.exists():
                input_circuit = str(p)
                break

    if not input_circuit:
        return ReplayResult(
            original_run_id=original_run_id,
            diff={"error": "No input circuit found in bundle."},
        )

    # Re-run the trace
    try:
        result = run_trace(
            adapter_name=adapter_name,
            input_source=input_circuit,
            pipeline=pipeline_conf if pipeline_conf else None,
            output=output,
            seeds=seeds,
        )
    except Exception as exc:
        return ReplayResult(
            original_run_id=original_run_id,
            diff={"error": f"Replay failed: {exc}"},
        )

    # Compare results
    orig_compiled_hash = ""
    replay_compiled_hash = result.get("compiled_hash", "")
    orig_input_hash = ""
    replay_input_hash = result.get("input_hash", "")

    # Try to recover original hashes from metrics or manifest
    input_metrics_orig = metrics_orig.get("input", {})
    compiled_metrics_orig = metrics_orig.get("compiled", {})

    replay_metrics = result.get("metrics_after", {})

    # Hash matching (requires circuits in original bundle)
    if root:
        root_path = Path(root)
        from qocc.adapters.base import get_adapter

        try:
            adapter = get_adapter(adapter_name)
            input_qasm_path = root_path / "circuits" / "input.qasm"
            selected_qasm_path = root_path / "circuits" / "selected.qasm"

            if input_qasm_path.exists():
                orig_input = adapter.ingest(str(input_qasm_path))
                orig_input_hash = orig_input.stable_hash()

            if selected_qasm_path.exists():
                orig_compiled = adapter.ingest(str(selected_qasm_path))
                orig_compiled_hash = orig_compiled.stable_hash()
        except Exception:
            pass

    input_hash_match = orig_input_hash == replay_input_hash if orig_input_hash else True
    compiled_hash_match = orig_compiled_hash == replay_compiled_hash if orig_compiled_hash else False

    # Metrics comparison
    metrics_diff: dict[str, Any] = {}
    for key in set(list(compiled_metrics_orig.keys()) + list(replay_metrics.keys())):
        if key == "gate_histogram":
            continue
        va = compiled_metrics_orig.get(key)
        vb = replay_metrics.get(key)
        if va != vb:
            metrics_diff[key] = {"original": va, "replay": vb}

    metrics_match = len(metrics_diff) == 0

    return ReplayResult(
        original_run_id=original_run_id,
        replay_bundle=result.get("bundle_zip"),
        input_hash_match=input_hash_match,
        compiled_hash_match=compiled_hash_match,
        metrics_match=metrics_match,
        original_hash=orig_compiled_hash,
        replay_hash=replay_compiled_hash,
        diff=metrics_diff if metrics_diff else {},
    )
