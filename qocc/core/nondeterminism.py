"""Nondeterminism detection for quantum compilation.

Hard requirement from spec §0:
  "Every stochastic transpiler pass must record its RNG seeds so
   the same input + same seeds → identical output."

This module detects when a compilation is NOT reproducible by running
it N times and comparing output hashes. It also provides utilities
for computing reproducibility confidence.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from qocc.core.circuit_handle import CircuitHandle, PipelineSpec
from qocc.core.hashing import hash_dict


@dataclass
class NondeterminismReport:
    """Report from a nondeterminism detection run.

    Attributes:
        reproducible: ``True`` if all runs produced identical output.
        num_runs: Number of compilation runs.
        unique_hashes: Number of distinct output hashes.
        hash_counts: Mapping ``{hash: count}``.
        confidence: Probability that the compilation is deterministic
            given the observed runs (Bayesian estimate).
        details: Extra data (per-run timings, hash list, etc.).
    """

    reproducible: bool
    num_runs: int
    unique_hashes: int
    hash_counts: dict[str, int] = field(default_factory=dict)
    confidence: float = 1.0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reproducible": self.reproducible,
            "num_runs": self.num_runs,
            "unique_hashes": self.unique_hashes,
            "hash_counts": self.hash_counts,
            "confidence": self.confidence,
            "details": self.details,
        }


def detect_nondeterminism(
    adapter: Any,
    circuit: CircuitHandle,
    pipeline: PipelineSpec,
    num_runs: int = 5,
) -> NondeterminismReport:
    """Run compilation multiple times and check for consistent output.

    Parameters:
        adapter: Adapter instance.
        circuit: Input circuit handle (normalized).
        pipeline: Pipeline spec to compile with.
        num_runs: How many times to repeat compilation.

    Returns:
        NondeterminismReport with reproducibility verdict.
    """
    import time

    hashes: list[str] = []
    timings: list[float] = []
    metrics_list: list[dict[str, Any]] = []

    for i in range(num_runs):
        t0 = time.perf_counter()
        result = adapter.compile(circuit, pipeline)
        dt = (time.perf_counter() - t0) * 1000.0  # ms

        compiled = result.circuit
        h = compiled.stable_hash()
        hashes.append(h)
        timings.append(dt)

        m = adapter.get_metrics(compiled)
        metrics_list.append(m.to_dict())

    # Count distinct hashes
    hash_counts: dict[str, int] = {}
    for h in hashes:
        hash_counts[h] = hash_counts.get(h, 0) + 1

    unique = len(hash_counts)
    reproducible = unique == 1

    # Compute reproducibility confidence
    # If we see K identical outputs in N runs, the Bayesian posterior
    # estimate (Beta(N+1,1)) gives P(deterministic) ≈ 1 - 1/(N+1)
    # when all are identical, and decreasing with more variation.
    if reproducible:
        confidence = 1.0 - 1.0 / (num_runs + 1)
    else:
        # Fraction of runs matching the most common hash
        max_count = max(hash_counts.values())
        confidence = max_count / num_runs * (1.0 - 1.0 / (num_runs + 1))

    # Detect metric drift even when hashes match
    if len(metrics_list) > 1:
        depth_values = [m.get("depth", 0) for m in metrics_list]
        gate_values = [m.get("total_gates", 0) for m in metrics_list]
        depth_std = statistics.stdev(depth_values) if len(depth_values) > 1 else 0.0
        gate_std = statistics.stdev(gate_values) if len(gate_values) > 1 else 0.0
    else:
        depth_std = 0.0
        gate_std = 0.0

    details: dict[str, Any] = {
        "hashes": hashes,
        "timings_ms": timings,
        "metrics_per_run": metrics_list,
        "timing_mean_ms": statistics.mean(timings) if timings else 0,
        "timing_std_ms": statistics.stdev(timings) if len(timings) > 1 else 0,
        "depth_std": depth_std,
        "gate_count_std": gate_std,
    }

    return NondeterminismReport(
        reproducible=reproducible,
        num_runs=num_runs,
        unique_hashes=unique,
        hash_counts=hash_counts,
        confidence=confidence,
        details=details,
    )


def compare_run_hashes(
    hashes_a: list[str],
    hashes_b: list[str],
) -> dict[str, Any]:
    """Cross-compare two sets of run hashes for reproducibility analysis.

    Useful for checking if two different environments produce the same
    compilation results.

    Returns:
        Dict with overlap statistics.
    """
    set_a = set(hashes_a)
    set_b = set(hashes_b)
    common = set_a & set_b
    only_a = set_a - set_b
    only_b = set_b - set_a

    return {
        "common": list(common),
        "only_a": list(only_a),
        "only_b": list(only_b),
        "overlap_fraction": len(common) / max(len(set_a | set_b), 1),
        "identical": set_a == set_b and len(set_a) == 1,
    }
