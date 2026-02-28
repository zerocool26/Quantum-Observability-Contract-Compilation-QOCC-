"""Machine-readable bundle diff format."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

@dataclass
class BundleDiff:
    """Structured representation of differences between two trace bundles."""
    
    metric_deltas: dict[str, Any] = field(default_factory=dict)
    contract_regressions: list[dict[str, Any]] = field(default_factory=list)
    pass_log_diffs: list[dict[str, Any]] = field(default_factory=list)
    env_diffs: dict[str, Any] = field(default_factory=dict)
    circuit_hash_change: bool = False
    regression_cause: Literal["TOOL_VERSION", "PASS_PARAM", "SEED", "ROUTING", "UNKNOWN"] = "UNKNOWN"

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "metric_deltas": self.metric_deltas,
            "contract_regressions": self.contract_regressions,
            "pass_log_diffs": self.pass_log_diffs,
            "env_diffs": self.env_diffs,
            "circuit_hash_change": self.circuit_hash_change,
            "regression_cause": self.regression_cause,
        }
