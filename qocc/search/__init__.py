"""Search subpackage — closed-loop compilation search (v3)."""

from __future__ import annotations

__all__ = [
    "SearchSpaceConfig",
    "generate_candidates",
    "surrogate_score",
    "rank_candidates",
    "select_best",
]

from qocc.search.space import SearchSpaceConfig, generate_candidates
from qocc.search.scorer import surrogate_score, rank_candidates
from qocc.search.selector import select_best
