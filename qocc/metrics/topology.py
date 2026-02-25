"""Topology analysis utilities."""

from __future__ import annotations

from typing import Any


def check_topology(
    circuit_handle: Any,
    coupling_map: list[tuple[int, int]],
) -> dict[str, Any]:
    """Check a circuit against a hardware coupling map.

    Returns:
        Dictionary with ``violations`` count and ``details``.
    """
    from qocc.metrics.compute import compute_metrics

    metrics = compute_metrics(circuit_handle, coupling_map=coupling_map)
    return {
        "coupling_map_size": len(coupling_map),
        "violations": metrics.get("topology_violations"),
    }


# Common coupling maps for testing/examples
LINEAR_5: list[tuple[int, int]] = [(i, i + 1) for i in range(4)]
GRID_2x3: list[tuple[int, int]] = [
    (0, 1), (1, 2),
    (3, 4), (4, 5),
    (0, 3), (1, 4), (2, 5),
]
HEAVY_HEX_7: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3),
    (3, 4), (4, 5), (5, 6),
    (0, 5), (1, 4), (2, 3),
]
