"""Topology analysis utilities."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from dataclasses import dataclass, field
import json

@dataclass
class TopologyNode:
    qubit_id: int
    t1: float | None = None
    t2: float | None = None
    readout_fidelity: float | None = None

@dataclass
class TopologyGraph:
    nodes: Dict[int, TopologyNode] = field(default_factory=dict)
    edges: List[Tuple[int, int]] = field(default_factory=list)
    
    @classmethod
    def from_ibm_backend(cls, backend: Any) -> TopologyGraph:
        """Load live topology from IBM Quantum backend."""
        config = backend.configuration()
        props = backend.properties()
        
        nodes = {}
        for q in range(config.n_qubits):
            t1 = props.t1(q) * 1e6 if props else None  # approx us
            t2 = props.t2(q) * 1e6 if props else None
            readout_error = props.readout_error(q) if props else 0.0
            nodes[q] = TopologyNode(q, t1=t1, t2=t2, readout_fidelity=1.0 - readout_error)
            
        edges = [tuple(edge) for edge in getattr(config, "coupling_map", []) or []]
        return cls(nodes=nodes, edges=edges)

    @classmethod
    def from_json(cls, path: str) -> TopologyGraph:
        """Load from JSON file (schema: topology.schema.json)."""
        with open(path, "r") as f:
            data = json.load(f)
            
        nodes = {
            node["qubit"]: TopologyNode(
                node["qubit"],
                t1=node.get("t1"),
                t2=node.get("t2"),
                readout_fidelity=node.get("readout_fidelity")
            )
            for node in data.get("nodes", [])
        }
        edges = [tuple(e) for e in data.get("edges", [])]
        return cls(nodes=nodes, edges=edges)

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
