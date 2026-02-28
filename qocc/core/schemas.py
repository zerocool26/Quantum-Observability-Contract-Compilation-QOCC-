"""JSON Schema definitions for all Trace Bundle files.

Each schema is a Python dict following JSON Schema Draft 2020-12.
A convenience function ``validate_bundle`` checks a bundle directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

# ======================================================================
# Schemas
# ======================================================================

MANIFEST_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "QOCC Trace Bundle Manifest",
    "type": "object",
    "required": ["schema_version", "created_at", "run_id"],
    "properties": {
        "schema_version": {"type": "string"},
        "created_at": {"type": "string", "format": "date-time"},
        "run_id": {"type": "string"},
        "qocc_version": {"type": "string"},
    },
    "additionalProperties": True,
}

ENV_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "QOCC Environment Snapshot",
    "type": "object",
    "required": ["os", "python"],
    "properties": {
        "os": {"type": "string"},
        "python": {"type": "string"},
        "python_executable": {"type": "string"},
        "packages": {"type": "object", "additionalProperties": {"type": "string"}},
        "git_sha": {"type": ["string", "null"]},
    },
    "additionalProperties": True,
}

SEEDS_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "QOCC Seeds",
    "type": "object",
    "properties": {
        "global_seed": {"type": ["integer", "null"]},
        "rng_algorithm": {"type": "string"},
        "stage_seeds": {
            "type": "object",
            "additionalProperties": {"type": "integer"},
        },
    },
    "additionalProperties": True,
}

METRICS_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "QOCC Metrics",
    "type": "object",
    "properties": {
        "width": {"type": "integer"},
        "total_gates": {"type": "integer"},
        "gates_1q": {"type": "integer"},
        "gates_2q": {"type": "integer"},
        "depth": {"type": "integer"},
        "depth_2q": {"type": ["integer", "null"]},
        "gate_histogram": {"type": "object", "additionalProperties": {"type": "integer"}},
        "topology_violations": {"type": ["integer", "null"]},
        "duration_estimate": {"type": ["number", "null"]},
        "proxy_error_score": {"type": ["number", "null"]},
        "mitigation": {"type": ["object", "null"]},
        "mitigation_shot_multiplier": {"type": ["number", "null"]},
        "mitigation_runtime_multiplier": {"type": ["number", "null"]},
        "mitigation_overhead_factor": {"type": ["number", "null"]},
    },
    "additionalProperties": True,
}

CONTRACTS_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "QOCC Contract Specifications",
    "type": "array",
    "items": {
        "type": "object",
        "required": ["name", "type"],
        "properties": {
            "name": {"type": "string"},
            "type": {
                "type": "string",
                "enum": ["observable", "distribution", "clifford", "exact", "cost", "qec"],
            },
            "spec": {"type": "object"},
            "tolerances": {"type": "object"},
            "confidence": {"type": "object"},
            "resource_budget": {"type": "object"},
        },
        "additionalProperties": True,
    },
}

CONTRACT_RESULTS_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "QOCC Contract Results",
    "type": "array",
    "items": {
        "type": "object",
        "required": ["name", "passed"],
        "properties": {
            "name": {"type": "string"},
            "passed": {"type": "boolean"},
            "details": {"type": "object"},
        },
        "additionalProperties": True,
    },
}

TRACE_SPAN_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "QOCC Trace Span (single JSON-Lines entry)",
    "type": "object",
    "required": ["trace_id", "span_id", "name", "start_time"],
    "properties": {
        "trace_id": {"type": "string"},
        "span_id": {"type": "string"},
        "parent_span_id": {"type": ["string", "null"]},
        "name": {"type": "string"},
        "start_time": {"type": "string"},
        "end_time": {"type": ["string", "null"]},
        "attributes": {"type": "object"},
        "events": {
            "type": "array",
            "items": {"type": "object"},
        },
        "links": {
            "type": "array",
            "items": {"type": "object"},
        },
        "status": {"type": "string", "enum": ["OK", "ERROR", "UNSET"]},
    },
    "additionalProperties": True,
}

CACHE_INDEX_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "QOCC Cache Index",
    "type": "array",
    "items": {
        "type": "object",
        "required": ["key", "hit"],
        "properties": {
            "key": {"type": "string"},
            "hit": {"type": "boolean"},
            "circuit_hash": {"type": "string"},
            "pipeline_hash": {"type": "string"},
            "candidate_id": {"type": "string"},
            "timestamp": {"type": "number"},
        },
        "additionalProperties": True,
    },
}

NONDETERMINISM_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "QOCC Nondeterminism Report",
    "type": "object",
    "required": ["reproducible", "num_runs", "unique_hashes"],
    "properties": {
        "reproducible": {"type": "boolean"},
        "num_runs": {"type": "integer"},
        "unique_hashes": {"type": "integer"},
        "confidence": {"type": "number"},
        "hashes": {
            "type": "array",
            "items": {"type": "string"},
        },
        "hash_counts": {
            "type": "object",
            "additionalProperties": {"type": "integer"},
        },
    },
    "additionalProperties": True,
}

SEARCH_RANKINGS_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "QOCC Search Rankings",
    "type": "array",
    "items": {
        "type": "object",
        "required": ["candidate_id", "surrogate_score"],
        "properties": {
            "candidate_id": {"type": "string"},
            "surrogate_score": {"type": "number"},
            "pipeline": {"type": "object"},
            "metrics": {"type": "object"},
            "validated": {"type": "boolean"},
            "contract_results": {"type": "array"},
        },
        "additionalProperties": True,
    },
}

SEARCH_RESULT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "QOCC Search Result",
    "type": "object",
    "required": ["feasible", "reason"],
    "properties": {
        "feasible": {"type": "boolean"},
        "reason": {"type": "string"},
        "selected": {
            "type": ["object", "null"],
            "properties": {
                "candidate_id": {"type": "string"},
                "surrogate_score": {"type": "number"},
            },
        },
        "pareto_frontier": {
            "type": ["array", "null"],
            "items": {"type": "object"},
        },
    },
    "additionalProperties": True,
}

DEM_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "QOCC Detector Error Model",
    "type": "object",
    "required": ["dem"],
    "properties": {
        "dem": {"type": "string"},
    },
    "additionalProperties": True,
}

LOGICAL_ERROR_RATES_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "QOCC Logical Error Rates",
    "type": "object",
    "properties": {
        "logical_error_rate": {"type": "number"},
        "shots": {"type": "integer"},
        "logical_errors": {"type": "integer"},
    },
    "additionalProperties": True,
}

DECODER_STATS_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "QOCC Decoder Stats",
    "type": "object",
    "properties": {
        "decoder_rounds": {"type": "integer"},
        "matching_graph_edges": {"type": ["integer", "null"]},
        "matching_graph_nodes": {"type": ["integer", "null"]},
        "logical_errors": {"type": "integer"},
        "code_distance": {"type": ["integer", "number"]},
        "mean_syndrome_weight": {"type": ["number", "null"]},
        "syndrome_weight_distribution": {
            "type": "object",
            "additionalProperties": {"type": ["integer", "number"]},
        },
    },
    "additionalProperties": True,
}

SIGNATURE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "QOCC Bundle Signature",
    "type": "object",
    "required": ["algorithm", "signer", "timestamp", "manifest_hash", "signature"],
    "properties": {
        "algorithm": {"type": "string", "enum": ["ed25519"]},
        "signer": {"type": "string"},
        "timestamp": {"type": "string", "format": "date-time"},
        "manifest_hash": {"type": "string"},
        "signature": {"type": "string"},
        "public_key_fingerprint": {"type": ["string", "null"]},
    },
    "additionalProperties": True,
}

NOISE_MODEL_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "QOCC Noise Model",
    "type": "object",
    "properties": {
        "single_qubit_error": {
            "oneOf": [
                {"type": "number", "minimum": 0.0, "maximum": 1.0},
                {
                    "type": "object",
                    "additionalProperties": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
            ]
        },
        "two_qubit_error": {
            "oneOf": [
                {"type": "number", "minimum": 0.0, "maximum": 1.0},
                {
                    "type": "object",
                    "additionalProperties": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
            ]
        },
        "readout_error": {
            "oneOf": [
                {"type": "number", "minimum": 0.0, "maximum": 1.0},
                {
                    "type": "object",
                    "additionalProperties": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
            ]
        },
        "t1": {
            "oneOf": [
                {"type": "number", "exclusiveMinimum": 0.0},
                {
                    "type": "object",
                    "additionalProperties": {"type": "number", "exclusiveMinimum": 0.0},
                },
                {"type": "null"},
            ]
        },
        "t2": {
            "oneOf": [
                {"type": "number", "exclusiveMinimum": 0.0},
                {
                    "type": "object",
                    "additionalProperties": {"type": "number", "exclusiveMinimum": 0.0},
                },
                {"type": "null"},
            ]
        },
    },
    "required": ["single_qubit_error", "two_qubit_error", "readout_error"],
    "additionalProperties": True,
}

# Registry for programmatic access
SCHEMAS: dict[str, dict[str, Any]] = {
    "manifest": MANIFEST_SCHEMA,
    "env": ENV_SCHEMA,
    "seeds": SEEDS_SCHEMA,
    "metrics": METRICS_SCHEMA,
    "contracts": CONTRACTS_SCHEMA,
    "contract_results": CONTRACT_RESULTS_SCHEMA,
    "trace_span": TRACE_SPAN_SCHEMA,
    "cache_index": CACHE_INDEX_SCHEMA,
    "nondeterminism": NONDETERMINISM_SCHEMA,
    "search_rankings": SEARCH_RANKINGS_SCHEMA,
    "search_result": SEARCH_RESULT_SCHEMA,
    "dem": DEM_SCHEMA,
    "logical_error_rates": LOGICAL_ERROR_RATES_SCHEMA,
    "decoder_stats": DECODER_STATS_SCHEMA,
    "noise_model": NOISE_MODEL_SCHEMA,
    "signature": SIGNATURE_SCHEMA,
}


# ======================================================================
# Validation
# ======================================================================


def validate_file(schema_name: str, data: Any) -> list[str]:
    """Validate *data* against the named schema. Returns list of error messages."""
    schema = SCHEMAS.get(schema_name)
    if schema is None:
        return [f"Unknown schema: {schema_name}"]
    errors: list[str] = []
    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.ValidationError as exc:
        errors.append(str(exc.message))
    return errors


def validate_bundle(bundle_dir: str | Path) -> dict[str, list[str]]:
    """Check all standard files inside *bundle_dir*. Returns ``{filename: errors}``."""
    root = Path(bundle_dir)
    results: dict[str, list[str]] = {}

    simple_maps = {
        "manifest.json": "manifest",
        "env.json": "env",
        "seeds.json": "seeds",
        "metrics.json": "metrics",
        "contracts.json": "contracts",
        "contract_results.json": "contract_results",
    }

    for fname, schema_name in simple_maps.items():
        fp = root / fname
        if fp.exists():
            data = json.loads(fp.read_text(encoding="utf-8"))
            results[fname] = validate_file(schema_name, data)
        else:
            results[fname] = [f"{fname} missing"]

    # trace.jsonl — validate each line
    trace_fp = root / "trace.jsonl"
    if trace_fp.exists():
        trace_errors: list[str] = []
        for i, line in enumerate(trace_fp.read_text(encoding="utf-8").strip().splitlines()):
            try:
                span = json.loads(line)
                errs = validate_file("trace_span", span)
                for e in errs:
                    trace_errors.append(f"line {i}: {e}")
            except json.JSONDecodeError as exc:
                trace_errors.append(f"line {i}: invalid JSON — {exc}")
        results["trace.jsonl"] = trace_errors
    else:
        results["trace.jsonl"] = ["trace.jsonl missing"]

    # Optional bundle files — validate if present
    optional_maps = {
        "cache_index.json": "cache_index",
        "nondeterminism.json": "nondeterminism",
        "search_rankings.json": "search_rankings",
        "search_result.json": "search_result",
        "dem.json": "dem",
        "logical_error_rates.json": "logical_error_rates",
        "decoder_stats.json": "decoder_stats",
        "signature.json": "signature",
    }
    for fname, schema_name in optional_maps.items():
        fp = root / fname
        if fp.exists():
            data = json.loads(fp.read_text(encoding="utf-8"))
            results[fname] = validate_file(schema_name, data)

    return results


def export_schemas(out_dir: str | Path) -> None:
    """Write all JSON schemas as standalone files."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, schema in SCHEMAS.items():
        (out / f"{name}.schema.json").write_text(
            json.dumps(schema, indent=2) + "\n", encoding="utf-8"
        )
