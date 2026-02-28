# Architecture Deep-Dive: Trace model

QOCC records workflow execution as structured spans compatible with OpenTelemetry-style concepts.

## Core concepts

- **Trace**: one logical run of `trace run`, `search_compile`, or batch execution.
- **Span**: timed operation with name, attributes, status, events, and links.
- **Event**: point-in-time annotation under a span (for example cache hits or polling).

## Span lifecycle

1. Create span at stage start
2. Attach contextual attributes (adapter, candidate id, hashes)
3. Add events for notable transitions
4. Finish span with status (`OK`, `ERROR`, `UNSET`)

## Typical stage sequence

- ingest
- normalize
- compile
- mitigation (optional)
- compute_metrics
- write_bundle
- export_zip

## Cross-thread and linked spans

Search and batch flows can emit child spans from worker threads while preserving parent/child relationships and links for trace visualization and analysis.
