# Architecture Deep-Dive: Bundle format

A QOCC Trace Bundle is the reproducibility envelope for one run.

## Key files

- `manifest.json` — run metadata and schema version
- `env.json` — environment snapshot
- `seeds.json` — RNG seeds and algorithms
- `metrics.json` — input/compiled metrics and pass logs
- `trace.jsonl` — emitted spans and events
- `contracts.json` / `contract_results.json` — expected constraints and outcomes
- `cache_index.json` — cache hit/miss provenance

Optional files include hardware payloads, QEC artifacts, search rankings, and batch summaries.

## Design goals

- deterministic and portable
- schema-validated
- directly usable in CI and regression systems
- support diffing and replay workflows

## Compatibility strategy

Schemas are additive-first where possible, and bundles preserve enough provenance to support cross-version analysis.
