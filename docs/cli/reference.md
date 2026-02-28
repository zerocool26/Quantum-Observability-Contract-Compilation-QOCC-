# CLI Reference

This document summarizes all top-level QOCC CLI groups and key flags.

## Global entrypoint

```bash
qocc --help
```

## `trace` commands

### `qocc trace run`

Primary trace-bundle generation command.

Key flags:

- `--adapter`
- `--input`
- `--pipeline`
- `--out`
- `--seed`
- `--repeat`
- `--db/--no-db`
- `--db-path`
- `--html/--no-html`
- `--html-out`

Exit behavior:

- `0` on success
- `1` on adapter import/runtime errors

### `qocc trace compare`

Compare two bundles and emit text or JSON diff.

Key flags:

- positional: `bundle_a`, `bundle_b`
- `--report`
- `--format {text,json}`

Exit behavior:

- `0` on successful comparison
- `1` on load/processing errors

### `qocc trace replay`

Replay a bundle for reproducibility validation.

### `qocc trace timeline`

ASCII timeline rendering for a bundle.

### `qocc trace html`

Generate interactive HTML report from a bundle.

### `qocc trace watch`

Poll pending hardware jobs and update bundle artifacts.

## `contract` commands

### `qocc contract check`

Evaluate contracts from JSON or `.qocc` DSL.

Key flags:

- `--bundle`
- `--contracts`
- `--max-cache-age-days`

Exit behavior:

- `0` all contracts pass
- `1` failures, parse errors, or runtime errors

## `compile` commands

### `qocc compile search`

Closed-loop optimization for one circuit.

Key flags:

- `--adapter`
- `--input` / `--bundle`
- `--search`
- `--contracts`
- `--topk`
- `--shots`
- `--mode {single,pareto}`
- `--strategy {grid,random,bayesian,evolutionary}`
- `--noise-model`
- `--prior-half-life`
- `--out`

### `qocc compile batch`

Run manifest-driven multi-circuit search.

Key flags:

- `--manifest`
- `--workers`
- `--out`

## `db` commands

### `qocc db ingest`

Ingest bundle rows into regression database.

### `qocc db query`

Query historical rows.

### `qocc db tag`

Tag a run (for baseline workflows).

## `bundle` commands

### `qocc bundle sign`

Sign a bundle with an Ed25519 private key and add `signature.json` plus
manifest provenance metadata.

Key flags:

- `--key` (private key PEM)
- `--signer` (optional identity)

### `qocc bundle verify`

Verify bundle signature with an Ed25519 public key.

Key flags:

- `--key` (public key PEM)
- `--format {text,json}`

## `validate` commands

Validate bundle files against schemas (supports strict mode and output formatting).

## `init` command

### `qocc init`

Bootstrap project scaffolding.

Key flags:

- `--project-root`
- `--adapter`
- `--yes`
- `--run-demo/--no-run-demo`
- `--force`

Generated outputs:

- `contracts/default_contracts.qocc`
- `pipeline_examples/<adapter>_default.json`
- `.github/workflows/qocc_ci.yml`
- `[tool.qocc]` section in `pyproject.toml`

## Exit code summary

- `0`: successful command completion
- `1`: command-level failure (validation errors, runtime exceptions, failing contract gates)
