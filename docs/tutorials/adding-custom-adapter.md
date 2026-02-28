# Tutorial: Adding a custom adapter

## Goal

Extend QOCC with a backend-specific adapter while preserving observability and reproducibility.

## Implement adapter class

Create `qocc/adapters/my_adapter.py` with a subclass of `BaseAdapter` implementing:

- `name()`
- `ingest()`
- `normalize()`
- `export()`
- `compile()`
- `get_metrics()`
- `hash()`
- `describe_backend()`

Optional capabilities:

- `simulate()`
- `execute()` for hardware jobs

## Register adapter

- Programmatically via `register_adapter("my_backend", MyAdapter)`
- Or via entry points in `pyproject.toml`

## Validation checklist

- deterministic `stable_hash` behavior
- span emission for compile stages
- metrics keys align with schemas
- unit tests under `tests/`

## CI recommendations

- include at least one trace-run smoke test
- include one contract-check smoke test
