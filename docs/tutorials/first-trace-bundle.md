# Tutorial: Your first trace bundle

## Goal

Compile a circuit with full observability and produce a portable bundle.

## Prerequisites

- QOCC installed (`pip install -e ".[all]"`)
- A circuit file such as `examples/ghz.qasm`

## Steps

1. Run a trace:

```bash
qocc trace run --adapter qiskit --input examples/ghz.qasm --out bundle.zip
```

2. Inspect core artifacts in the extracted bundle:

- `manifest.json`
- `metrics.json`
- `trace.jsonl`
- `circuits/input.qasm`
- `circuits/selected.qasm`

3. Validate schema compliance:

```bash
qocc validate --bundle bundle.zip --strict
```

4. Compare against another run:

```bash
qocc trace compare bundle.zip other_bundle.zip --report reports/
```

## Success criteria

- `trace run` exits with code `0`
- bundle zip contains required files
- metrics and trace spans are present
