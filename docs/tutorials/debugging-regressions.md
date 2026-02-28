# Tutorial: Debugging regressions

## Goal

Find why a compilation result regressed and identify likely root causes.

## 1) Compare two bundles

```bash
qocc trace compare baseline.zip candidate.zip --report reports/
```

This produces metric/env/contract diffs and a regression analysis summary.

## 2) Ingest history into regression DB

```bash
qocc db ingest baseline.zip
qocc db ingest candidate.zip
qocc db tag baseline.zip --tag baseline
```

## 3) Query historical patterns

```bash
qocc db query --adapter qiskit --since 2026-01-01
```

## 4) CI automation pattern

- Build baseline bundle from `main`
- Build candidate bundle from PR
- Compare and post summary
- Fail if regressions exceed threshold

Reference templates:

- `examples/ci/qocc_pr_check.yml`
- `examples/ci/qocc_baseline.yml`
