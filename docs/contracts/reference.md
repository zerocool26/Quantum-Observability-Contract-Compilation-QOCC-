# Contract Reference

This page summarizes QOCC contract types and the statistical methodology used in evaluation.

## `distribution`

- Purpose: preserve output distribution closeness
- Metrics/tests: TVD, optional chi-square and G-test
- CI methods: bootstrap confidence intervals
- Early stopping: SPRT support where configured

## `observable`

- Purpose: preserve observable expectation values
- CI methods: Hoeffding bounds and bootstrap

## `clifford`

- Purpose: exact Clifford behavior preservation
- Method: stabilizer-based comparison, with fallback to sampling when needed

## `exact`

- Purpose: exact statevector equivalence
- Method: statevector simulation and equivalence checks

## `cost`

- Purpose: enforce resource budgets and optimization constraints
- Typical limits: depth, 2Q gate count, total gate count, duration, proxy error

## `qec`

- Purpose: QEC-oriented quality thresholds
- Inputs: logical error rates, decoder stats, syndrome budget-related metadata

## `zne`

- Purpose: extrapolated zero-noise expectation fidelity
- Method: Richardson extrapolation over configurable noise scale factors

## Composite contracts

Supported operators:

- `all_of`
- `any_of`
- `best_effort`
- `with_fallback`

## Notes on statistical rigor

QOCC uses explicit confidence parameters, resource budgets, and deterministic seed tracking to improve reproducibility and comparability of contract outcomes.
