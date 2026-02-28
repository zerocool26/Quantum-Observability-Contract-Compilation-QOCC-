# Architecture Deep-Dive: Search pipeline

QOCC search composes candidate generation, compilation, scoring, validation, and selection.

## Pipeline stages

1. Candidate generation (`grid`, `random`, `bayesian`, `evolutionary`)
2. Candidate compilation with cache-aware reuse
3. Surrogate scoring (optionally noise-model aware)
4. Top-k expensive validation via simulation
5. Contract evaluation on validated candidates
6. Selection (`single` or Pareto mode)

## Optimization policies

- **Bayesian**: adaptive proposals with optional historical priors and half-life decay
- **Evolutionary**: generation-level search with tournament, crossover, mutation, elitism
- **Batch mode**: multiple circuit-level searches with aggregate metrics and traces

## Observability

Search emits stage and candidate spans, with strategy-specific attributes (`generations_run`, `prior_size`, cache hits, validated counts).
