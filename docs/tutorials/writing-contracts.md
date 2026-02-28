# Tutorial: Writing contracts

## Goal

Define semantic and resource constraints that can gate optimization choices.

## Contract formats

- JSON (`.json`)
- QOCC DSL (`.qocc`)

## Example DSL

```text
contract dist_stability:
    type: distribution
    tolerance: tvd <= 0.08
    confidence: 0.95
    shots: 2048 .. 16384

contract depth_budget:
    type: cost
    assert: depth <= input_depth + 5
    assert: two_qubit_gates <= input_gates_2q + 12
```

## Evaluate contracts

```bash
qocc contract check --bundle bundle.zip --contracts contracts/default_contracts.qocc
```

## Caching controls

Use cache age control when evaluating repeatedly:

```bash
qocc contract check --bundle bundle.zip --contracts contracts/default_contracts.qocc --max-cache-age-days 7
```

## Exit semantics

- `0`: all contracts pass
- `1`: at least one contract fails or contract loading fails
