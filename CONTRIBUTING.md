# Contributing to QOCC

Thank you for your interest in contributing to QOCC! This document provides guidelines and instructions for contributing.

## Getting Started

### Prerequisites
- Python 3.11+
- Git

### Development Setup

```bash
# Clone the repository
git clone https://github.com/zerocool26/Quantum-Observability-Contract-Compilation-QOCC-.git
cd Quantum-Observability-Contract-Compilation-QOCC-

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install in development mode with all extras
pip install -e ".[all]"

# Run tests
pytest

# Run linter
ruff check .

# Run type checker
mypy qocc/
```

## Code Standards

### Type Hints
- **All** functions and methods must have type hints.
- Use `from __future__ import annotations` in every module.

### Formatting & Linting
- We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.
- Line length: 100 characters.
- Run `ruff check .` before committing.

### Error Handling
- Every exception must become a trace event with stack + context.
- Use structured logging; avoid bare `print()` in library code.

### Testing
- All new features must include tests.
- Tests live in `tests/`.
- Run with `pytest -v`.

## Project Structure

```
qocc/
├── core/          # Circuit handles, canonicalization, hashing, artifacts, schemas
├── trace/         # Span model, emitter, exporters
├── adapters/      # Vendor-specific adapters (Qiskit, Cirq, etc.)
├── metrics/       # Circuit metrics computation
├── contracts/     # Semantic + cost contract evaluation
├── search/        # Closed-loop compilation search (v3)
├── cli/           # CLI commands
├── api.py         # Public Python API
tests/             # Test suite
examples/          # End-to-end examples
```

## Adding a New Adapter

1. Create `qocc/adapters/your_adapter.py`
2. Subclass `BaseAdapter` from `qocc.adapters.base`
3. Implement all abstract methods:
   - `name()`, `ingest()`, `normalize()`, `export()`, `compile()`,
     `get_metrics()`, `hash()`, `describe_backend()`
4. Register with `register_adapter("your_name", YourAdapter)`
5. Add tests in `tests/test_your_adapter.py`
6. Add an example in `examples/`

## Adding a New Contract Evaluator

1. Create your evaluator in `qocc/contracts/`
2. Accept a `ContractSpec` and relevant data
3. Return a `ContractResult`
4. Add tests validating pass/fail behavior
5. Wire into the `check_contract` API if it's a new contract type

## Pull Request Process

1. Fork the repository and create a feature branch.
2. Make your changes with appropriate tests.
3. Ensure all tests pass: `pytest`
4. Ensure linting passes: `ruff check .`
5. Write a clear PR description explaining what and why.
6. Reference any related issues.

## Commit Messages

Use clear, descriptive commit messages:
```
feat(adapters): add pytket adapter with basic compilation support
fix(hashing): ensure stable ordering in QASM canonicalization
test(contracts): add edge cases for TVD bootstrap CI
docs: update roadmap with QEC milestones
```

## Reporting Issues

- Use GitHub Issues.
- Include: Python version, OS, installed packages, and a minimal reproducer.
- For regressions, include a Trace Bundle diff if possible.

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
