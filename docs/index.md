# QOCC Documentation

QOCC is a trace-first, vendor-agnostic platform for reproducible quantum compilation and contract-driven validation.

## Scope

This documentation scaffold includes:

- API reference generated from Python docstrings
- End-to-end tutorials for common workflows
- Architecture deep dives for trace, bundle, and search systems
- Contract reference with statistical methodology
- CLI command, flag, and exit-code reference

## Quick navigation

- API: [api_reference.md](api_reference.md)
- Tutorials: [tutorials/first-trace-bundle.md](tutorials/first-trace-bundle.md)
- Architecture: [architecture/trace-model.md](architecture/trace-model.md)
- Contracts: [contracts/reference.md](contracts/reference.md)
- CLI: [cli/reference.md](cli/reference.md)

## Regenerating API docs

Run:

```bash
python docs/generate_api_reference.py
```

This regenerates `docs/api_reference.md` from package docstrings and signatures.
