"""Phase 16.4 tests for documentation scaffold and API generator."""

from __future__ import annotations

from pathlib import Path


def test_docs_scaffold_files_exist() -> None:
    root = Path(__file__).resolve().parent.parent
    required = [
        root / "mkdocs.yml",
        root / "docs" / "index.md",
        root / "docs" / "api_reference.md",
        root / "docs" / "tutorials" / "first-trace-bundle.md",
        root / "docs" / "tutorials" / "writing-contracts.md",
        root / "docs" / "tutorials" / "debugging-regressions.md",
        root / "docs" / "tutorials" / "adding-custom-adapter.md",
        root / "docs" / "architecture" / "trace-model.md",
        root / "docs" / "architecture" / "bundle-format.md",
        root / "docs" / "architecture" / "search-pipeline.md",
        root / "docs" / "contracts" / "reference.md",
        root / "docs" / "cli" / "reference.md",
    ]
    for fp in required:
        assert fp.exists(), f"Missing docs file: {fp}"


def test_api_reference_generator_contains_core_symbols() -> None:
    from docs.generate_api_reference import generate

    content = generate()
    assert "## `qocc.api`" in content
    assert "run_trace(" in content
    assert "search_compile(" in content
    assert "## `qocc.adapters.base`" in content
