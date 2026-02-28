"""Phase 16 tests for pre-commit template and qocc init wizard."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner


def test_pre_commit_template_exists() -> None:
    root = Path(__file__).resolve().parent.parent
    path = root / "examples" / "ci" / "pre_commit_config.yaml"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "qocc-contract-check" in text
    assert "qocc contract check" in text


def test_qocc_init_creates_project_scaffold(tmp_path: Path) -> None:
    from qocc.cli.main import cli

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "init",
            "--project-root",
            str(tmp_path),
            "--yes",
            "--adapter",
            "qiskit",
        ],
    )

    assert result.exit_code == 0
    assert (tmp_path / "contracts" / "default_contracts.qocc").exists()
    assert (tmp_path / "pipeline_examples" / "qiskit_default.json").exists()
    assert (tmp_path / ".github" / "workflows" / "qocc_ci.yml").exists()

    pyproject = (tmp_path / "pyproject.toml").read_text(encoding="utf-8")
    assert "[tool.qocc]" in pyproject
    assert 'adapter = "qiskit"' in pyproject


def test_qocc_init_respects_no_force_then_force(tmp_path: Path) -> None:
    from qocc.cli.main import cli

    contracts_path = tmp_path / "contracts" / "default_contracts.qocc"
    contracts_path.parent.mkdir(parents=True, exist_ok=True)
    contracts_path.write_text("ORIGINAL\n", encoding="utf-8")

    runner = CliRunner()

    res_no_force = runner.invoke(
        cli,
        [
            "init",
            "--project-root",
            str(tmp_path),
            "--yes",
            "--adapter",
            "qiskit",
        ],
    )
    assert res_no_force.exit_code == 0
    assert contracts_path.read_text(encoding="utf-8") == "ORIGINAL\n"

    res_force = runner.invoke(
        cli,
        [
            "init",
            "--project-root",
            str(tmp_path),
            "--yes",
            "--force",
            "--adapter",
            "qiskit",
        ],
    )
    assert res_force.exit_code == 0
    assert "dist_stability" in contracts_path.read_text(encoding="utf-8")
