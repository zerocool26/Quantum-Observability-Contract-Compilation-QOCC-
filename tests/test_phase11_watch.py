"""Phase 11.3 tests for hardware watch mode."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner


def _make_bundle_dir(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    root.mkdir(parents=True, exist_ok=True)
    (root / "hardware").mkdir(parents=True, exist_ok=True)
    (root / "trace.jsonl").write_text("", encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({"run_id": "r1"}), encoding="utf-8")
    (root / "metrics.json").write_text(json.dumps({"input": {}, "compiled": {}}), encoding="utf-8")
    return root


def test_watch_bundle_updates_pending_and_writes_results(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from qocc.trace import watch as w

    root = _make_bundle_dir(tmp_path)
    pending = [
        {
            "provider": "ibm",
            "backend_name": "ibm_fake",
            "backend_spec": {"backend_name": "ibm_fake"},
            "job_id": "job-1",
            "shots": 16,
            "status": "pending",
        }
    ]
    (root / "hardware" / "pending_jobs.json").write_text(json.dumps(pending), encoding="utf-8")

    calls = {"n": 0}

    def _fake_poll(job: dict[str, Any]) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] == 1:
            return {"status": "running", "done": False}
        return {
            "status": "completed",
            "done": True,
            "result": {
                "counts": {"00": 16},
                "backend_name": "ibm_fake",
                "metadata": {"provider": "ibm"},
            },
        }

    monkeypatch.setattr(w, "_poll_job_entry", _fake_poll)

    summary = w.watch_bundle_jobs(str(root), poll_interval_s=0.001, timeout_s=2.0)
    assert summary["completed"] == 1
    assert summary["failed"] == 0

    pending_after = json.loads((root / "hardware" / "pending_jobs.json").read_text(encoding="utf-8"))
    assert pending_after[0]["status"].lower() == "completed"

    hardware_json = json.loads((root / "hardware" / "hardware.json").read_text(encoding="utf-8"))
    assert hardware_json["counts"] == {"00": 16}

    result_file = root / "hardware" / "job-1_result.json"
    assert result_file.exists()

    trace_lines = [ln for ln in (root / "trace.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert trace_lines
    parsed = [json.loads(x) for x in trace_lines]
    assert any(s.get("name") == "job_complete" for s in parsed)


def test_watch_bundle_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from qocc.trace import watch as w

    root = _make_bundle_dir(tmp_path)
    pending = [{"provider": "ibm", "job_id": "job-1", "status": "pending", "backend_spec": {"backend_name": "ibm_fake"}}]
    (root / "hardware" / "pending_jobs.json").write_text(json.dumps(pending), encoding="utf-8")

    monkeypatch.setattr(w, "_poll_job_entry", lambda _job: {"status": "running", "done": False})

    with pytest.raises(TimeoutError):
        w.watch_bundle_jobs(str(root), poll_interval_s=0.001, timeout_s=0.01)


def test_watch_on_complete_hook(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from qocc.trace import watch as w

    root = _make_bundle_dir(tmp_path)
    pending = [{"provider": "ibm", "job_id": "job-1", "status": "pending", "backend_spec": {"backend_name": "ibm_fake"}}]
    (root / "hardware" / "pending_jobs.json").write_text(json.dumps(pending), encoding="utf-8")

    monkeypatch.setattr(
        w,
        "_poll_job_entry",
        lambda _job: {"status": "completed", "done": True, "result": {"counts": {"0": 1}, "metadata": {}}},
    )

    ran: dict[str, Any] = {}

    def _fake_run(command: str, bundle_path: str) -> dict[str, Any]:
        ran["command"] = command
        ran["bundle_path"] = bundle_path
        return {"returncode": 0, "stdout": "ok", "stderr": "", "command": ["echo", "ok"]}

    monkeypatch.setattr(w, "_run_on_complete", _fake_run)

    summary = w.watch_bundle_jobs(str(root), poll_interval_s=0.001, timeout_s=1.0, on_complete="echo {bundle}")
    assert summary["on_complete"]["returncode"] == 0
    assert ran["command"] == "echo {bundle}"


def test_trace_watch_cli_invokes_engine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from qocc.cli.commands_trace import trace
    from qocc.trace import watch as w

    root = _make_bundle_dir(tmp_path)

    captured: dict[str, Any] = {}

    def _fake_watch(bundle_path: str, poll_interval_s: float, timeout_s: float | None, on_complete: str | None):
        captured["bundle"] = bundle_path
        captured["poll"] = poll_interval_s
        captured["timeout"] = timeout_s
        captured["on_complete"] = on_complete
        return {"completed": 0, "failed": 0, "pending": 0}

    monkeypatch.setattr(w, "watch_bundle_jobs", _fake_watch)

    runner = CliRunner()
    result = runner.invoke(
        trace,
        [
            "watch",
            "--bundle",
            str(root),
            "--poll-interval",
            "0.5",
            "--timeout",
            "3",
            "--on-complete",
            "echo {bundle}",
        ],
    )
    assert result.exit_code == 0
    assert captured["bundle"] == str(root)
    assert captured["poll"] == 0.5
    assert captured["timeout"] == 3.0
