"""Phase 17.1 tests for bundle signing and provenance."""

from __future__ import annotations

import json
import importlib
from pathlib import Path

import pytest
from click.testing import CliRunner


def _require_crypto() -> None:
    pytest.importorskip("cryptography")


def _write_keys(tmp_path: Path) -> tuple[Path, Path]:
    _require_crypto()
    serialization = importlib.import_module("cryptography.hazmat.primitives.serialization")
    ed25519 = importlib.import_module("cryptography.hazmat.primitives.asymmetric.ed25519")

    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    priv_path = tmp_path / "ed25519_private.pem"
    pub_path = tmp_path / "ed25519_public.pem"

    priv_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    pub_path.write_bytes(
        public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return priv_path, pub_path


def _make_bundle_dir(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "created_at": "2026-01-01T00:00:00+00:00",
                "run_id": "signing-test",
                "qocc_version": "0.1.0",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def test_sign_and_verify_directory_bundle(tmp_path: Path) -> None:
    _require_crypto()
    from qocc.core.signing import sign_bundle, verify_bundle

    bundle = _make_bundle_dir(tmp_path)
    priv, pub = _write_keys(tmp_path)

    payload = sign_bundle(bundle, priv, signer_identity="ci-bot")
    assert payload["algorithm"] == "ed25519"
    assert payload["signer"] == "ci-bot"
    assert (bundle / "signature.json").exists()

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    provenance = manifest.get("provenance", {})
    assert provenance.get("signer") == "ci-bot"
    assert provenance.get("signature_file") == "signature.json"

    verification = verify_bundle(bundle, pub)
    assert verification.valid is True
    assert verification.signer == "ci-bot"


def test_verify_detects_manifest_tamper(tmp_path: Path) -> None:
    _require_crypto()
    from qocc.core.signing import sign_bundle, verify_bundle

    bundle = _make_bundle_dir(tmp_path)
    priv, pub = _write_keys(tmp_path)

    sign_bundle(bundle, priv)

    tampered = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    tampered["qocc_version"] = "9.9.9-tampered"
    (bundle / "manifest.json").write_text(json.dumps(tampered, indent=2) + "\n", encoding="utf-8")

    verification = verify_bundle(bundle, pub)
    assert verification.valid is False
    assert "hash mismatch" in verification.reason


def test_sign_and_verify_zip_bundle(tmp_path: Path) -> None:
    _require_crypto()
    from qocc.core.artifacts import ArtifactStore
    from qocc.core.signing import sign_bundle, verify_bundle

    root = _make_bundle_dir(tmp_path)
    zip_path = tmp_path / "bundle.zip"
    ArtifactStore(root).export_zip(zip_path)

    priv, pub = _write_keys(tmp_path)

    sign_bundle(zip_path, priv, signer_identity="zip-signer")
    verification = verify_bundle(zip_path, pub)
    assert verification.valid is True
    assert verification.signer == "zip-signer"


def test_bundle_cli_sign_verify(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from qocc.cli.main import cli

    called: dict[str, object] = {}

    def _fake_sign(bundle_path: str, private_key_path: str, signer_identity: str | None = None) -> dict[str, object]:
        called["sign"] = (bundle_path, private_key_path, signer_identity)
        return {
            "signer": signer_identity or "fake",
            "manifest_hash": "abcd" * 16,
            "public_key_fingerprint": "f00d" * 16,
        }

    class _FakeResult:
        def __init__(self, valid: bool = True) -> None:
            self.valid = valid
            self.signer = "fake"
            self.reason = "ok" if valid else "bad"
            self.public_key_fingerprint = "f00d" * 16

        def to_dict(self) -> dict[str, object]:
            return {
                "valid": self.valid,
                "signer": self.signer,
                "reason": self.reason,
                "public_key_fingerprint": self.public_key_fingerprint,
            }

    def _fake_verify(bundle_path: str, public_key_path: str) -> _FakeResult:
        called["verify"] = (bundle_path, public_key_path)
        return _FakeResult(valid=True)

    monkeypatch.setattr("qocc.core.signing.sign_bundle", _fake_sign)
    monkeypatch.setattr("qocc.core.signing.verify_bundle", _fake_verify)

    bundle = _make_bundle_dir(tmp_path)
    key_file = tmp_path / "dummy.pem"
    key_file.write_text("dummy", encoding="utf-8")

    runner = CliRunner()
    sign_res = runner.invoke(cli, ["bundle", "sign", "--key", str(key_file), str(bundle)])
    assert sign_res.exit_code == 0

    verify_res = runner.invoke(cli, ["bundle", "verify", "--key", str(key_file), str(bundle)])
    assert verify_res.exit_code == 0

    assert "sign" in called
    assert "verify" in called
