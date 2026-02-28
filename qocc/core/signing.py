"""Bundle signing and verification utilities (Phase 17.1)."""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class VerificationResult:
    """Result of bundle signature verification."""

    valid: bool
    signer: str | None = None
    reason: str = ""
    manifest_hash: str | None = None
    expected_manifest_hash: str | None = None
    timestamp: str | None = None
    public_key_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "signer": self.signer,
            "reason": self.reason,
            "manifest_hash": self.manifest_hash,
            "expected_manifest_hash": self.expected_manifest_hash,
            "timestamp": self.timestamp,
            "public_key_fingerprint": self.public_key_fingerprint,
        }


def sign_bundle(
    bundle_path: str | Path,
    private_key_path: str | Path,
    signer_identity: str | None = None,
) -> dict[str, Any]:
    """Sign a bundle with Ed25519 and write ``signature.json``.

    Signature payload contains signer identity, timestamp, manifest hash, and
    signature over ``SHA-256(manifest.json)``.
    """
    ed25519, serialization, _ = _import_crypto()

    root, extracted_tmp, is_zip = _prepare_bundle_root(bundle_path)
    try:
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found in bundle: {bundle_path}")

        manifest_bytes = manifest_path.read_bytes()
        manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()

        private_key_raw = Path(private_key_path).read_bytes()
        key = serialization.load_pem_private_key(private_key_raw, password=None)
        if not isinstance(key, ed25519.Ed25519PrivateKey):
            raise ValueError("Private key is not an Ed25519 key")

        signature = key.sign(bytes.fromhex(manifest_hash))
        public_key = key.public_key()
        pub_raw = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        pub_fingerprint = hashlib.sha256(pub_raw).hexdigest()

        timestamp = datetime.now(timezone.utc).isoformat()
        signer = signer_identity or f"key:{pub_fingerprint[:16]}"
        payload: dict[str, Any] = {
            "algorithm": "ed25519",
            "signer": signer,
            "timestamp": timestamp,
            "manifest_hash": manifest_hash,
            "signature": base64.b64encode(signature).decode("ascii"),
            "public_key_fingerprint": pub_fingerprint,
        }

        (root / "signature.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

        manifest = json.loads(manifest_bytes.decode("utf-8"))
        provenance = manifest.get("provenance")
        if not isinstance(provenance, dict):
            provenance = {}
        provenance.update(
            {
                "signer": signer,
                "public_key_fingerprint": pub_fingerprint,
                "signed_at": timestamp,
                "signature_file": "signature.json",
            }
        )
        manifest["provenance"] = provenance
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        if is_zip:
            _pack_to_zip(root, Path(bundle_path))

        return payload
    finally:
        if extracted_tmp is not None and extracted_tmp.exists():
            shutil.rmtree(extracted_tmp, ignore_errors=True)


def verify_bundle(
    bundle_path: str | Path,
    public_key_path: str | Path,
) -> VerificationResult:
    """Verify a signed bundle using an Ed25519 public key."""
    ed25519, serialization, invalid_sig = _import_crypto()

    root, extracted_tmp, _ = _prepare_bundle_root(bundle_path)
    try:
        manifest_path = root / "manifest.json"
        signature_path = root / "signature.json"
        if not manifest_path.exists():
            return VerificationResult(valid=False, reason="manifest.json missing")
        if not signature_path.exists():
            return VerificationResult(valid=False, reason="signature.json missing")

        manifest_bytes = manifest_path.read_bytes()
        expected_manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()

        payload = json.loads(signature_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return VerificationResult(valid=False, reason="Invalid signature payload format")

        signer = str(payload.get("signer")) if payload.get("signer") is not None else None
        manifest_hash = payload.get("manifest_hash")
        timestamp = str(payload.get("timestamp")) if payload.get("timestamp") is not None else None
        payload_fingerprint = payload.get("public_key_fingerprint")

        if not isinstance(manifest_hash, str):
            return VerificationResult(
                valid=False,
                signer=signer,
                reason="signature payload missing manifest_hash",
                expected_manifest_hash=expected_manifest_hash,
                timestamp=timestamp,
                public_key_fingerprint=str(payload_fingerprint) if payload_fingerprint is not None else None,
            )

        if manifest_hash != expected_manifest_hash:
            return VerificationResult(
                valid=False,
                signer=signer,
                reason="manifest hash mismatch",
                manifest_hash=manifest_hash,
                expected_manifest_hash=expected_manifest_hash,
                timestamp=timestamp,
                public_key_fingerprint=str(payload_fingerprint) if payload_fingerprint is not None else None,
            )

        signature_b64 = payload.get("signature")
        if not isinstance(signature_b64, str):
            return VerificationResult(
                valid=False,
                signer=signer,
                reason="signature payload missing signature",
                manifest_hash=manifest_hash,
                expected_manifest_hash=expected_manifest_hash,
                timestamp=timestamp,
                public_key_fingerprint=str(payload_fingerprint) if payload_fingerprint is not None else None,
            )

        public_key_raw = Path(public_key_path).read_bytes()
        key = serialization.load_pem_public_key(public_key_raw)
        if not isinstance(key, ed25519.Ed25519PublicKey):
            return VerificationResult(
                valid=False,
                signer=signer,
                reason="Public key is not an Ed25519 key",
                manifest_hash=manifest_hash,
                expected_manifest_hash=expected_manifest_hash,
                timestamp=timestamp,
                public_key_fingerprint=str(payload_fingerprint) if payload_fingerprint is not None else None,
            )

        pub_raw = key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        actual_fingerprint = hashlib.sha256(pub_raw).hexdigest()
        if isinstance(payload_fingerprint, str) and payload_fingerprint != actual_fingerprint:
            return VerificationResult(
                valid=False,
                signer=signer,
                reason="public key fingerprint mismatch",
                manifest_hash=manifest_hash,
                expected_manifest_hash=expected_manifest_hash,
                timestamp=timestamp,
                public_key_fingerprint=actual_fingerprint,
            )

        signature = base64.b64decode(signature_b64)
        try:
            key.verify(signature, bytes.fromhex(manifest_hash))
        except invalid_sig:
            return VerificationResult(
                valid=False,
                signer=signer,
                reason="invalid signature",
                manifest_hash=manifest_hash,
                expected_manifest_hash=expected_manifest_hash,
                timestamp=timestamp,
                public_key_fingerprint=actual_fingerprint,
            )

        return VerificationResult(
            valid=True,
            signer=signer,
            reason="ok",
            manifest_hash=manifest_hash,
            expected_manifest_hash=expected_manifest_hash,
            timestamp=timestamp,
            public_key_fingerprint=actual_fingerprint,
        )
    finally:
        if extracted_tmp is not None and extracted_tmp.exists():
            shutil.rmtree(extracted_tmp, ignore_errors=True)


def _import_crypto() -> tuple[Any, Any, Any]:
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ed25519

        return ed25519, serialization, InvalidSignature
    except ImportError as exc:
        raise ImportError(
            "Bundle signing requires cryptography. Install with: pip install 'qocc[signing]'"
        ) from exc


def _prepare_bundle_root(bundle_path: str | Path) -> tuple[Path, Path | None, bool]:
    path = Path(bundle_path)
    if path.is_dir():
        return path, None, False
    if path.is_file() and path.suffix == ".zip":
        tmp = Path(tempfile.mkdtemp(prefix="qocc_sign_"))
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(tmp)
        return tmp, tmp, True
    raise ValueError(f"Bundle path must be a directory or .zip file: {bundle_path}")


def _pack_to_zip(root: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(root.rglob("*")):
            if file.is_file():
                zf.write(file, file.relative_to(root))
