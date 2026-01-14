import json
import subprocess
import sys
from pathlib import Path


def test_sign_and_verify(tmp_path):
    try:
        import nacl.signing  # type: ignore
    except Exception:
        import pytest

        pytest.skip("PyNaCl not installed; skipping signature attestation")

    index = tmp_path / "index.json"
    key = tmp_path / "private.key"
    pub = tmp_path / "public.key"
    sig = tmp_path / "index.json.sig"

    index.write_text(json.dumps({"foo": "bar"}, sort_keys=True, separators=(",", ":")), encoding="utf-8")

    import nacl.signing as signing

    k = signing.SigningKey.generate()
    key.write_bytes(bytes(k))
    pub.write_bytes(bytes(k.verify_key))

    p = subprocess.run(
        [
            sys.executable,
            str(Path("tools") / "sign_index.py"),
            "--index",
            str(index),
            "--key",
            str(key),
            "--out",
            str(sig),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert p.returncode == 0, f"sign_index failed: {p.stdout}\n{p.stderr}"
    assert sig.exists()

    p2 = subprocess.run(
        [
            sys.executable,
            str(Path("tools") / "verify_signature.py"),
            "--index",
            str(index),
            "--sig",
            str(sig),
            "--pubkey",
            str(pub),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert p2.returncode == 0, f"verify_signature failed: {p2.stdout}\n{p2.stderr}"
