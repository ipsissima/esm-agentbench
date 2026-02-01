"""GPG-backed signer adapter."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from ports.signer import SignerPort


class GpgSignerAdapter(SignerPort):
    """Adapter that signs artifacts using GPG."""

    def __init__(self, signer_id: str, gpg_key: str) -> None:
        self.signer_id = signer_id
        self._gpg_key = gpg_key

    def sign_bytes(self, data: bytes) -> bytes:
        if not shutil.which("gpg"):
            raise RuntimeError("gpg is required to sign bytes but was not found.")

        with tempfile.TemporaryDirectory(prefix="esm_gpg_") as tmpdir:
            payload_path = Path(tmpdir) / "payload.bin"
            sig_path = Path(tmpdir) / "payload.asc"
            payload_path.write_bytes(data)
            self.sign_file(payload_path, sig_path)
            return sig_path.read_bytes()

    def sign_file(
        self,
        filepath: Path,
        output_path: Optional[Path] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if not shutil.which("gpg"):
            raise RuntimeError("gpg is required to sign files but was not found.")

        output_path = output_path or filepath.with_suffix(filepath.suffix + ".asc")

        cmd = [
            "gpg",
            "--armor",
            "--detach-sign",
            "--default-key",
            self._gpg_key,
            "--output",
            str(output_path),
            str(filepath),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"GPG signing failed: {result.stderr.strip()}")

        metadata = {
            "signer_id": self.signer_id,
            "signature_path": str(output_path),
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        return metadata
