#!/usr/bin/env bash
set -euo pipefail
# This script is invoked from the workflows. It signs .kernel_out/kernel_verified.so.sha256
# using tools/sign_index.py if available. It generates ephemeral keys if needed.

python3 - <<'PY'
import os
import sys

# If the tools sign_index.py is missing, bail out (workflow will continue).
if not os.path.exists("tools/sign_index.py"):
    print("tools/sign_index.py not found; skipping signing", file=sys.stderr)
    sys.exit(0)

# Ensure UELAT exists.
os.makedirs("UELAT", exist_ok=True)

if not os.path.exists("UELAT/private.key"):
    try:
        from nacl.signing import SigningKey
        k = SigningKey.generate()
        open("UELAT/private.key","wb").write(k.encode())
        open("UELAT/public.key","wb").write(k.verify_key.encode())
        print("Generated ephemeral signing key pair in UELAT/")
    except Exception as e:
        print(
            "PyNaCl not available or error generating key, skipping signature generation:",
            e,
            file=sys.stderr,
        )
        sys.exit(0)

# Sign the sha file.
ret = os.system(
    "python3 tools/sign_index.py --index .kernel_out/kernel_verified.so.sha256 "
    "--key UELAT/private.key --out .kernel_out/kernel_verified.so.sha256.sig"
)
if ret != 0:
    print("Signing tool returned non-zero exit; aborting", file=sys.stderr)
    sys.exit(ret)
print(
    "Signed .kernel_out/kernel_verified.so.sha256 -> "
    ".kernel_out/kernel_verified.so.sha256.sig"
)
PY

ls -la .kernel_out || true
