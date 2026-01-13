#!/usr/bin/env python3
"""Sign index.json with ed25519 private key.

This tool creates a detached signature (index.json.sig) for an index.json file.
The signature is computed over the canonical JSON encoding (no whitespace).

Usage:
    python tools/sign_index.py --index path/to/index.json --key path/to/private_key --out path/to/index.json.sig

The private key should be a 32-byte raw ed25519 private key (binary file).

Example:
    # Generate a key pair (for testing)
    python -c "import nacl.signing; k=nacl.signing.SigningKey.generate(); \
               open('private.key','wb').write(bytes(k)); \
               open('public.key','wb').write(bytes(k.verify_key))"
    
    # Sign an index
    python tools/sign_index.py --index traces/index.json --key private.key --out traces/index.json.sig
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import nacl.signing
except ImportError:
    print("ERROR: PyNaCl not installed. Install with: pip install PyNaCl", file=sys.stderr)
    sys.exit(1)


def sign_index(index_path: Path, key_path: Path, output_path: Path) -> None:
    """Sign index.json with ed25519 private key.
    
    Parameters
    ----------
    index_path : Path
        Path to index.json file
    key_path : Path
        Path to private key (32 bytes raw)
    output_path : Path
        Path to write signature file
    """
    # Load index.json
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")
    
    with open(index_path, 'r') as f:
        index_data = json.load(f)
    
    # Canonical JSON encoding (no whitespace, sorted keys)
    canonical_json = json.dumps(index_data, sort_keys=True, separators=(',', ':')).encode('utf-8')
    
    # Load private key
    if not key_path.exists():
        raise FileNotFoundError(f"Private key not found: {key_path}")
    
    with open(key_path, 'rb') as f:
        key_bytes = f.read()
    
    # Key should be exactly 32 bytes for ed25519
    if len(key_bytes) != 32:
        raise ValueError(f"Private key must be exactly 32 bytes, got {len(key_bytes)}")
    
    # Sign the canonical JSON
    signing_key = nacl.signing.SigningKey(key_bytes)
    signed = signing_key.sign(canonical_json)
    signature = signed.signature
    
    # Write signature to output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(signature)
    
    print(f"✓ Signed {index_path}")
    print(f"✓ Signature written to {output_path}")
    print(f"  Signature length: {len(signature)} bytes")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sign index.json with ed25519 private key",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--index',
        type=Path,
        required=True,
        help='Path to index.json file'
    )
    parser.add_argument(
        '--key',
        type=Path,
        required=True,
        help='Path to private key file (32 bytes raw ed25519 key)'
    )
    parser.add_argument(
        '--out',
        type=Path,
        required=True,
        help='Path to output signature file (e.g., index.json.sig)'
    )
    
    args = parser.parse_args()
    
    try:
        sign_index(args.index, args.key, args.out)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
