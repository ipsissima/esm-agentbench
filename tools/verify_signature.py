#!/usr/bin/env python3
"""Verify ed25519 signature for index.json.

This tool verifies a detached signature (index.json.sig) for an index.json file.

Usage:
    python tools/verify_signature.py --index path/to/index.json --sig path/to/index.json.sig --pubkey path/to/public_key

The public key should be a 32-byte raw ed25519 public key (binary file).

Example:
    python tools/verify_signature.py --index traces/index.json --sig traces/index.json.sig --pubkey public.key

Exit codes:
    0 - Signature is valid
    1 - Signature is invalid or error occurred
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import nacl.signing
    import nacl.exceptions
except ImportError:
    print(
        "ERROR: PyNaCl not installed. Install with: pip install .[signing] (or pip install PyNaCl)",
        file=sys.stderr,
    )
    sys.exit(1)


def verify_signature(index_path: Path, sig_path: Path, pubkey_path: Path) -> bool:
    """Verify ed25519 signature for index.json.
    
    Parameters
    ----------
    index_path : Path
        Path to index.json file
    sig_path : Path
        Path to signature file
    pubkey_path : Path
        Path to public key (32 bytes raw)
        
    Returns
    -------
    bool
        True if signature is valid, False otherwise
    """
    # Load index.json
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")
    
    with open(index_path, 'r') as f:
        index_data = json.load(f)
    
    # Canonical JSON encoding (no whitespace, sorted keys)
    canonical_json = json.dumps(index_data, sort_keys=True, separators=(',', ':')).encode('utf-8')
    
    # Load signature
    if not sig_path.exists():
        raise FileNotFoundError(f"Signature file not found: {sig_path}")
    
    with open(sig_path, 'rb') as f:
        signature = f.read()
    
    # Load public key
    if not pubkey_path.exists():
        raise FileNotFoundError(f"Public key not found: {pubkey_path}")
    
    with open(pubkey_path, 'rb') as f:
        pubkey_bytes = f.read()
    
    # Public key should be exactly 32 bytes for ed25519
    if len(pubkey_bytes) != 32:
        raise ValueError(f"Public key must be exactly 32 bytes, got {len(pubkey_bytes)}")
    
    # Verify signature
    verify_key = nacl.signing.VerifyKey(pubkey_bytes)
    
    try:
        # Combine signature and message for verification
        signed_message = signature + canonical_json
        verify_key.verify(signed_message)
        return True
    except nacl.exceptions.BadSignatureError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify ed25519 signature for index.json",
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
        '--sig',
        type=Path,
        required=True,
        help='Path to signature file (e.g., index.json.sig)'
    )
    parser.add_argument(
        '--pubkey',
        type=Path,
        required=True,
        help='Path to public key file (32 bytes raw ed25519 key)'
    )
    
    args = parser.parse_args()
    
    try:
        is_valid = verify_signature(args.index, args.sig, args.pubkey)
        
        if is_valid:
            print(f"✓ Signature is VALID for {args.index}")
            print(f"  Verified with {args.pubkey}")
            sys.exit(0)
        else:
            print(f"✗ Signature is INVALID for {args.index}", file=sys.stderr)
            print(f"  Verification failed with {args.pubkey}", file=sys.stderr)
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
