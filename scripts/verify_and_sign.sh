#!/bin/bash
# Verify certificate and create signed bundle
#
# This script runs the verification pipeline:
# 1. Export kernel input from trace
# 2. Run verified kernel
# 3. Create signed certificate bundle
#
# Usage:
#   ./scripts/verify_and_sign.sh <trace.json> [--key GPG_KEY] [--output-dir DIR]
#
# Options:
#   --key GPG_KEY      GPG key ID for signing (optional)
#   --output-dir DIR   Output directory for bundle (default: ./bundle)
#   --precision BITS   Precision bits for kernel (default: 128)
#   --mode MODE        Kernel mode: prototype|arb|mpfi (default: prototype)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default values
GPG_KEY=""
OUTPUT_DIR="./bundle"
PRECISION=128
KERNEL_MODE="prototype"

# Parse arguments
TRACE_FILE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --key)
            GPG_KEY="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --precision)
            PRECISION="$2"
            shift 2
            ;;
        --mode)
            KERNEL_MODE="$2"
            shift 2
            ;;
        -*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            if [ -z "$TRACE_FILE" ]; then
                TRACE_FILE="$1"
            else
                echo "Unexpected argument: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

if [ -z "$TRACE_FILE" ]; then
    echo "Usage: $0 <trace.json> [--key GPG_KEY] [--output-dir DIR]"
    exit 1
fi

if [ ! -f "$TRACE_FILE" ]; then
    echo "ERROR: Trace file not found: $TRACE_FILE"
    exit 1
fi

echo "=== Verify and Sign Certificate ==="
echo "Trace: $TRACE_FILE"
echo "Output: $OUTPUT_DIR"
echo "Precision: $PRECISION bits"
echo "Kernel mode: $KERNEL_MODE"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate unique trace ID
TRACE_ID=$(sha256sum "$TRACE_FILE" | cut -d' ' -f1 | head -c 16)
echo "Trace ID: $TRACE_ID"

# Create temp files
KERNEL_INPUT=$(mktemp --suffix=.json)
KERNEL_OUTPUT=$(mktemp --suffix=.json)
CERTIFICATE=$(mktemp --suffix=.json)

cleanup() {
    rm -f "$KERNEL_INPUT" "$KERNEL_OUTPUT" "$CERTIFICATE"
}
trap cleanup EXIT

# Step 1: Export kernel input
echo ""
echo "=== Step 1: Export kernel input ==="
python3 -c "
import sys
sys.path.insert(0, '$REPO_ROOT')

import json
import numpy as np
from certificates.make_certificate import compute_certificate, export_kernel_input

# Load trace
with open('$TRACE_FILE', 'r') as f:
    trace = json.load(f)

# Extract embeddings from trace
if 'embeddings' in trace:
    embeddings = np.array(trace['embeddings'])
elif 'steps' in trace:
    embeddings = np.array([s.get('embedding', [0]*128) for s in trace['steps']])
else:
    # Generate placeholder for testing
    print('WARNING: No embeddings in trace, using placeholder')
    embeddings = np.random.randn(20, 128)

# Augment with bias
X_aug = np.concatenate([embeddings, np.ones((len(embeddings), 1))], axis=1)

# Export kernel input
export_kernel_input(
    X_aug=X_aug,
    trace_id='$TRACE_ID',
    output_path='$KERNEL_INPUT',
    precision_bits=$PRECISION,
    kernel_mode='$KERNEL_MODE',
)

# Also compute certificate
cert = compute_certificate(embeddings, r=10, kernel_strict=False)
with open('$CERTIFICATE', 'w') as f:
    # Remove non-serializable fields
    cert_clean = {k: v for k, v in cert.items() if not isinstance(v, np.ndarray)}
    json.dump(cert_clean, f, indent=2, default=str)

print('Kernel input exported to $KERNEL_INPUT')
print('Certificate computed')
" || {
    echo "ERROR: Failed to export kernel input"
    exit 1
}

# Step 2: Run verified kernel
echo ""
echo "=== Step 2: Run verified kernel ==="

if [ "$KERNEL_MODE" = "prototype" ]; then
    # Run Python prototype
    python3 "$REPO_ROOT/kernel/prototype/prototype_kernel.py" \
        "$KERNEL_INPUT" \
        "$KERNEL_OUTPUT" \
        --precision "$PRECISION" || {
            echo "ERROR: Prototype kernel failed"
            exit 1
        }
else
    # Run Docker kernel
    docker run --rm \
        -v "$KERNEL_INPUT:/data/kernel_input.json:ro" \
        -v "$KERNEL_OUTPUT:/data/kernel_output.json:rw" \
        -e "PRECISION_BITS=$PRECISION" \
        -e "KERNEL_MODE=$KERNEL_MODE" \
        "${ESM_KERNEL_IMAGE:-ipsissima/kernel:latest}" || {
            echo "ERROR: Docker kernel failed"
            exit 1
        }
fi

# Check kernel output
KERNEL_PASS=$(python3 -c "
import json
with open('$KERNEL_OUTPUT') as f:
    out = json.load(f)
print('true' if out.get('checks', {}).get('theoretical_bound', {}).get('pass', False) else 'false')
")

if [ "$KERNEL_PASS" = "true" ]; then
    echo "Kernel checks PASSED"
else
    echo "WARNING: Kernel checks FAILED"
fi

# Step 3: Create bundle
echo ""
echo "=== Step 3: Create certificate bundle ==="

# Copy files to bundle
cp "$TRACE_FILE" "$OUTPUT_DIR/trace.json"
cp "$KERNEL_INPUT" "$OUTPUT_DIR/kernel_input.json"
cp "$KERNEL_OUTPUT" "$OUTPUT_DIR/kernel_output.json"
cp "$CERTIFICATE" "$OUTPUT_DIR/certificate.json"

# Create metadata
python3 -c "
import sys
sys.path.insert(0, '$REPO_ROOT')

from certificates.cert_bundle import create_metadata
import json

files = {
    'trace.json': '$OUTPUT_DIR/trace.json',
    'kernel_input.json': '$OUTPUT_DIR/kernel_input.json',
    'kernel_output.json': '$OUTPUT_DIR/kernel_output.json',
    'certificate.json': '$OUTPUT_DIR/certificate.json',
}

metadata = create_metadata(
    files=files,
    kernel_mode='$KERNEL_MODE',
    extra={'trace_id': '$TRACE_ID', 'precision_bits': $PRECISION},
)

with open('$OUTPUT_DIR/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print('Metadata created')
"

# Step 4: Sign bundle (if GPG key provided)
if [ -n "$GPG_KEY" ]; then
    echo ""
    echo "=== Step 4: Sign bundle ==="

    gpg --armor --detach-sign \
        --default-key "$GPG_KEY" \
        --output "$OUTPUT_DIR/signature.asc" \
        "$OUTPUT_DIR/metadata.json" || {
            echo "WARNING: GPG signing failed"
        }

    if [ -f "$OUTPUT_DIR/signature.asc" ]; then
        echo "Bundle signed with key: $GPG_KEY"
    fi
else
    echo ""
    echo "=== Step 4: Signing skipped (no GPG key) ==="
fi

# Create archive
echo ""
echo "=== Creating archive ==="
ARCHIVE="$OUTPUT_DIR.tar.gz"
tar -czf "$ARCHIVE" -C "$(dirname "$OUTPUT_DIR")" "$(basename "$OUTPUT_DIR")"
echo "Archive created: $ARCHIVE"

# Sign archive if GPG key provided
if [ -n "$GPG_KEY" ] && [ -f "$ARCHIVE" ]; then
    gpg --armor --detach-sign \
        --default-key "$GPG_KEY" \
        --output "$ARCHIVE.asc" \
        "$ARCHIVE" 2>/dev/null || true

    if [ -f "$ARCHIVE.asc" ]; then
        echo "Archive signed: $ARCHIVE.asc"
    fi
fi

echo ""
echo "=== Done ==="
echo "Bundle directory: $OUTPUT_DIR"
echo "Bundle archive: $ARCHIVE"
echo ""
echo "To verify:"
echo "  tar xzf $ARCHIVE"
echo "  python -m certificates.cert_bundle verify $(basename $OUTPUT_DIR)/"
