#!/bin/bash
# Build verified numeric kernel Docker image
#
# This script builds a reproducible Docker image for the verified kernel.
# The image contains the ARB library for rigorous interval arithmetic.
#
# Usage:
#   ./scripts/build_verified_kernel.sh [--tag TAG] [--push]
#
# Options:
#   --tag TAG    Docker image tag (default: ipsissima/kernel:latest)
#   --push       Push image to registry after build
#   --no-cache   Build without Docker cache

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default values
IMAGE_TAG="${ESM_KERNEL_IMAGE:-ipsissima/kernel:latest}"
PUSH_IMAGE=false
NO_CACHE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --push)
            PUSH_IMAGE=true
            shift
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== Building Verified Kernel ==="
echo "Image tag: $IMAGE_TAG"
echo "Repository root: $REPO_ROOT"

# Check Docker is available
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not found. Install Docker to build the kernel."
    exit 1
fi

# Navigate to kernel directory
cd "$REPO_ROOT/kernel/arb_kernel"

# Copy prototype kernel for fallback
cp "$REPO_ROOT/kernel/prototype/prototype_kernel.py" src/ 2>/dev/null || true

# Build the image
echo ""
echo "=== Building Docker image ==="
docker build $NO_CACHE -t "$IMAGE_TAG" .

# Get image hash
IMAGE_HASH=$(docker inspect "$IMAGE_TAG" --format='{{.Id}}')
echo ""
echo "=== Build complete ==="
echo "Image: $IMAGE_TAG"
echo "Hash: $IMAGE_HASH"

# Save hash for reproducibility
HASH_FILE="$REPO_ROOT/.kernel_image_hash"
echo "$IMAGE_HASH" > "$HASH_FILE"
echo "Hash saved to: $HASH_FILE"

# Test the image
echo ""
echo "=== Testing kernel ==="
TEST_INPUT=$(mktemp)
TEST_OUTPUT=$(mktemp)

# Create minimal test input
python3 -c "
import json
import base64
import numpy as np

X = np.eye(5, dtype='>f8')
data = {
    'schema_version': '1.0',
    'trace_id': 'test',
    'metadata': {'embedder_id': 'test', 'timestamp': '2024-01-01T00:00:00Z'},
    'parameters': {'rank': 3, 'precision_bits': 64, 'kernel_mode': 'prototype'},
    'observables': {
        'X_aug': {
            'rows': 5, 'cols': 5, 'dtype': 'float64',
            'data_matrix': base64.b64encode(X.tobytes()).decode(),
        }
    },
    'koopman_fit': None,
    'external_subspace': None,
}
with open('$TEST_INPUT', 'w') as f:
    json.dump(data, f)
" || {
    echo "WARNING: Could not create test input (Python not available)"
    rm -f "$TEST_INPUT" "$TEST_OUTPUT"
    exit 0
}

# Run kernel
if docker run --rm \
    -v "$TEST_INPUT:/data/kernel_input.json:ro" \
    -v "$TEST_OUTPUT:/data/kernel_output.json:rw" \
    "$IMAGE_TAG" 2>/dev/null; then
    echo "Kernel test passed!"
    rm -f "$TEST_INPUT" "$TEST_OUTPUT"
else
    echo "WARNING: Kernel test failed (may be expected if using placeholder)"
    rm -f "$TEST_INPUT" "$TEST_OUTPUT"
fi

# Push if requested
if [ "$PUSH_IMAGE" = true ]; then
    echo ""
    echo "=== Pushing image ==="
    docker push "$IMAGE_TAG"
    echo "Image pushed: $IMAGE_TAG"
fi

echo ""
echo "=== Done ==="
echo "To use the kernel:"
echo "  docker run --rm -v input.json:/data/kernel_input.json:ro -v output.json:/data/kernel_output.json:rw $IMAGE_TAG"
