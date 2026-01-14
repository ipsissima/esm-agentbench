#!/bin/bash
# Build script that handles kernel security restrictions in cloud shell environments
# Addresses "unshare: operation not permitted" errors in Docker-in-Docker setups
set -e

IMAGE_NAME="${IMAGE_NAME:-esm-agentbench:ci}"
BASE_IMAGE="python:3.11-slim"

echo "=== ESM AgentBench CI Image Builder ==="
echo "Target image: $IMAGE_NAME"
echo ""

# Function to try normal docker build
try_normal_build() {
    echo "[1] Attempting normal docker build..."
    if docker build -t "$IMAGE_NAME" . 2>&1; then
        echo "✓ Normal build succeeded"
        return 0
    else
        echo "✗ Normal build failed"
        return 1
    fi
}

# Function to use the commit-based workaround
# This bypasses unshare/namespace operations by manually building in a running container
commit_workaround() {
    echo ""
    echo "[2] Using commit-based workaround for restricted environment..."
    echo "    (This bypasses kernel unshare restrictions)"
    echo ""

    BUILDER_NAME="esm-builder-$$"

    # Cleanup any previous builder
    docker rm -f "$BUILDER_NAME" 2>/dev/null || true

    echo "  → Starting base container..."
    docker run -d --name "$BUILDER_NAME" "$BASE_IMAGE" sleep 3600

    echo "  → Installing dependencies..."
    docker exec "$BUILDER_NAME" pip install --no-cache-dir --upgrade pip
    docker exec "$BUILDER_NAME" pip install --no-cache-dir \
        "huggingface-hub==0.16.4" "sentence-transformers==2.3.1" "transformers==4.35.2" \
        flask numpy scikit-learn joblib pytest tomli \
        gunicorn==20.1.0 matplotlib==3.8.1 \
        pandas==2.2.3 pytest-asyncio PyYAML 'pydantic>=1.10'

    echo "  → Pre-downloading sentence-transformers model..."
    docker exec "$BUILDER_NAME" python -c \
        "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

    echo "  → Creating app directory..."
    docker exec "$BUILDER_NAME" mkdir -p /app

    echo "  → Copying application files..."
    docker cp . "$BUILDER_NAME":/app/

    echo "  → Setting up user and permissions..."
    docker exec "$BUILDER_NAME" bash -c "useradd --create-home --shell /bin/bash appuser && \
        mkdir -p /app/demo_traces && chown -R appuser:appuser /app"

    echo "  → Committing container to image..."
    docker commit \
        --change 'WORKDIR /app' \
        --change 'USER appuser' \
        --change 'EXPOSE 8080' \
        --change 'CMD ["python", "-m", "esmassessor.green_server", "--host", "0.0.0.0", "--port", "8080", "--serve-only"]' \
        "$BUILDER_NAME" "$IMAGE_NAME"

    echo "  → Cleaning up builder container..."
    docker rm -f "$BUILDER_NAME"

    echo "✓ Commit-based build succeeded"
    return 0
}

# Check if docker is available
if ! command -v docker &> /dev/null; then
    echo "ERROR: docker command not found"
    exit 1
fi

# Try normal build first, fall back to workaround
if try_normal_build; then
    BUILD_METHOD="normal"
else
    # Check if it's the kernel security error
    if docker build -t "$IMAGE_NAME" . 2>&1 | grep -q "unshare: operation not permitted"; then
        echo ""
        echo "Detected kernel security restriction (unshare not permitted)"
        commit_workaround
        BUILD_METHOD="commit-workaround"
    else
        echo ""
        echo "Build failed for unknown reason. Trying commit workaround anyway..."
        commit_workaround
        BUILD_METHOD="commit-workaround"
    fi
fi

echo ""
echo "=== Build Complete ==="
echo "Image: $IMAGE_NAME"
echo "Method: $BUILD_METHOD"
echo ""
echo "To validate the image:"
echo "  docker run --rm $IMAGE_NAME python tools/validate_real_traces.py"
echo ""
echo "To run the server:"
echo "  docker run -p 8080:8080 $IMAGE_NAME"
