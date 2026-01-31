#!/bin/bash
# Build qwen3-tts Docker image with GPU auto-detection
#
# Usage: ./build-docker.sh [image-name] [features]
#   image-name: Docker image name (default: qwen3-tts)
#   features:   Cargo features (default: flash-attn,cli)
#
# Requires: NVIDIA GPU + nvidia-container-toolkit

set -e

IMAGE_NAME="${1:-qwen3-tts}"
FEATURES="${2:-flash-attn,cli}"
CONTAINER_NAME="qwen3-tts-builder-$$"
BASE_IMAGE="nvcr.io/nvidia/pytorch:25.11-py3"

echo "=== Building $IMAGE_NAME with features: $FEATURES ==="

# Cleanup on exit
cleanup() {
    echo "Cleaning up..."
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
}
trap cleanup EXIT

# Start container with GPU access
echo "Starting build container with GPU access..."
docker run -d --name "$CONTAINER_NAME" --gpus all \
    "$BASE_IMAGE" sleep infinity

# Copy source into container (avoids mount issues)
echo "Copying source..."
docker cp . "$CONTAINER_NAME":/workspace/project

# Install Rust
echo "Installing Rust toolchain..."
docker exec "$CONTAINER_NAME" bash -c \
    'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y'

# Build
echo "Building with features: $FEATURES"
docker exec "$CONTAINER_NAME" bash -c \
    "source ~/.cargo/env && cd /workspace/project && cargo build --release --features '$FEATURES'"

# Install binary and Python tools
echo "Installing binary..."
docker exec "$CONTAINER_NAME" bash -c \
    'cp /workspace/project/target/release/generate_audio /usr/local/bin/ && \
     mkdir -p /examples/data && \
     cp /workspace/project/examples/data/clone_2.wav /examples/data/ 2>/dev/null || true && \
     cp /workspace/project/scripts/transcribe.py /usr/local/bin/ 2>/dev/null || true && \
     mkdir -p /output && \
     rm -rf /workspace/project'

echo "Installing Python tools (whisper, flash-attn)..."
docker exec "$CONTAINER_NAME" bash -c \
    'pip install --no-cache-dir openai-whisper scipy flash-attn'

# Commit to new image
echo "Creating image..."
docker commit \
    --change 'WORKDIR /output' \
    --change 'ENTRYPOINT ["generate_audio"]' \
    --change 'CMD ["--help"]' \
    "$CONTAINER_NAME" "$IMAGE_NAME"

echo ""
echo "=== Done! ==="
echo "Image: $IMAGE_NAME"
echo ""
echo "Usage:"
echo "  docker run --gpus all -v /path/to/models:/models -v /path/to/output:/output $IMAGE_NAME \\"
echo "    --model-dir /models/1.7b-customvoice --speaker ryan --text \"Hello\" --device cuda --output /output/out.wav"
