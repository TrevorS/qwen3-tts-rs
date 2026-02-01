#!/usr/bin/env bash
# Quick-compile and run a specific kernel unit test in Docker.
#
# Usage: ./scripts/test-kernel.sh <test_filter>
#   e.g. ./scripts/test-kernel.sh fused_rmsnorm
#        ./scripts/test-kernel.sh sampling
#
# Requires: Docker + NVIDIA Container Toolkit

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <test_filter>"
    echo "  e.g. $0 fused_rmsnorm"
    exit 1
fi

TEST_FILTER="$1"
CONTAINER_NAME="qwen3-tts-kernel-test-$$"
BASE_IMAGE="nvcr.io/nvidia/pytorch:25.11-py3"

cleanup() {
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Testing kernel: $TEST_FILTER ==="

# Start container with GPU
docker run -d --name "$CONTAINER_NAME" --gpus all \
    "$BASE_IMAGE" sleep infinity

# Copy source
docker cp . "$CONTAINER_NAME":/workspace/project

# Install Rust + run test
docker exec "$CONTAINER_NAME" bash -c "
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source ~/.cargo/env
    cd /workspace/project
    echo '--- Building ---'
    cargo test --lib --features cuda -- $TEST_FILTER --no-run 2>&1 | tail -5
    echo '--- Running tests matching: $TEST_FILTER ---'
    cargo test --lib --features cuda -- $TEST_FILTER --nocapture
"
