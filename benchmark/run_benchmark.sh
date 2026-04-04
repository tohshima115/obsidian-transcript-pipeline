#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Building ARM64 Docker image (QEMU emulation)..."
echo "This may take 10-20 minutes on first run (downloading ARM packages)."
echo ""

docker buildx build \
    --platform linux/arm64 \
    -t transcripts-bench-arm \
    -f "$SCRIPT_DIR/Dockerfile.arm" \
    "$PROJECT_DIR"

echo ""
echo "Running benchmark..."
echo ""

docker run \
    --platform linux/arm64 \
    --rm \
    transcripts-bench-arm \
    python benchmark/benchmark_arm.py
