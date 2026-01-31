# Dockerfile for qwen3-tts
#
# Uses the NGC PyTorch container which includes CUDA 13.0 toolkit,
# cuDNN, and NCCL — matching the validated build environment.
#
# Build:
#   docker build -t qwen3-tts .
#
# Run (GPU):
#   docker run --gpus all \
#     -v /path/to/models:/models \
#     -v /path/to/output:/output \
#     qwen3-tts \
#       --model-dir /models/0.6b-customvoice \
#       --speaker ryan \
#       --text "Hello world, this is a test." \
#       --device cuda \
#       --output /output/hello.wav
#
# Run (CPU):
#   docker run \
#     -v /path/to/models:/models \
#     -v /path/to/output:/output \
#     qwen3-tts \
#       --model-dir /models/0.6b-customvoice \
#       --speaker ryan \
#       --text "Hello world, this is a test." \
#       --output /output/hello.wav
#
# CPU-only build (smaller, no CUDA):
#   docker build --build-arg FEATURES=cli --build-arg BASE=ubuntu:22.04 -t qwen3-tts-cpu .

ARG BASE=nvcr.io/nvidia/pytorch:25.11-py3

FROM ${BASE}

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies (curl + build-essential may already exist in NGC image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Build features
ARG FEATURES=flash-attn,cli

# CUDA compute capability for bindgen_cuda (nvidia-smi unavailable during build).
# 90 = SM_90 (Hopper GH200). Override with --build-arg CUDA_COMPUTE_CAP=89 etc.
ARG CUDA_COMPUTE_CAP=90
ENV CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP}

WORKDIR /build
COPY . .
RUN cargo build --release --features "${FEATURES}"

# Install binary to system path
RUN cp target/release/generate_audio /usr/local/bin/ \
    && rm -rf target/

# Install uv + Whisper for audio intelligibility testing
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
RUN uv pip install --system --no-cache openai-whisper scipy flash-attn

# Copy example audio and whisper transcribe script
RUN mkdir -p /examples/data \
    && cp examples/data/clone_2.wav /examples/data/
COPY scripts/transcribe.py /usr/local/bin/transcribe.py

RUN mkdir -p /output
WORKDIR /output

ENTRYPOINT ["generate_audio"]
CMD ["--help"]
