# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test

```bash
cargo build                                    # CPU debug
cargo build --release --features cuda,cli      # CUDA release with CLI
cargo test --lib                               # Unit tests (no model weights needed)
cargo test --test integration                  # Integration tests (no weights)
cargo test --lib -- generation::sampling       # Single test module
cargo clippy --features cli,hub -- -D warnings  # Lint (matches CI)
cargo fmt -- --check                           # Format check
cargo bench                                    # Criterion micro-benchmarks (no weights)
```

Python scripts in `scripts/` are linted with:

```bash
uvx ruff format --check scripts/
uvx ruff check scripts/
```

Pre-commit (runs both Rust and Python checks):

```bash
make pre-commit
```

## Profiling & Benchmarks

Model weights required. Run inside Docker for CUDA:

```bash
make profile-chrome MODEL_DIR=test_data/models/1.7B-CustomVoice
make profile-flamegraph MODEL_DIR=test_data/models/1.7B-CustomVoice
make audit-gpu-syncs
```

E2E benchmarks:

```bash
cargo run --release --features cuda,cli --bin e2e_bench -- \
  --model-dir test_data/models/1.7B-CustomVoice --iterations 3 --warmup 2 --streaming
```

## Architecture

Three-stage TTS pipeline, all in `src/`:

1. **TalkerModel** (`models/talker.rs`) — 28-layer transformer generating semantic tokens from text. Uses MRoPE, KV caching. 0.6B models: hidden=1024, 1.7B models: hidden=2048.

1. **CodePredictor** (`models/code_predictor.rs`) — 5-layer transformer generating 15 acoustic codes per semantic token. Always hidden=1024; 1.7B models use `small_to_mtp_projection` to bridge from talker's 2048-dim space. Called every frame during generation.

1. **Decoder12Hz** (`models/codec/decoder_12hz.rs`) — ConvNeXt + transposed convolution decoder converting 16-codebook codes to 24kHz audio. Always F32.

The generation loop (`lib.rs::generate_codes`) ties them together:

```
For each frame:
  1. CodePredictor generates 15 acoustic codes from last_hidden + semantic embedding
  2. All 16 codes (semantic + 15 acoustic) are embedded and summed (residual VQ)
  3. Trailing text embedding fused in (or tts_pad after text exhausted)
  4. Talker forward step → next hidden state + logits
  5. Sample next semantic token from logits
```

## Key Types

- `Qwen3TTS` (`lib.rs`) — main facade, owns all model components
- `StreamingSession` (`lib.rs`) — iterator yielding audio chunks
- `VoiceClonePrompt` (`lib.rs`) — speaker embedding + optional ICL data
- `SynthesisOptions` / `GenerationConfig` — hyperparameters
- `AudioBuffer` (`audio/io.rs`) — PCM samples + sample_rate
- `ModelType` (`models/config.rs`) — enum: Base, CustomVoice, VoiceDesign

## Model Variants

Five variants, auto-detected from `config.json`:

- **Base** (0.6B, 1.7B): Voice cloning via ECAPA-TDNN speaker encoder. ICL mode uses speech encoder + reference text.
- **CustomVoice** (0.6B, 1.7B): 9 preset speakers (Ryan, Serena, etc.) via discrete speaker token IDs.
- **VoiceDesign** (1.7B only): Text-described voices via instruct prompt with ChatML framing.

## Feature Flags

| Feature | Effect |
|---------|--------|
| `cpu` (default) | CPU inference |
| `cuda` | NVIDIA GPU, BF16 compute for talker/code_predictor |
| `metal` | Apple Silicon GPU |
| `flash-attn` | Flash Attention 2 (implies cuda) |
| `mkl` / `accelerate` | BLAS acceleration |
| `cli` | CLI binaries (generate_audio, e2e_bench) |
| `hub` | HuggingFace Hub downloads |
| `profiling` | tracing-chrome spans (zero overhead when disabled) |

## Codec Token IDs

Generation uses codec vocabulary (0–3071), not text vocabulary:

- EOS: 2150 (generation stops here)
- BOS: 2149, PAD: 2148
- Speakers: Ryan=3061, Serena=3066, etc.
- Languages: English=2050, Chinese=2055, etc.

## Conventions

- Profiling spans: `#[cfg(feature = "profiling")]` gated, `info_span!("snake_case")`
- GPU sync points: `tracing::trace!(target: "gpu_sync", ...)` markers
- Compute dtype: BF16 on CUDA/Metal, F32 on CPU (see `compute_dtype_for_device`)
- Decoder and speaker encoder always F32 regardless of device
- Tests don't require model weights — use synthetic tensors or mock VarBuilders
