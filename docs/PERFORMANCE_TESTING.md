# Performance Testing Plan

This document outlines the strategy for benchmarking qwen3-tts in both CPU and GPU configurations.

## Goals

1. Catch performance regressions on every PR (CI-friendly, no model weights)
1. Measure real-world inference latency across devices (CPU, CUDA, Metal)
1. Profile each pipeline stage independently (Talker → CodePredictor → Decoder12Hz)
1. Track memory usage and streaming latency

______________________________________________________________________

## Tier 1: Micro-Benchmarks (criterion)

**No model weights required.** Runs in CI on every PR.

criterion 0.8 is already in `dev-dependencies`. We need a `benches/` directory with benchmark harnesses.

### Candidates

| Benchmark | What it measures | Input |
|-----------|-----------------|-------|
| `sampling` | Top-k, top-p, temperature, repetition penalty | Synthetic logit tensors |
| `codes_to_tensor` | Frame-code packing into tensor | Random `Vec<FrameCodes>` |
| `mel_spectrogram` | Mel computation from raw audio | Synthetic sine wave |
| `resample` | 12kHz → 24kHz resampling | Synthetic audio buffer |
| `token_suppression` | Suppress/unsuppress token masks | Synthetic logit tensors |

### File layout

```
benches/
  sampling.rs        — top_k, top_p, temperature scaling, rep penalty
  audio.rs           — mel spectrogram, resampling
  tensor_ops.rs      — codes_to_tensor, token suppression
```

### Implementation notes

- Each benchmark uses `criterion::Criterion` with `BenchmarkGroup` for parameterized sizes
- Sampling benchmarks: vary vocab size (1k, 32k, 152k) and top-k (50, 200)
- Audio benchmarks: vary duration (0.5s, 2s, 10s)
- All benchmarks create tensors on `Device::Cpu` — no weights needed
- Add `[[bench]]` entries to `Cargo.toml` with `harness = false`

### CI integration

```yaml
# .github/workflows/bench.yml
- name: Run benchmarks
  run: cargo bench --features cpu -- --output-format bencher
```

Use `criterion-compare` or GitHub Actions benchmark action to post regression comments on PRs.

______________________________________________________________________

## Tier 2: End-to-End Benchmarks (integration)

**Requires model weights.** Runs manually or on self-hosted runners with GPU access.

### What to measure

| Metric | Description |
|--------|-------------|
| **Time to first audio chunk** | Streaming latency from `synthesize_streaming()` to first `next()` yield |
| **Total synthesis wall-clock** | Full `synthesize()` for a reference sentence |
| **Per-stage breakdown** | Talker prefill, Talker decode loop, CodePredictor, Decoder12Hz |
| **Peak memory** | RSS or CUDA memory at peak |
| **Tokens/second** | Semantic token generation rate (Talker) |
| **Real-time factor** | Audio duration / wall-clock time |

### Reference inputs

Define a standard test corpus in `benches/fixtures/`:

```
fixtures/
  short.txt    — "Hello world."               (~1s audio)
  medium.txt   — A full sentence              (~5s audio)
  long.txt     — A full paragraph             (~20s audio)
```

### Device matrix

| Device | Features | Notes |
|--------|----------|-------|
| CPU (default) | `--features cpu` | Baseline, always available |
| CPU + MKL | `--features cpu,mkl` | Intel-optimized |
| CPU + Accelerate | `--features cpu,accelerate` | macOS-optimized |
| CUDA (bf16) | `--features cuda` | Primary GPU target |
| CUDA + Flash Attention | `--features cuda,flash-attn` | Best GPU perf |
| Metal | `--features metal` | macOS GPU |

### Implementation approach

Extend `scripts/test-variants.sh` or create a new `scripts/bench-e2e.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR to the Qwen3-TTS weights directory}"
DEVICE="${DEVICE:-cpu}"
FEATURES="${FEATURES:-cpu}"

TEXTS=(
  "Hello world."
  "The quick brown fox jumps over the lazy dog near the riverbank."
  # ... longer text
)

for text in "${TEXTS[@]}"; do
  echo "=== Benchmarking: ${text:0:40}... ==="
  hyperfine --warmup 1 --runs 3 \
    "cargo run --release --features ${FEATURES},cli --bin generate_audio -- \
      --model-dir ${MODEL_DIR} --device ${DEVICE} --text '${text}' --output /dev/null"
done
```

### Per-stage tracing

**Implemented.** The pipeline is instrumented with feature-gated tracing spans
(`#[cfg(feature = "profiling")]`) and a `SynthesisTiming` API for wall-clock
stage breakdowns. See [PROFILING.md](PROFILING.md) for usage details.

Key spans: `synthesize`, `prefill`, `generate_frames`, `code_predictor`,
`talker_step`, `sampling`, `top_k`, `top_p`, `decode`, `code_predictor_inner`.

GPU→CPU sync points are marked with `tracing::trace!(target: "gpu_sync", ...)`.

### E2E benchmarks

**Implemented.** The `e2e_bench` binary (`benches/e2e_bench.rs`, requires `cli` feature)
supports:

- Per-stage timing breakdown (prefill / generation / decode)
- Streaming TTFA measurement (`--streaming`)
- JSON output (`--json-output results.json`)
- Configurable warmup and iterations
- Memory tracking via CUDA APIs

Run all 4 variants sequentially:

```bash
for model in 0.6B-Base 1.7B-Base 1.7B-CustomVoice 1.7B-VoiceDesign; do
  cargo run --release --features cuda,cli --bin e2e_bench -- \
    --model-dir test_data/models/$model --iterations 3 --warmup 2 --streaming
done
```

### Memory profiling

- **CPU**: Use `/proc/self/status` (VmRSS) or `jemalloc-ctl` stats at checkpoints
- **CUDA**: Use `candle_core::cuda_backend::CudaDevice::mem_info()` if available, or `nvidia-smi` polling
- **Automated**: Add a `--profile-memory` flag to `generate_audio` that logs memory at each stage boundary

______________________________________________________________________

## Reporting

### Format

Benchmark results should be machine-readable (JSON) alongside human-readable output:

```json
{
  "commit": "abc1234",
  "device": "cuda",
  "features": "cuda,flash-attn",
  "results": [
    {
      "input": "short",
      "wall_clock_ms": 342,
      "time_to_first_chunk_ms": 180,
      "peak_memory_mb": 2048,
      "tokens_per_second": 45.2,
      "realtime_factor": 2.9,
      "stages": {
        "talker_ms": 150,
        "code_predictor_ms": 120,
        "decoder_12hz_ms": 72
      }
    }
  ]
}
```

### Historical tracking

Store results in `bench-results/` (gitignored) or push to a separate branch for trend visualization. Consider [Bencher](https://bencher.dev) or [GitHub Pages](https://github.com/benchmark-action/github-action-benchmark) for charts.

______________________________________________________________________

## Implementation Order

1. **Tier 1 micro-benchmarks** — `benches/sampling.rs`, `benches/audio.rs`, `benches/tensor_ops.rs`
1. **CI workflow** — `.github/workflows/bench.yml` with regression detection
1. **Tracing spans** — instrument pipeline stages in `lib.rs`
1. **E2E script** — `scripts/bench-e2e.sh` with hyperfine
1. **Memory profiling** — add checkpoint logging behind a feature flag or CLI arg
1. **Reporting** — JSON output + historical tracking setup
