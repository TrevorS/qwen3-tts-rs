# Benchmarks

Performance measurements for `qwen3-tts-rs` inference across CPU and GPU.

All results use default generation parameters
(temperature=0.9, top_k=50, top_p=0.9, repetition_penalty=1.05, seed=42).
2 warmup runs, 3 timed iterations.

## Test Hardware

| | Spec |
|---|---|
| **Platform** | NVIDIA DGX Spark |
| **CPU** | ARM Cortex-X925 + Cortex-A725, 20 cores |
| **GPU** | NVIDIA GB10 (Blackwell) |
| **RAM** | 120 GB unified |
| **OS** | Linux 6.14 (aarch64) |
| **CUDA** | 13.0, Driver 580.95 |

## Test Corpus

| Label | Words | Text |
|-------|------:|------|
| Short | 13 | "The quick brown fox jumps over the lazy dog near the river bank." |
| Medium | 53 | "In a quiet village nestled between rolling hills and dense forests, there lived an old clockmaker who spent his days repairing timepieces from centuries past. His workshop, filled with the gentle ticking of a hundred clocks, was a place where time itself seemed to slow down and the outside world faded into silence." |
| Long | 115 | "The development of artificial intelligence has been one of the most transformative technological advances of the twenty-first century. From natural language processing to computer vision, machine learning models have achieved remarkable performance across a wide range of tasks that were once considered the exclusive domain of human intelligence. Speech synthesis, in particular, has seen dramatic improvements with the introduction of neural network architectures that can generate high-fidelity audio from text input. These systems learn complex patterns of prosody, intonation, and rhythm from large datasets of recorded speech, producing output that is increasingly difficult to distinguish from natural human speech. The implications of this technology extend across many fields, including accessibility, entertainment, education, and human-computer interaction." |

## End-to-End Synthesis

Real-time factor (RTF) = wall-clock time / audio duration. **Lower is better; < 1.0 means faster than real-time.**

Each cell shows the average of 3 timed iterations after 2 warmup runs, executed in isolation (no concurrent GPU workloads).

### Non-Streaming (batch synthesis)

Uses `synthesize_with_timing` — the optimized `generate_codes` path with GPU-side
penalty mask and deferred acoustic codes transfer.

#### 0.6B Base — CUDA (BF16)

| Text | Words | Wall Clock | Audio Duration | RTF | Tok/s | Memory | Prefill | Generate | Decode |
|------|-------|------------|----------------|-----|-------|--------|---------|----------|--------|
| Short | 13 | 1.82 sec | 3.76 sec | **0.49** | 25.8 | 756 MB | 12ms (1%) | 1671ms (92%) | 140ms (8%) |
| Medium | 53 | 8.19 sec | 17.04 sec | **0.48** | 26.0 | 761 MB | 12ms (0%) | 7672ms (94%) | 504ms (6%) |
| Long | 115 | 23.02 sec | 45.68 sec | **0.50** | 24.8 | 767 MB | 12ms (0%) | 21622ms (94%) | 1384ms (6%) |

#### 1.7B Base — CUDA (BF16)

| Text | Words | Wall Clock | Audio Duration | RTF | Tok/s | Memory | Prefill | Generate | Decode |
|------|-------|------------|----------------|-----|-------|--------|---------|----------|--------|
| Short | 13 | 2.22 sec | 3.44 sec | **0.64** | 19.4 | 756 MB | 21ms (1%) | 2065ms (93%) | 129ms (6%) |
| Medium | 53 | 11.22 sec | 17.60 sec | **0.64** | 19.6 | 761 MB | 22ms (0%) | 10672ms (95%) | 521ms (5%) |
| Long | 115 | 29.82 sec | 45.68 sec | **0.65** | 19.2 | 767 MB | 22ms (0%) | 28409ms (95%) | 1382ms (5%) |

#### 1.7B CustomVoice — CUDA (BF16)

| Text | Words | Wall Clock | Audio Duration | RTF | Tok/s | Memory | Prefill | Generate | Decode |
|------|-------|------------|----------------|-----|-------|--------|---------|----------|--------|
| Short | 13 | 3.02 sec | 4.72 sec | **0.64** | 19.6 | 756 MB | 22ms (1%) | 2834ms (94%) | 161ms (5%) |
| Medium | 53 | 20.06 sec | 31.12 sec | **0.64** | 19.4 | 763 MB | 21ms (0%) | 19094ms (95%) | 945ms (5%) |
| Long | 115 | 45.60 sec | 68.00 sec | **0.67** | 18.6 | 772 MB | 22ms (0%) | 43535ms (95%) | 2040ms (4%) |

#### 1.7B VoiceDesign — CUDA (BF16)

| Text | Words | Wall Clock | Audio Duration | RTF | Tok/s | Memory | Prefill | Generate | Decode |
|------|-------|------------|----------------|-----|-------|--------|---------|----------|--------|
| Short | 13 | 3.13 sec | 4.88 sec | **0.64** | 19.5 | 756 MB | 22ms (1%) | 2938ms (94%) | 165ms (5%) |
| Medium | 53 | 13.52 sec | 21.12 sec | **0.64** | 19.5 | 761 MB | 22ms (0%) | 12867ms (95%) | 626ms (5%) |
| Long | 115 | 42.14 sec | 62.96 sec | **0.67** | 18.7 | 770 MB | 23ms (0%) | 40215ms (95%) | 1896ms (4%) |

### Streaming (with TTFA)

Uses `synthesize_streaming` — yields audio chunks incrementally. Both paths now
use GPU-side penalty mask. Streaming is ~8-12% slower than non-streaming due to
incremental decode overhead and per-frame `to_vec1` for the frame buffer.

#### 0.6B Base — CUDA (BF16)

| Text | Words | Wall Clock | Audio Duration | RTF | TTFA | Tok/s | Memory |
|------|-------|------------|----------------|-----|------|-------|--------|
| Short | 13 | 2.05 sec | 3.76 sec | **0.55** | 443 ms | 22.9 | 814 MB |
| Medium | 53 | 9.38 sec | 17.04 sec | **0.55** | 444 ms | 22.7 | 817 MB |
| Long | 115 | 26.01 sec | 45.68 sec | **0.57** | 445 ms | 22.0 | 820 MB |

#### 1.7B Base — CUDA (BF16)

| Text | Words | Wall Clock | Audio Duration | RTF | TTFA | Tok/s | Memory |
|------|-------|------------|----------------|-----|------|-------|--------|
| Short | 13 | 2.45 sec | 3.44 sec | **0.71** | 576 ms | 17.6 | 762 MB |
| Medium | 53 | 12.37 sec | 17.60 sec | **0.70** | 579 ms | 17.8 | 765 MB |
| Long | 115 | 32.94 sec | 45.68 sec | **0.72** | 576 ms | 17.3 | 768 MB |

#### 1.7B CustomVoice — CUDA (BF16)

| Text | Words | Wall Clock | Audio Duration | RTF | TTFA | Tok/s | Memory |
|------|-------|------------|----------------|-----|------|-------|--------|
| Short | 13 | 3.34 sec | 4.72 sec | **0.71** | 582 ms | 17.7 | 762 MB |
| Medium | 53 | 22.25 sec | 31.12 sec | **0.72** | 581 ms | 17.5 | 767 MB |
| Long | 115 | 50.52 sec | 68.00 sec | **0.74** | 585 ms | 16.8 | 773 MB |

#### 1.7B VoiceDesign — CUDA (BF16)

| Text | Words | Wall Clock | Audio Duration | RTF | TTFA | Tok/s | Memory |
|------|-------|------------|----------------|-----|------|-------|--------|
| Short | 13 | 3.50 sec | 4.88 sec | **0.72** | 584 ms | 17.4 | 762 MB |
| Medium | 53 | 15.04 sec | 21.12 sec | **0.71** | 582 ms | 17.6 | 765 MB |
| Long | 115 | 46.46 sec | 62.96 sec | **0.74** | 582 ms | 16.9 | 771 MB |

### CPU (F32, no MKL/BLAS)

| Text | Words | Frames | Wall Clock | Audio Duration | RTF | Tok/s | Memory |
|------|-------|--------|------------|----------------|-----|-------|--------|
| Short | 13 | 47 | 20.28 sec | 3.76 sec | 5.39 | 2.3 | 9.1 GB |
| Medium | 53 | 379 | 182.22 sec | 30.32 sec | 6.01 | 2.1 | 9.1 GB |
| Long | 115 | 703 | 364.17 sec | 56.24 sec | 6.48 | 1.9 | 9.1 GB |

### Summary

**Non-streaming** (batch synthesis — optimized `generate_codes` path):

| Metric | CPU (1.7B) | 0.6B Base | 1.7B Base | 1.7B CustomVoice | 1.7B VoiceDesign |
|--------|----------:|---------:|---------:|----------------:|----------------:|
| RTF (avg) | 5.96 | **0.49** | 0.64 | 0.65 | 0.65 |
| Tokens/sec | 2.1 | **25.5** | **19.4** | 19.2 | 19.2 |
| Peak memory | 9.1 GB | 767 MB | 767 MB | 772 MB | 770 MB |

**Streaming** (incremental chunks with TTFA):

| Metric | 0.6B Base | 1.7B Base | 1.7B CustomVoice | 1.7B VoiceDesign |
|--------|---------:|---------:|----------------:|----------------:|
| RTF (avg) | **0.55** | 0.71 | 0.72 | 0.72 |
| Tokens/sec | **22.5** | 17.6 | 17.3 | 17.3 |
| TTFA | **444 ms** | 577 ms | 583 ms | 583 ms |
| Peak memory | 820 MB | 768 MB | 773 MB | 771 MB |

**CUDA delivers faster-than-real-time synthesis** across all text lengths and
all model variants. Non-streaming is ~8-12% faster than streaming due to
deferred acoustic codes transfer in the `generate_codes` path; both paths
use the GPU-side penalty mask.

The 0.6B model is ~30% faster than 1.7B variants, at the cost of reduced
voice quality.

CPU is ~6x slower than real-time without BLAS acceleration — expected for
a 1.7B parameter model in F32. Enabling MKL (x86) or Accelerate (macOS)
would improve CPU performance significantly.

TTFA (time to first audio) via streaming is stable at ~580ms (1.7B) or ~444ms (0.6B)
regardless of input length, making the streaming API suitable for interactive use cases.

## Micro-Benchmarks

Component-level benchmarks run via [Criterion](https://bheisler.github.io/criterion.rs/book/).
No model weights required.

```
cargo bench
```

### Sampling (codec vocab = 3072)

| Operation | Time |
|-----------|-----:|
| Top-k sampling (k=50) | 53 µs |
| Top-p sampling (p=0.9) | 69 µs |
| Repetition penalty (500 prev tokens) | 834 ns |
| Token suppression | 684 ns |

Top-k with a large text vocab (32k) takes ~556 µs — the codec vocab (3k) keeps
per-step sampling overhead well under 100 µs.

### Audio Processing

| Operation | 0.5s | 2s | 10s |
|-----------|-----:|---:|----:|
| Mel spectrogram | 747 µs | 3.0 ms | 16.2 ms |
| Resample 12kHz → 24kHz | 691 µs | 1.4 ms | 5.4 ms |
| Resample 48kHz → 24kHz | 694 µs | 1.4 ms | 5.5 ms |

### Tensor Operations

| Operation | 1s (12 frames) | 5s (60 frames) | 20s (240 frames) |
|-----------|---------------:|----------------:|------------------:|
| codes_to_tensor | 162 ns | 420 ns | 1.4 µs |

## Reproducing

```bash
# Micro-benchmarks (no model weights needed)
cargo bench

# Single benchmark group
cargo bench -- sampling

# End-to-end (requires model weights)
cargo run --release --features cuda,cli --bin e2e_bench -- \
  --model-dir <path-to-model> --device cuda --iterations 3

# With streaming TTFA measurement and JSON export
cargo run --release --features cuda,cli --bin e2e_bench -- \
  --model-dir <path-to-model> --device cuda --streaming \
  --warmup 2 --json-output results.json

# Audio quality sanity check (optional)
python scripts/quality_check.py output.wav "expected transcription"
```

## Glossary

| Term | Definition |
|------|-----------|
| **RTF** | Real-time factor: wall-clock / audio duration. < 1.0 = faster than real-time. |
| **TTFA** | Time to first audio: latency until the first streaming chunk is available. |
| **Tok/s** | Semantic frames generated per second of wall-clock time. Each frame is one 12 Hz codec step (80ms of audio). |
