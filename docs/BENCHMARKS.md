# Benchmarks

Performance measurements for `qwen3-tts-rs` inference across CPU and GPU.

All results use the **1.7B CustomVoice** model with default generation parameters
(temperature=0.9, top_k=50, top_p=0.9, repetition_penalty=1.05, seed=42).

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

Each cell shows the average of 3 timed iterations after 2 warmup runs, executed in isolation (no concurrent workloads).

### CUDA (BF16)

| Text | Words | Frames | Wall Clock | Audio Duration | RTF | TTFA | Tok/s | Memory |
|------|-------|--------|------------|----------------|-----|------|-------|--------|
| Short | 13 | 46 | 2.86 sec | 3.68 sec | **0.78** | 627 ms | 16.1 | 761 MB |
| Medium | 53 | 425 | 26.62 sec | 34.00 sec | **0.78** | 630 ms | 16.0 | 765 MB |
| Long | 115 | 762 | 49.20 sec | 60.96 sec | **0.81** | 632 ms | 15.5 | 768 MB |

### CPU (F32, no MKL/BLAS)

| Text | Words | Frames | Wall Clock | Audio Duration | RTF | Tok/s | Memory |
|------|-------|--------|------------|----------------|-----|-------|--------|
| Short | 13 | 47 | 20.28 sec | 3.76 sec | 5.39 | 2.3 | 9.1 GB |
| Medium | 53 | 379 | 182.22 sec | 30.32 sec | 6.01 | 2.1 | 9.1 GB |
| Long | 115 | 703 | 364.17 sec | 56.24 sec | 6.48 | 1.9 | 9.1 GB |

### Summary

| Metric | CPU | CUDA | Speedup |
|--------|----:|-----:|--------:|
| RTF (avg) | 5.96 | 0.79 | **7.5x** |
| Tokens/sec | 2.1 | 15.9 | **7.6x** |
| Time to first audio | — | 630ms | — |
| Peak memory | 9.1 GB | 765 MB | 12x less |

**CUDA delivers faster-than-real-time synthesis** across all text lengths.
CPU is ~6x slower than real-time without BLAS acceleration — expected for
a 1.7B parameter model in F32. Enabling MKL (x86) or Accelerate (macOS)
would improve CPU performance significantly.

TTFA (time to first audio) via streaming is stable at ~630ms regardless of
input length, making the streaming API suitable for interactive use cases.

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
cargo run --release --features cli --bin e2e_bench -- \
  --model-dir <path-to-model> --device auto --iterations 3

# With streaming TTFA measurement and JSON export
cargo run --release --features cli --bin e2e_bench -- \
  --model-dir <path-to-model> --device cuda --streaming \
  --json-output results.json

# Audio quality sanity check (optional)
python scripts/quality_check.py output.wav "expected transcription"
```

## Glossary

| Term | Definition |
|------|-----------|
| **RTF** | Real-time factor: wall-clock / audio duration. < 1.0 = faster than real-time. |
| **TTFA** | Time to first audio: latency until the first streaming chunk is available. |
| **Tok/s** | Semantic frames generated per second of wall-clock time. Each frame is one 12 Hz codec step (80ms of audio). |
