# Profiling Guide

## Quick Start

### Chrome Trace (recommended first step)

```bash
make profile-chrome MODEL_DIR=path/to/model
# Opens trace.json in chrome://tracing or https://ui.perfetto.dev
```

This produces a hierarchical trace of every pipeline stage:

```
synthesize
├── prefill
├── generate_frames
│   ├── code_predictor (per frame)
│   ├── talker_step (per frame)
│   └── sampling (per frame)
│       ├── top_k
│       └── top_p
└── decode
```

### Flamegraph

```bash
make profile-flamegraph MODEL_DIR=path/to/model
# Produces flamegraph.svg
```

### Nsight Systems (CUDA)

```bash
make profile-nsys MODEL_DIR=path/to/model
# Produces nsys_report.nsys-rep — open with nsys-ui
```

In the Nsight timeline:

- **Kernel gaps** = CPU is the bottleneck (likely sampling or `to_vec1` syncs)
- **Continuous kernels** = GPU is saturated (good)
- Correlate with tracing span names in the NVTX row

## Per-Stage Timing

The `e2e_bench` binary reports a stage breakdown when not using `--streaming`:

```
Label     Words  Wall (ms)  Audio (s)      RTF    Tok/s  Mem (MB)    Prefill   Generate     Decode
----
short        13      450.2       1.28    0.352     34.2      1800   12ms (3%)  410ms (91%)  28ms (6%)
```

Programmatic access:

```rust
let (audio, timing) = model.synthesize_with_timing(text, speaker, lang, None)?;
println!("Prefill: {:.1}ms", timing.prefill_ms);
```

## Adding New Spans

Convention:

- Gate with `#[cfg(feature = "profiling")]`
- Use `tracing::info_span!("snake_case_name")` for stage-level spans
- Use `tracing::trace!(target: "gpu_sync", ...)` to mark GPU→CPU sync points
- Per-frame spans include `frame = frame_idx` as a field

Example:

```rust
#[cfg(feature = "profiling")]
let _span = tracing::info_span!("my_new_stage").entered();

// ... work ...

#[cfg(feature = "profiling")]
drop(_span);
```

## GPU Sync Point Optimization

Every `to_vec1()` call forces a GPU→CPU synchronization. List them with:

```bash
make audit-gpu-syncs
```

After optimization, only 1 unavoidable sync per frame remains (reading the
sampled semantic token for the next iteration's embedding lookup).

Previously eliminated:

1. **Code predictor**: Batched argmax — 15 of 16 per-frame `to_vec1` calls removed
1. **Sampling**: GPU-side top-k/top-p filtering — no logit vector transfer
1. **Token suppression**: GPU-native ops with cached suppression mask

## Overhead

When `profiling` is **not** enabled:

- Zero runtime overhead — all spans are `#[cfg]`-gated
- `tracing-chrome` is not compiled in (optional dependency)
- `SynthesisTiming` uses only `std::time::Instant`, no tracing dependency

When `profiling` **is** enabled:

- ~1-3% overhead from span enter/exit (negligible vs. model compute)
- `trace.json` grows ~1KB per generated frame
