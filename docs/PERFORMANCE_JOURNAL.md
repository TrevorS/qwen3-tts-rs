# Performance Journal

## Baseline (Jan 2025, DGX Spark)

| Label | Words | Wall (ms) | Audio (s) | RTF | Tok/s | Prefill | Generate | Decode |
|-------|-------|-----------|-----------|-----|-------|---------|----------|--------|
| short | 13 | 5235 | 3.68 | 1.423 | 8.8 | 21ms (1%) | 2724ms (71%) | 1109ms (29%) |
| medium | 53 | 23786 | 34.00 | 0.700 | 17.9 | 20ms (0%) | 22694ms (95%) | 1057ms (4%) |
| long | 115 | 43797 | 60.96 | 0.718 | 17.4 | 19ms (0%) | 41861ms (96%) | 1886ms (4%) |

## Optimizations 1-3: GPU sync elimination (2026-01-31)

All three optimizations applied together:

1. Batch code_predictor argmax (15 fewer GPU→CPU syncs per frame)
1. GPU-side top-k/top-p filtering (no logit vector CPU transfer)
1. GPU-side repetition penalty & EOS suppression (on-device masking)

| Label | Words | Wall (ms) | ±stddev | Audio (s) | RTF | Tok/s | Mem (MB) | Prefill | Generate | Decode |
|-------|-------|-----------|---------|-----------|-----|-------|----------|---------|----------|--------|
| short | 13 | 2430 | 3 | 3.68 | 0.660 | 18.9 | 834 | 19ms (1%) | 2274ms (94%) | 136ms (6%) |
| medium | 53 | 22159 | 90 | 33.12 | 0.669 | 18.7 | 835 | 19ms (0%) | 21130ms (95%) | 1009ms (5%) |
| long | 115 | 41766 | 11 | 60.32 | 0.692 | 18.1 | 835 | 20ms (0%) | 39906ms (96%) | 1839ms (4%) |

## Optimization 4: Eliminate no-op causal masks (2026-01-31)

Skip creating all-zeros causal masks for single-token generation steps (both
talker and code_predictor). Also cached the token suppression mask to avoid
rebuilding it each frame.

| Label | Words | Wall (ms) | ±stddev | Audio (s) | RTF | Tok/s | Mem (MB) | Prefill | Generate | Decode |
|-------|-------|-----------|---------|-----------|-----|-------|----------|---------|----------|--------|
| short | 13 | 2412 | 7 | 3.68 | 0.655 | 19.1 | 835 | 19ms (1%) | 2258ms (94%) | 134ms (6%) |
| medium | 53 | 21935 | 94 | 33.12 | 0.662 | 18.9 | 836 | 20ms (0%) | 20903ms (95%) | 1010ms (5%) |
| long | 115 | 41320 | 59 | 60.32 | 0.685 | 18.2 | 847 | 20ms (0%) | 39476ms (96%) | 1823ms (4%) |

## Optimization 5: Pre-allocated KV cache + GPU-side token forwarding (2026-01-31)

Two changes:

1. **PreAllocKVCache with InplaceOp2**: Pre-allocate fixed-size KV buffers and
   write via `copy2d` (CUDA) instead of `Tensor::cat` per step. Eliminates all
   growing allocations during generation. CodePredictor: 17-slot buffers (5
   layers). TalkerModel: `max_new_tokens + 256` slots (28 layers).
2. **GPU-side semantic token forwarding**: Keep sampled token tensor on GPU for
   codec embedding lookup instead of roundtripping through CPU `u32` →
   `Tensor::new`.

| Label | Words | Wall (ms) | ±stddev | Audio (s) | RTF | Tok/s | Mem (MB) | Prefill | Generate | Decode |
|-------|-------|-----------|---------|-----------|-----|-------|----------|---------|----------|--------|
| short | 13 | 2393 | 3 | 3.68 | 0.650 | 19.2 | 833 | 21ms (1%) | 2236ms (93%) | 135ms (6%) |
| medium | 53 | 21735 | 15 | 33.12 | 0.656 | 19.0 | 835 | 22ms (0%) | 20709ms (95%) | 1003ms (5%) |
| long | 115 | 41097 | 31 | 60.32 | 0.681 | 18.3 | 845 | 22ms (0%) | 39252ms (96%) | 1822ms (4%) |

Marginal improvement (~1%) over Optimization 4. Memory dropped ~10 MB despite
pre-allocating buffers, likely from eliminating intermediate tensors in
`Tensor::cat`. At 95% of theoretical throughput, allocation overhead was not a
significant bottleneck.

## Optimization 6: GPU-side penalty mask + deferred acoustic codes transfer (2026-02-01)

Two changes to eliminate remaining per-frame GPU→CPU syncs:

1. **GPU-side repetition penalty mask**: Maintain a `[1, vocab]` mask tensor on GPU,
   updated incrementally via `slice_assign` after each sampled token. Replaces the
   old pattern of transferring all generated token IDs (growing each frame) to CPU
   to build the penalty mask, then uploading it back.
2. **Deferred acoustic codes transfer**: Accumulate frame codes as GPU tensors during
   generation, then do a single bulk `Tensor::stack` + `to_vec1` after the loop ends.
   Eliminates the per-frame `to_vec1` of 15 acoustic codes.

Net effect: 3 GPU→CPU syncs per frame → 1 (the unavoidable 4-byte semantic token
read for EOS check). Applied to both `generate_codes` (non-streaming) and
`StreamingSession` (streaming) paths.

1.7B CustomVoice (non-streaming):

| Label | Words | Wall (ms) | ±stddev | Audio (s) | RTF | Tok/s | Mem (MB) | Prefill | Generate | Decode |
|-------|-------|-----------|---------|-----------|-----|-------|----------|---------|----------|--------|
| short | 13 | 3020 | 1 | 4.72 | 0.640 | 19.5 | 756 | 22ms (1%) | 2837ms (94%) | 161ms (5%) |
| medium | 53 | 20051 | 9 | 31.12 | 0.644 | 19.4 | 763 | 22ms (0%) | 19088ms (95%) | 941ms (5%) |
| long | 115 | 45541 | 90 | 68.00 | 0.670 | 18.7 | 771 | 22ms (0%) | 43479ms (95%) | 2037ms (4%) |

All variants (non-streaming):

| Model | RTF (short) | RTF (med) | RTF (long) | Tok/s (avg) | Memory |
|-------|------------|-----------|-----------|------------|--------|
| 0.6B Base | **0.48** | **0.47** | **0.50** | 25.9 | 835 MB |
| 1.7B Base | 0.65 | 0.64 | 0.65 | 19.4 | 767 MB |
| 1.7B CustomVoice | 0.64 | 0.64 | 0.67 | 19.2 | 771 MB |
| 1.7B VoiceDesign | 0.64 | 0.64 | 0.66 | 19.3 | 770 MB |

~2-4% improvement over Optimization 5 (non-streaming). Note: audio durations
differ from previous runs because seed=42 generation is not bitwise identical
after the penalty mask change (penalty is applied in a different order —
mask-based vs token-list-based). RTF comparison against the same run's audio
duration remains valid.

## Summary

1.7B CustomVoice non-streaming, baseline → final (6 optimizations):

| Label | Baseline RTF | Final RTF | Speedup | Baseline tok/s | Final tok/s |
|-------|-------------|-----------|---------|---------------|-------------|
| short | 1.423 | 0.640 | **2.22x** | 8.8 | 19.5 |
| medium | 0.700 | 0.644 | **1.09x** | 17.9 | 19.4 |
| long | 0.718 | 0.670 | **1.07x** | 17.4 | 18.7 |

## Analysis: Theoretical Ceiling

Chrome trace analysis of per-frame timing (long sentence, 756 frames):

| Span | Avg (ms) | % of frame |
|------|----------|------------|
| code_predictor | 25.88 | 50% |
| sampling (incl. GPU sync) | 15.78 | 30% |
| talker_step (CPU launch) | 10.15 | 20% |
| **Total** | **51.81** | |

The "sampling" span absorbs the GPU sync cost for the preceding async talker_step.
Actual GPU compute per frame is ~51ms, dominated by model forward passes (talker
28 layers + code_predictor 5 layers × 15 autoregressive steps).

At ~52ms/frame, theoretical max is ~19.2 tok/s. We're at 18.7-19.5 tok/s
(**~97-100% of theoretical throughput**). The remaining gap is framework overhead.

### What won't help further

- **Flash attention**: Tested — no improvement for single-token KV-cache steps
  (batch=1, seq_len=1). Flash attention benefits long-sequence prefill, not generation.
- **Further GPU sync reduction**: Only 1 unavoidable sync per frame remains
  (sampling the semantic token for EOS check).

### Remaining opportunities (diminishing returns)

- **Quantization (INT8/INT4)**: Reduce memory bandwidth for matmuls.
- **Custom CUDA kernels**: Fused attention + MLP, fused embedding + projection.
- **Batched inference**: Process multiple utterances simultaneously.
