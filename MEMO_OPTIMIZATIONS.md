# Qwen3-TTS Rust Performance Optimizations

**Date:** 2025-02-25  
**Hardware:** NVIDIA RTX 3090 (24GB), CUDA 12.6  
**Model:** Qwen3-TTS-12Hz-1.7B-CustomVoice (3.6GB)

## Executive Summary

Achieved **2.1x speedup** over official Python implementation on identical hardware through systematic optimization of the inference pipeline.

| Implementation | RTF | Relative Speed |
|---------------|-----|----------------|
| Python Official | ~1.26 | 1.0x (baseline) |
| **Rust (optimized)** | **0.58-0.61** | **2.1x** |
| Python dffdeeq fork (RTX 5090) | ~0.1 | ~12x* |

*On faster hardware (RTX 5090 ~2x faster than RTX 3090)

---

## Optimizations Implemented

### 1. Flash Attention (5-17% improvement)

**File:** `src/models/transformer.rs`

Enabled via `--features flash-attn` compile flag. Uses `candle-flash-attn` for fused attention computation:
- Eliminates materialization of large attention matrices
- Handles GQA natively without repeat_kv
- Uses causal=true instead of explicit masks

```rust
#[cfg(feature = "flash-attn")]
let use_flash = q.device().is_cuda();
```

### 2. Fused RMSNorm + Residual (existing)

**File:** `src/models/fused_ops.rs`

Custom CUDA PTX kernel that combines residual addition with RMSNorm in a single kernel launch:
- Reduces from 2 kernel launches to 1
- Cuts memory bandwidth in half for this operation
- Falls back to sequential ops on CPU/Metal

```rust
pub fn forward_residual(&self, x: &Tensor, residual: &Tensor) -> Result<(Tensor, Tensor)>
```

### 3. Fused Gate + Up Projection (2% improvement)

**File:** `src/models/transformer.rs:389-448`

SwiGLU MLP originally used two separate linear projections:
```rust
// Before: 2 matmuls
let gate = self.gate_proj.forward(x)?;
let up = self.up_proj.forward(x)?;
```

Now fuses weights at load time and uses single matmul:
```rust
// After: 1 matmul + narrow
let combined = fused.forward(x)?; // [batch, seq, 2*intermediate]
let gate = combined.narrow(D::Minus1, 0, intermediate_size)?;
let up = combined.narrow(D::Minus1, intermediate_size, intermediate_size)?;
```

### 4. Pre-Allocated KV Caches

**File:** `src/models/kv_cache.rs`, `src/models/code_predictor.rs`

Code predictor uses `PreAllocKVCache` instead of `ConcatKVCache`:
- Pre-allocates tensor storage for max sequence length (17 tokens)
- Avoids per-step tensor reallocation
- Reset via `cache.reset()` instead of new allocation

```rust
PreAllocKVCache::new(1, num_heads, 17, head_dim, dtype, device)
```

### 5. Cached Prefill Mask

**File:** `src/models/code_predictor.rs:159`

The 2×2 causal mask for prefill is created once at model load:
```rust
let prefill_mask = super::transformer::create_causal_mask(2, 0, vb.device())?;
```

Previously created fresh on every frame generation.

### 6. Sliding Window Decoder (2-3% improvement)

**File:** `src/models/codec/decoder_12hz.rs:415-490`

Limits transformer attention context for long sequences:
```rust
pub fn decode_with_window(&self, codes: &Tensor, window_frames: usize) -> Result<Tensor>
```

Controlled via `SynthesisOptions.decode_window_frames`:
- 0 = full context (default, backward compatible)
- 80 = ~6.4s context window (matches Python fork)

### 7. Optimized Code Predictor

**File:** `src/models/code_predictor.rs:580-660`

`generate_acoustic_codes_optimized()` minimizes allocations:
- Uses cached prefill mask
- Pre-allocates output tensor once
- Flattens argmax results for efficient slice_assign
- No intermediate GPU→CPU syncs

### 8. Streaming Integration

**File:** `src/lib.rs:1671-1675`

Streaming session uses optimized code predictor:
```rust
let acoustic_codes_tensor = self.model.code_predictor.generate_acoustic_codes_optimized(
    &self.last_hidden,
    &semantic_embed,
    &mut self.cp_kv_caches,
)?;
```

---

## Benchmark Results

### Non-Streaming Performance

| Text | Words | Wall (ms) | RTF | Tok/s | Memory |
|------|-------|-----------|-----|-------|--------|
| short | 13 | 2423 | 0.606 | 20.6 | 634 MB |
| medium | 53 | 17020 | 0.578 | 21.6 | 640 MB |
| long | 115 | 36323 | 0.587 | 21.3 | 647 MB |

### Streaming Performance

| Metric | Value |
|--------|-------|
| RTF | 0.68-0.69 |
| **TTFA** | **~560ms** |
| Memory | ~640 MB |

### Stage Breakdown

| Stage | Time | % of Total |
|-------|------|------------|
| Prefill | 15-18ms | 0-1% |
| **Generation (Code Predictor)** | 35,469ms | **98%** |
| Decode (Audio Decoder) | 835ms | 2% |

---

## Remaining Optimization Opportunities

### 1. INT8 Quantization (high impact)
- Would reduce memory bandwidth by ~2x
- Code predictor is 98% of compute time
- Candle supports GGUF quantization formats

### 2. CUDA Graphs (high impact)
- Capture the 15-step autoregressive decode loop
- Eliminates kernel launch overhead
- Requires custom CUDA bindings (not exposed by candle-core)

### 3. Custom Fused Kernels (medium impact)
- Embedding + projection fusion
- Layer norm + linear fusion
- Argmax + slice_assign fusion

### 4. FP16 Compute (low impact on RTX 30xx)
- RTX 30xx has slower FP16 than BF16
- RTX 40xx+ would benefit more

---

## How to Build

```bash
# Standard build (no Flash Attention)
cargo build --release --features cuda,cli

# With Flash Attention (recommended, 45s compile)
cargo build --release --features cuda,cli,flash-attn

# Run benchmark
./target/release/e2e_bench --model-dir test_data/models/1.7B-CustomVoice --iterations 5
```

---

## References

- Python fork with 6x speedup: https://github.com/dffdeeq/Qwen3-TTS-streaming
- Official Qwen3-TTS: https://github.com/QwenLM/Qwen3-TTS
- Candle ML framework: https://github.com/huggingface/candle
