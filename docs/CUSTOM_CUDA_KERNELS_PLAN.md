# Custom CUDA Kernels Plan

## Current State

- **19.2 tok/s** (short), **18.3 tok/s** (long) on DGX A100
- ~95% of theoretical throughput given current kernel launch pattern
- ~625-740 CUDA kernel launches per talker decode step (28 layers × ~24 kernels/layer)
- Decode is memory-bandwidth bound at batch=1
- Target: **25-27 tok/s** (~40% improvement)

## Phase 1: Fused Residual + RMSNorm (estimated 15-25% speedup)

**Why first:** Executes 33 times per frame (28 talker + 5 code predictor layers). Currently
3 separate kernel launches per norm (residual add → variance reduction → normalize+scale).
Fusing to 1 kernel eliminates 66 launches/frame and halves memory traffic.

**Approach:** Use the existing `candle-layer-norm` crate which has fused RMSNorm CUDA kernels
for candle. If it doesn't support residual-add fusion, extend or write our own PTX kernel.

**Steps:**
1. Add `candle-layer-norm` dependency, feature-gated behind `cuda`
2. Create `FusedRmsNorm` wrapper matching candle's `RmsNorm` interface
3. Wire into `DecoderLayer::forward()` in `transformer.rs`
4. Wire into `CodePredictor` layers
5. Unit test: compare fused vs sequential output on random tensors
6. Benchmark with e2e_bench

**Files:**
- `Cargo.toml`
- `src/models/transformer.rs` — DecoderLayer norm calls
- `src/models/code_predictor.rs` — CodePredictor norm calls

## Phase 2: Fused SwiGLU MLP (estimated 5-10% speedup)

**Why:** MLP does gate_proj → silu → up_proj → mul → down_proj. The silu+mul step is 2
kernel launches per layer that can become 1. gate_proj and up_proj share input (one load).

**Approach:** Write a custom PTX kernel via candle's `get_or_load_custom_func()`:
- Fused op: element-wise `silu(a) * b` (matmuls stay in cuBLAS)
- Eliminates 2 → 1 kernel launches per layer (×33 layers = 33 fewer launches/frame)

**Steps:**
1. Write `kernels/fused_silu_mul.cu` — element-wise `silu(a) * b`
2. Compile to PTX, embed via `include_str!`
3. Implement as `CustomOp2` in `src/models/fused_ops.rs`
4. Replace `Activation::Silu` + `Tensor::mul` in MLP::forward
5. Unit test: compare against sequential silu+mul
6. Benchmark

**Files:**
- `kernels/fused_silu_mul.cu` (new)
- `src/models/fused_ops.rs` (new)
- `src/models/transformer.rs` — MLP::forward
- `build.rs` or inline PTX string

## Phase 3: Fused RoPE (estimated 2-5% speedup)

**Why:** RoPE does cos/sin computation + element-wise ops as separate kernels. Fusing
saves memory round-trips. Runs twice per layer (Q and K) × 28 layers = 56 calls/frame.

**Steps:**
1. Write `kernels/fused_rope.cu` — combined cos/sin rotation
2. Implement as `CustomOp1`
3. Replace multi-step RoPE in `Attention::forward`
4. Unit test + benchmark

**Files:**
- `kernels/fused_rope.cu` (new)
- `src/models/fused_ops.rs` — add RoPE op
- `src/models/transformer.rs` — Attention::forward

## Iteration Protocol

After each phase:
1. `cargo test --lib` (must pass)
2. `cargo clippy --lib -- -D warnings`
3. e2e_bench with 3 iterations
4. Record in `docs/PERFORMANCE_JOURNAL.md`
5. Chrome trace to verify kernel count reduction
6. Commit + PR

## Expected Cumulative Impact

| After Phase | Estimated tok/s | Kernel launches/frame |
|-------------|----------------|-----------------------|
| Baseline | 18-19 | ~700 |
| 1 (Fused RmsNorm) | 22-24 | ~634 |
| 2 (Fused SwiGLU) | 24-26 | ~601 |
| 3 (Fused RoPE) | 25-27 | ~545 |
