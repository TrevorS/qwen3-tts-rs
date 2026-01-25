# Validation Status

Component-by-component validation of the Rust implementation against Python reference.

## Summary

| Component | Status | Max Diff | Notes |
|-----------|--------|----------|-------|
| Text Embedding | ✅ Pass | 0.0 | Exact match |
| Text Projection | ✅ Pass | < 1e-6 | SwiGLU MLP |
| RMS Norm | ✅ Pass | < 1e-6 | |
| QKV Projections | ✅ Pass | < 1e-6 | With QK normalization |
| RoPE | ✅ Pass | < 1e-6 | Rotary position embeddings |
| Attention | ✅ Pass | < 1e-5 | GQA with repeat_kv |
| O Projection | ✅ Pass | < 1e-6 | |
| MLP | ✅ Pass | < 1e-6 | SwiGLU |
| Full Layer 0 | ✅ Pass | < 1e-5 | End-to-end layer |
| 28-Layer Forward | ✅ Pass | 4.3e-5 | Full talker model |
| Final Norm + Codec Head | ✅ Pass | < 1e-4 | Semantic token logits |
| Code Predictor (5 layers) | ✅ Pass | 3.4e-5 | Acoustic token prediction |
| Quantizer Decode | ✅ Pass | < 1e-5 | Codebook lookup + sum |
| Pre-transformer (8 layers) | ✅ Pass | < 1e-6 | With layer scale |
| Causal Conv1d | ✅ Pass | < 1e-6 | Left-padded causal conv |
| SnakeBeta | ✅ Pass | < 1e-6 | x + (1/β)sin²(αx) |
| CausalTransConv1d | ✅ Pass | < 1e-6 | Right-only trimming |
| ConvNeXtBlock | ✅ Pass | < 1e-5 | dwconv + norm + pwconv |
| ResidualUnit | ✅ Pass | < 1e-5 | Dilated causal convs |
| DecoderBlock | ✅ Pass | < 1e-5 | Upsample + residual units |
| Full 12Hz Decoder | ✅ Pass | 3e-6 | Quantizer → audio |
| End-to-End Pipeline | ✅ Pass | 2e-6 | Text → audio |

**Test Totals:** 179 unit tests + 29 reference validation tests = 208 tests passing

## Architecture Details

### 1. Talker Model (Semantic Token Generation)

Generates semantic tokens (codebook 1) from text input.

**Architecture:**
- Text embedding: 151936 vocab → 2048 dim
- Text projection: 2048 → 1024 (SwiGLU MLP)
- 28 transformer layers:
  - Hidden size: 1024
  - Attention heads: 16 (query), 8 (KV) - GQA
  - Head dim: 128 (explicit override)
  - Intermediate size: 3072
  - QK normalization: RMSNorm per-head
  - RoPE theta: 1,000,000
- Codec head: 1024 → 3072 (semantic vocab)

### 2. Code Predictor (Acoustic Token Generation)

Generates 15 acoustic tokens (codebooks 2-16) per semantic token.

**Architecture:**
- 5 transformer layers (same structure as talker)
- 15 codec embeddings (2048 vocab → 1024 dim each)
- 15 lm_heads (1024 → 2048 each)

### 3. Decoder12Hz (Codes → Audio)

Converts 16-codebook tokens to 24kHz audio waveform.

**Architecture:**
- Split RVQ: 1 semantic + 15 acoustic quantizers
- Codebook dim: 256, output proj to 512
- Pre-conv: causal 1D conv (512 → 1024, kernel=3)
- Pre-transformer: 8 layers with layer scale 0.01
- Upsampling: ratios [8, 5, 4, 3, 2] → 960x total
- Decoder blocks: SnakeBeta + CausalTransConv + ResidualUnits
- Output: 24kHz mono audio

## Key Fixes Applied

### 1. CausalTransConv1d Trimming (Critical)

**Problem:** 75 frames produced 143445 samples instead of 144000.

**Root cause:** Symmetric trimming (left + right) instead of right-only.

**Fix:**
```rust
// Correct: right-only trim for exact input * stride output
let right_trim = kernel_size.saturating_sub(stride);
let left_trim = 0;

// Wrong: symmetric trim
let pad = (kernel_size - stride) / 2;
let left_trim = pad;
let right_trim = kernel_size - stride - pad;
```

### 2. QK Normalization

Apply RMSNorm to Q and K after projection, before RoPE:
```rust
let q = self.q_norm.forward(&q)?;
let k = self.k_norm.forward(&k)?;
```

### 3. RoPE Formula

```rust
// Correct: [x1*cos - x2*sin, x2*cos + x1*sin]
let rotated = Tensor::cat(&[
    &(x1.mul(&cos)? - x2.mul(&sin)?)?,
    &(x2.mul(&cos)? + x1.mul(&sin)?)?,  // NOT x1*sin
], D::Minus1)?;
```

### 4. Head Dimension Override

The model uses head_dim=128 explicitly, not hidden_size/num_heads=64:
```rust
pub fn head_dim(&self) -> usize {
    self.head_dim_override.unwrap_or(self.hidden_size / self.num_attention_heads)
}
```

### 5. Linear for 3D Tensors

Candle's matmul doesn't auto-broadcast 3D @ 2D:
```rust
fn linear(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let (batch, seq, features) = x.dims3()?;
    let x_2d = x.reshape((batch * seq, features))?;
    let out_2d = x_2d.matmul(&weight.t()?)?;
    out_2d.reshape((batch, seq, out_2d.dim(1)?))
}
```

## Running Validation

1. Download test data:
```bash
./scripts/download_test_data.sh
```

2. Export Python reference values:
```bash
cd tools && uv sync
uv run python export_reference_values.py
uv run python export_decoder_reference.py
```

3. Run validation tests:
```bash
cargo test --test reference_validation -- --nocapture
```

## Test Output Example

```
=== Full 28-Layer Forward Pass Validation ===
  Layer 0: mean=-0.001917
  Layer 7: mean=-0.017344
  Layer 14: mean=-0.032283
  Layer 21: mean=-0.027164
  Layer 27: mean=-0.006411
  after_all_layers: max_diff=0.000043
  28-LAYER FORWARD PASS!

=== Code Predictor Validation ===
  Layer 0: mean=0.045735
  Layer 4: mean=0.051924
  code_predictor_final: max_diff=0.000034
  acoustic_logits_0: max_diff=0.000016
  CODE PREDICTOR PASS!

=== Full 12Hz Decoder Validation ===
  Rust audio shape: [1, 1, 144000]
  Python audio shape: [1, 1, 144000]
  Max diff: 0.000003
  DECODER PASS!

=== End-to-End Pipeline Validation ===
  Text: "Hello" -> Audio: 144000 samples
  Max diff from Python: 0.000002
  END-TO-END PASS!
```
