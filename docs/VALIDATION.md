# Validation Status

This document tracks the validation of the Rust implementation against the Python reference.

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

**Total: 215 tests passing** (157 unit + 43 integration + 15 reference validation)

## Validated Components

### 1. Talker Model (Semantic Token Generation)

The talker model generates semantic tokens (group 1) from text input.

**Architecture:**
- Text embedding: 151936 vocab → 2048 dim
- Text projection: 2048 → 1024 (SwiGLU MLP)
- 28 transformer layers:
  - Hidden size: 1024
  - Attention heads: 16 (query), 8 (KV) - GQA
  - Head dim: 128 (explicit, not hidden/heads)
  - Intermediate size: 3072
  - QK normalization: RMSNorm per-head
  - RoPE theta: 1,000,000
- Codec head: 1024 → 3072 (semantic vocab)

**Validation:**
```
Input: "Hello, this is a" → [9707, 11, 419, 374, 264]
Predictions (Rust): [1501, 1231, 1732, 1353, 963]
Predictions (Python): [1501, 1231, 1732, 1353, 963]
Result: EXACT MATCH
```

### 2. Code Predictor (Acoustic Token Generation)

The code predictor generates 15 acoustic tokens (groups 2-16) given the semantic token and talker hidden state.

**Architecture:**
- 5 transformer layers (same structure as talker)
- 15 codec embeddings (2048 vocab → 1024 dim each)
- 15 lm_heads (1024 → 2048 each)

**Validation:**
```
Semantic token: 963
Acoustic token 0 (Rust): 281
Acoustic token 0 (Python): 281
Result: EXACT MATCH
```

### 3. Speech Tokenizer Decoder (Codes → Audio)

The decoder converts codec tokens back to audio waveforms.

**Architecture:**
- Split RVQ: 1 semantic + 15 acoustic quantizers
- Codebook dim: 256, output proj to 512
- Pre-conv: causal 1D conv (512 → 1024, kernel=3)
- Pre-transformer: 8 layers
  - Hidden size: 512
  - Attention heads: 16
  - Head dim: 64
  - Layer scale: 0.01
- Upsampling: ratios [8, 5, 4, 3] → 480x
- Final decoder blocks

**Validated:**
- ✅ Quantizer decode (codebook lookup + sum + projection)
- ✅ Pre-transformer (8 layers with layer scale)
- ⏳ Pre-conv (needs causal conv implementation)
- ⏳ Upsampling blocks
- ⏳ Final decoder

## Key Fixes Applied During Validation

### 1. QK Normalization
The attention layer applies RMSNorm to Q and K after projection, before RoPE:
```rust
let q = self.q_norm.forward(&q)?;
let k = self.k_norm.forward(&k)?;
```

### 2. RoPE Formula
Correct formula for rotary embeddings:
```rust
// [x1*cos - x2*sin, x2*cos + x1*sin]
let rotated = Tensor::cat(&[
    &(x1.mul(&cos)? - x2.mul(&sin)?)?,
    &(x2.mul(&cos)? + x1.mul(&sin)?)?,  // NOT x1*sin
], D::Minus1)?;
```

### 3. Head Dimension Override
The model uses head_dim=128 explicitly, not hidden_size/num_heads=64:
```rust
pub fn head_dim(&self) -> usize {
    self.head_dim_override.unwrap_or(self.hidden_size / self.num_attention_heads)
}
```

### 4. Attention Mask Broadcasting
Use `broadcast_add` for combining 4D attention weights with 2D mask:
```rust
let attn_weights = attn_weights.broadcast_add(mask)?;
```

### 5. Linear for 3D Tensors
Candle's matmul doesn't auto-broadcast 3D @ 2D. Flatten, matmul, reshape:
```rust
fn linear(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let (batch, seq, features) = x.dims3()?;
    let x_2d = x.reshape((batch * seq, features))?;
    let out_2d = x_2d.matmul(&weight.t()?)?;
    out_2d.reshape((batch, seq, out_2d.dim(1)?))
}
```

## TODO

### High Priority
- [ ] Implement causal 1D convolution
- [ ] Implement upsampling blocks (ConvTranspose1d with ratios [8, 5, 4, 3])
- [ ] Implement final decoder blocks
- [ ] End-to-end audio generation test

### Medium Priority
- [ ] KV cache optimization for inference
- [ ] Streaming generation support
- [ ] Speaker embedding integration

### Low Priority
- [ ] GPU acceleration (CUDA)
- [ ] Quantization (INT8/FP16)
- [ ] ONNX export

## Running Validation

1. Ensure test data is in place:
```bash
ls test_data/model/model.safetensors
ls test_data/speech_tokenizer/model.safetensors
```

2. Export Python reference values:
```bash
source ../.venv/bin/activate
python tools/export_reference_values.py
python tools/export_decoder_reference.py
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
  Acoustic token 0 prediction: 281
  CODE PREDICTOR PASS!

=== Speech Tokenizer Decoder Validation ===
  quantized: max_diff=0.000000
  Quantizer decode PASS!
  pre_transformer: max_diff=0.000000
  Pre-transformer PASS!
  SPEECH TOKENIZER DECODER PASS!
```
