# Continuation Prompt

Copy this to resume the validation blitz:

---

## Context

We're porting Qwen3-TTS from Python to Rust using Candle. The approach is methodical:
1. Export reference values from Python
2. Implement in Rust
3. Write validation test comparing to Python
4. Fix bugs until exact match
5. Move to next component

## Current Status

**Validated (all exact match):**
- Talker: 28-layer transformer, final norm, codec head → semantic tokens ✓
- Code Predictor: 5-layer transformer, 15 lm_heads → acoustic tokens ✓
- Speech Tokenizer Decoder: quantizer decode, 8-layer pre-transformer ✓
- Causal Conv1d: left-padded causal convolution for decoder.pre_conv ✓
- SnakeBeta: activation function x + (1/β)sin²(αx) ✓
- CausalTransConv1d: upsampling with causal output trimming ✓
- ConvNeXtBlock: dwconv + LayerNorm + pwconv1 + GELU + pwconv2 + gamma ✓
- ResidualUnit: SnakeBeta + dilated CausalConv + SnakeBeta + 1x1 CausalConv + residual ✓
- DecoderBlock: SnakeBeta + CausalTransConv + 3 ResidualUnits (dilations 1, 3, 9) ✓

**See:** `docs/VALIDATION.md` for details, `tests/reference_validation.rs` for tests (21 tests passing)

## Next Steps (in order)

1. **Full decoder integration** - Compose all validated parts into full decoder
   - quantizer → pre_conv → pre_transformer → output_proj → upsample stages (2x) → decoder.0 (initial conv) → decoder blocks (4x) → decoder.5 (final SnakeBeta) → decoder.6 (final conv) → audio
   - Wire up all components in CodecDecoder
   - Export end-to-end reference from Python and validate

2. **End-to-end test** - Full pipeline
   - Text → tokenize → talker → code_predictor → decoder → audio
   - Compare waveform to Python output

## How to Work

```bash
# 1. Check current test status
cargo test --test reference_validation -- --nocapture

# 2. Explore tensor structure for next component
cargo run --example inspect_tensors

# 3. Add to Python export script
# Edit tools/export_decoder_reference.py

# 4. Run export
source ../.venv/bin/activate && python tools/export_decoder_reference.py

# 5. Add Rust validation test
# Edit tests/reference_validation.rs

# 6. Run test, fix bugs until pass
cargo test --test reference_validation test_<component> -- --nocapture

# 7. Once passing, run full suite
cargo test
```

## Key Files

- `tools/export_reference_values.py` - Talker + code predictor reference export
- `tools/export_decoder_reference.py` - Speech tokenizer decoder reference export
- `tests/reference_validation.rs` - All validation tests
- `src/models/code_predictor.rs` - Acoustic token predictor (implemented)
- `src/models/codec/decoder.rs` - Audio decoder (partial)

## Patterns We've Established

**Linear for 3D tensors:** Candle doesn't broadcast 3D @ 2D, use helper:
```rust
fn linear(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor>
```

**RoPE:** Duplicate cos/sin with `.repeat(1, 2)` for full head_dim

**Layer scale:** Multiply output by learned scalar (decoder uses 0.01 init)

**Tolerance:** Use max_diff < 1e-3 for multi-layer forward, < 1e-5 for single ops

---

**Start with:** Implement causal conv1d, then work through upsampling stages.
