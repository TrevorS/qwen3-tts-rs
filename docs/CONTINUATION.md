# Project Status

## Overview

Pure Rust implementation of Qwen3-TTS using Candle ML framework. The implementation is feature-complete and validated against Python reference.

## Validation Status

**All components validated with exact Python match:**

| Component | Status | Max Diff |
|-----------|--------|----------|
| Talker (28-layer transformer) | ✅ | 4.3e-5 |
| Code Predictor (5-layer) | ✅ | 3.4e-5 |
| Quantizer decode | ✅ | < 1e-5 |
| Pre-transformer (8-layer) | ✅ | < 1e-6 |
| Causal Conv1d | ✅ | < 1e-6 |
| CausalTransConv1d | ✅ | < 1e-6 |
| SnakeBeta activation | ✅ | < 1e-6 |
| ConvNeXtBlock | ✅ | < 1e-5 |
| ResidualUnit | ✅ | < 1e-5 |
| DecoderBlock | ✅ | < 1e-5 |
| Full 12Hz Decoder | ✅ | 3e-6 |
| End-to-end pipeline | ✅ | 2e-6 |

**Test counts:** 170 unit tests (1 ignored) + 41 integration tests + 29 reference validation tests

## Audio Quality Status

Tested on NVIDIA GB10 (bf16 + flash-attn) with seed=42, duration=3.0s, text="Hello world, this is a test."

| Variant | Mode | Status | Notes |
|---------|------|--------|-------|
| 0.6B Base | x_vector_only | Working | Produces audio |
| 0.6B Base | ICL | Not working | No intelligible speech |
| 0.6B CustomVoice | Ryan | Working | Says "hello world" in exaggerated slow voice — good progress |
| 0.6B CustomVoice | Serena | Working | Produces audio |
| 1.7B Base | x_vector_only | Working | Produces audio |
| 1.7B Base | ICL | Not working | No intelligible speech |
| 1.7B CustomVoice | Ryan | Working | Produces audio |
| 1.7B CustomVoice | Serena | Working | Produces audio |
| 1.7B VoiceDesign | * | Not supported | CLI lacks text-prompted voice description; `spk_id` is empty in config |

### Known Issues

- **ICL voice cloning** produces no intelligible speech on either model size. The generation completes without errors but the audio is garbled. This needs investigation — likely an issue in the ICL prefill or trailing text fusion.
- **VoiceDesign** uses `tts_model_type: voice_design` with an empty `spk_id` map. It requires a natural language voice description, which the CLI and Rust API don't support yet. Passing `--speaker ryan` falls through to undefined behavior (both speakers sound the same/male).
- **0.6B CustomVoice Ryan** speaks correctly but in an exaggerated, slow cadence. May be a generation config issue (temperature, repetition penalty, frame count).

## Recent Changes

### Runtime flash-attn fallback

- Changed flash-attn from compile-time `#[cfg]` gate to runtime `device.is_cuda()` check
- Same binary now works on both CPU (standard attention) and CUDA (flash attention 2)
- Removed `#[allow(dead_code)]` from `repeat_kv` since it's always reachable now
- 20/20 variant tests pass (10 CPU + 10 CUDA, all 5 models)

### Full bf16 + Flash Attention 2 pipeline

- Added `compute_dtype` field to `Qwen3TTS` — BF16 on CUDA, F32 on CPU
- Talker and code predictor now run entirely in bf16 on CUDA, matching the official Python recommendation (`dtype=torch.bfloat16, attn_implementation="flash_attention_2"`)
- Codec decoder and speaker encoder remain in F32 (convolutional, no attention)
- Added `flash-attn` feature flag for Flash Attention 2 via `candle-flash-attn`
- Fixed dtype boundary points: RoPE cos/sin casting, attention mask casting, logit F32 casting for sampling, speaker embedding casting at talker boundary, hardcoded F32 empty tensor in `get_projected_text_embeddings()`
- Validated on NVIDIA GB10 (Grace Blackwell, aarch64) in NGC container: clippy clean, 170 tests pass, 1.7B ICL inference produces intelligible speech

### Deep cleanup pass: VarBuilder conversion & dead code removal

- Converted `TalkerModel` from raw `HashMap<String, Tensor>` lookups to `VarBuilder` + typed candle_nn layers (`Embedding`, `Linear`, `RmsNorm`, `DecoderLayer`)
- Deleted `TalkerDecoderLayer` (~240 lines) — now reuses `DecoderLayer` from `transformer.rs`
- Moved `RoPEType` enum to `transformer.rs`, updated `DecoderLayer::forward()` and `CodePredictor` to use it
- Removed 5 dead methods from `TalkerModel`: `generate_custom_voice()`, `generate()`, `get_last_hidden()`, `generate_step()`, `generate_step_with_text()`
- Added config auto-detection in `TalkerModel::from_weights()` based on norm weight shape (1024 → Base, 2048 → CustomVoice)
- Removed dead `AudioCodec` wrapper, `AudioCodecConfig`, and `presets` module
- Removed unused `SpeakerEncoder` re-export from `models/mod.rs`
- Cleaned up underscore-prefixed fields (`_config`, `_device`) from `CodecDecoder`, `SpeakerEncoder`, `Decoder12Hz`
- Fixed README streaming example and CLI tools section

### Dead code cleanup & public API wiring

- Deleted stale binaries: `main.rs`, `tts_generate.rs`, `custom_voice_tts.rs`
- Renamed `qwen3_tts.rs` → `transformer.rs` (shared building blocks only)
- Removed dead `Qwen3TTSModel` struct (superseded by `TalkerModel`)
- Stripped `generation/tts.rs` to just `apply_token_suppression`
- Rewired `Qwen3TTS` public API to use the correct generation loop:
  residual VQ summation, trailing text fusion, autoregressive code prediction
  via `generate_step_with_embed()` (matching `generate_audio.rs`)
- Simplified `SynthesisOptions` (removed unused `speaker_embedding`, `language`)
- `synthesize_streaming()` now takes `Speaker` + `Language` parameters
- `StreamingSession` uses trailing text state for proper generation

## Features

### Core TTS Pipeline

- Text → TalkerModel → semantic tokens (CustomVoice prefill + autoregressive)
- Per-frame: semantic embed → CodePredictor → 15 acoustic codes
- Residual VQ sum + trailing text → next talker step
- All 16 codebook codes → Decoder12Hz → 24kHz audio

### API Features

- `Qwen3TTS::synthesize()` - Simple text-to-speech (Ryan/English defaults)
- `Qwen3TTS::synthesize_with_voice()` - CustomVoice speaker + language selection
- `Qwen3TTS::synthesize_streaming()` - Low-latency streaming with voice selection
- `ModelPaths::download()` - HuggingFace Hub integration

### Hardware Support

- CPU (default, F32)
- CUDA (feature flag, auto bf16 for transformer components)
- CUDA + Flash Attention 2 (`flash-attn` feature, requires CUDA toolkit)
- Metal (feature flag)
- MKL/Accelerate acceleration

## Quick Start

```rust
use qwen3_tts::{Qwen3TTS, auto_device};

let device = auto_device()?;
let model = Qwen3TTS::from_pretrained("path/to/model", device)?;
let audio = model.synthesize("Hello, world!", None)?;
audio.save("output.wav")?;
```

## Running Tests

```bash
# Unit tests
cargo test --lib

# Reference validation (requires test_data/)
cargo test --test reference_validation -- --nocapture

# All tests
cargo test
```

## Project Structure

```
src/
├── lib.rs              # Main API (Qwen3TTS, StreamingSession)
├── hub.rs              # HuggingFace Hub integration
├── audio/              # Audio I/O, mel spectrograms, resampling
├── generation/         # Token sampling and generation config
├── models/
│   ├── transformer.rs  # Shared building blocks (KVCache, RoPE, RoPEType, Attention, MLP, DecoderLayer)
│   ├── talker.rs       # TalkerModel (semantic token generation)
│   ├── code_predictor.rs # Acoustic token predictor
│   ├── config.rs       # Model configurations (Qwen3TTSConfig)
│   ├── speaker.rs      # Speaker encoder (ECAPA-TDNN)
│   └── codec/          # Audio decoder (12Hz)
└── tokenizer/          # Text tokenizer

tests/
├── reference_validation.rs  # Python reference comparison
└── integration.rs           # Integration tests
```

## Key Implementation Notes

### CausalTransConv1d Trimming

Trim from right side only for exact `input * stride` output:

```rust
let right_trim = kernel_size.saturating_sub(stride);
let left_trim = 0;  // NOT kernel_size / 2
```

### Linear for 3D Tensors

Candle's `candle_nn::Linear` handles 3D inputs natively. For modules still using raw
weight tensors (e.g. `Decoder12Hz`), a manual reshape is needed:

```rust
fn linear_3d(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let (batch, seq, features) = x.dims3()?;
    let x_2d = x.reshape((batch * seq, features))?;
    let out_2d = x_2d.matmul(&weight.t()?)?;
    out_2d.reshape((batch, seq, out_2d.dim(1)?))
}
```
