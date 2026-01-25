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

**Test counts:** 179 unit tests + 29 reference validation tests

## Features

### Core TTS Pipeline
- Text → TalkerModel → semantic tokens
- Semantic token → CodePredictor → 16 acoustic tokens
- Acoustic tokens → Decoder12Hz → 24kHz audio

### API Features
- `Qwen3TTS::synthesize()` - Simple text-to-speech
- `Qwen3TTS::synthesize_streaming()` - Low-latency streaming
- `Qwen3TTS::synthesize_with_voice()` - CustomVoice speaker selection
- `ModelPaths::download()` - HuggingFace Hub integration

### Hardware Support
- CPU (default)
- CUDA (feature flag)
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
│   ├── talker.rs       # TalkerModel (semantic token generation)
│   ├── code_predictor.rs # Acoustic token predictor
│   ├── speaker.rs      # Speaker encoder (ECAPA-TDNN)
│   └── codec/          # Audio decoder (12Hz)
└── tokenizer/          # Text tokenizer

tests/
├── reference_validation.rs  # Python reference comparison
└── debug_decoder_stages.rs  # Decoder debugging tests
```

## Key Implementation Notes

### CausalTransConv1d Trimming
Trim from right side only for exact `input * stride` output:
```rust
let right_trim = kernel_size.saturating_sub(stride);
let left_trim = 0;  // NOT kernel_size / 2
```

### Linear for 3D Tensors
Candle doesn't auto-broadcast 3D @ 2D:
```rust
fn linear(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let (batch, seq, features) = x.dims3()?;
    let x_2d = x.reshape((batch * seq, features))?;
    let out_2d = x_2d.matmul(&weight.t()?)?;
    out_2d.reshape((batch, seq, out_2d.dim(1)?))
}
```

### RoPE Duplication
Duplicate cos/sin with `.repeat(1, 2)` for full head_dim coverage.
