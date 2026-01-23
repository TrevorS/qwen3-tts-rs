# qwen3-tts-rs

Rust implementation of Qwen3-TTS (Text-to-Speech) using the Candle ML framework.

## Status

This is a work-in-progress port of the Qwen3-TTS model from Python/PyTorch to Rust. The implementation is being validated component-by-component against the Python reference to ensure numerical accuracy.

See [docs/VALIDATION.md](docs/VALIDATION.md) for detailed validation status.

### What Works

- **Talker Model**: 28-layer transformer for semantic token generation - fully validated
- **Code Predictor**: 5-layer transformer for acoustic token generation - fully validated
- **Speech Tokenizer Decoder**: Quantizer + 8-layer pre-transformer - validated (partial)

### What's Left

- Causal convolutions
- Upsampling blocks
- Final audio decoder
- End-to-end audio generation

## Project Structure

```
src/
├── lib.rs              # Main library entry point
├── audio/              # Audio I/O, mel spectrograms, resampling
├── generation/         # Token sampling and generation config
├── models/
│   ├── qwen3_tts.rs    # Main Talker model (transformer layers)
│   ├── code_predictor.rs # Acoustic token predictor
│   ├── config.rs       # Model configurations
│   ├── speaker.rs      # Speaker encoder (ECAPA-TDNN)
│   └── codec/          # Audio codec (quantizer, decoder)
└── tokenizer/          # Text tokenizer

tests/
├── integration.rs      # Integration tests with real weights
└── reference_validation.rs  # Numerical validation against Python

tools/
├── export_reference_values.py  # Export Python reference tensors
└── export_decoder_reference.py # Export decoder reference tensors
```

## Running Tests

```bash
# Run all tests
cargo test

# Run reference validation tests (requires test_data/)
cargo test --test reference_validation -- --nocapture

# Run integration tests
cargo test --test integration
```

## Test Data

Tests require model weights in `test_data/`:

```
test_data/
├── model/
│   └── model.safetensors     # Talker model weights (1.83GB)
├── tokenizer/
│   ├── tokenizer.json        # Text tokenizer
│   └── config.json           # Tokenizer config
└── speech_tokenizer/
    ├── model.safetensors     # Speech tokenizer weights
    └── config.json           # Decoder config
```

## License

See the main Qwen3-TTS repository for license information.
