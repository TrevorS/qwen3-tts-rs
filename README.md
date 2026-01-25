# qwen3-tts

Pure Rust inference for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), a high-quality text-to-speech model from Alibaba.

## Features

- **CPU inference** with optional MKL/Accelerate for faster BLAS operations
- **CUDA** support for NVIDIA GPU acceleration
- **Metal** support for Apple Silicon
- **Streaming synthesis** for low-latency audio output
- **Voice cloning** with predefined speakers (CustomVoice model)
- **HuggingFace Hub integration** for easy model downloads

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
qwen3-tts = { version = "0.1", features = ["hub"] }
```

### Feature Flags

| Feature | Description |
|---------|-------------|
| `cpu` (default) | CPU inference |
| `cuda` | NVIDIA GPU acceleration |
| `metal` | Apple Silicon GPU acceleration |
| `mkl` | Intel MKL for faster CPU inference |
| `accelerate` | Apple Accelerate framework |
| `hub` | HuggingFace Hub model downloads |
| `cli` | Command-line tools |

## Quick Start

```rust
use qwen3_tts::{Qwen3TTS, SynthesisOptions, auto_device};

fn main() -> anyhow::Result<()> {
    // Select best available device (CUDA > Metal > CPU)
    let device = auto_device()?;

    // Load model from local path
    let model = Qwen3TTS::from_pretrained("path/to/model", device)?;

    // Synthesize speech
    let audio = model.synthesize("Hello, world!", None)?;
    audio.save("output.wav")?;

    // With custom options
    let options = SynthesisOptions {
        temperature: 0.8,
        top_k: 30,
        ..Default::default()
    };
    let audio = model.synthesize("Custom settings!", Some(options))?;

    Ok(())
}
```

### With HuggingFace Hub

```rust
use qwen3_tts::{Qwen3TTS, ModelPaths, auto_device};

fn main() -> anyhow::Result<()> {
    // Download model from HuggingFace (cached automatically)
    let paths = ModelPaths::download(None)?;
    let device = auto_device()?;

    let model = Qwen3TTS::from_paths(&paths, device)?;
    let audio = model.synthesize("Hello from HuggingFace!", None)?;
    audio.save("output.wav")?;

    Ok(())
}
```

### Streaming Synthesis

For low-latency applications, use streaming mode which yields audio in chunks:

```rust
use qwen3_tts::{Qwen3TTS, SynthesisOptions, auto_device};

fn main() -> anyhow::Result<()> {
    let device = auto_device()?;
    let model = Qwen3TTS::from_pretrained("path/to/model", device)?;

    let options = SynthesisOptions {
        chunk_frames: 10, // ~800ms per chunk
        ..Default::default()
    };

    for chunk in model.synthesize_streaming("Hello, world!", options)? {
        let audio = chunk?;
        // Play or stream this chunk (each ~800ms)
        println!("Got {} samples", audio.samples.len());
    }

    Ok(())
}
```

### Voice Cloning (CustomVoice Model)

Use predefined speakers from the CustomVoice model:

```rust
use qwen3_tts::{Qwen3TTS, Speaker, Language, SynthesisOptions, auto_device};

fn main() -> anyhow::Result<()> {
    let device = auto_device()?;
    // Load CustomVoice model (different from base model)
    let model = Qwen3TTS::from_pretrained("path/to/customvoice_model", device)?;

    let audio = model.synthesize_with_voice(
        "Hello in Ryan's voice!",
        Speaker::Ryan,
        Language::English,
        None,
    )?;
    audio.save("ryan.wav")?;

    Ok(())
}
```

Available speakers: `Serena`, `Vivian`, `UncleFu`, `Ryan`, `Aiden`, `OnoAnna`, `Sohee`, `Eric`, `Dylan`

## Architecture

The TTS pipeline consists of three stages:

1. **TalkerModel**: Transformer (28 layers) that generates semantic tokens from text autoregressively
2. **CodePredictor**: For each semantic token, generates 15 acoustic tokens using a 5-layer decoder
3. **Decoder12Hz**: Converts 16-codebook tokens to 24kHz audio via ConvNeXt blocks and upsampling

```
Text → TalkerModel → Semantic Token → CodePredictor → [16 tokens] → Decoder → Audio
              ↑                              ↑
         (per token)                   (per frame)
```

## CLI Tools

Build and run the CLI tools with:

```bash
# Basic TTS generation
cargo run --features cli --bin tts_generate -- --text "Hello, world!" --output output.wav

# Generate with specific seed (for reproducibility)
cargo run --features cli --bin generate_audio -- --text "Hello" --seed 42 --frames 25

# CustomVoice generation
cargo run --features cli --bin custom_voice_tts -- --text "Hello" --speaker ryan
```

## Test Data

Download test data for running validation tests:

```bash
./scripts/download_test_data.sh
cargo test --test reference_validation
```

## Model Files

| Component | HuggingFace Repo | Size |
|-----------|------------------|------|
| Base Model | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | 1.8 GB |
| Speech Tokenizer | `Qwen/Qwen3-TTS-Tokenizer-12Hz` | 682 MB |
| Text Tokenizer | `Qwen/Qwen2-0.5B` | 7 MB |
| CustomVoice | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | 1.8 GB |

## Sample Rate

Output audio is always 24kHz mono. Use `audio::resample()` if you need a different sample rate:

```rust
use qwen3_tts::audio;

let audio_48k = audio::resample(&audio, 48000)?;
```

## License

MIT License. See the main Qwen3-TTS repository for model license information.
