# Plan: Python Bindings for qwen3-tts

Add a `qwen3-tts-py` subcrate using **PyO3 + maturin** to expose the Rust TTS library to Python.

## Structure

No file moves needed. Add workspace to existing `Cargo.toml` and create a new subcrate:

```
qwen3-tts-rs/
├── Cargo.toml              # Add [workspace] members = ["qwen3-tts-py"]
├── src/                    # Existing library (unchanged)
├── qwen3-tts-py/
│   ├── Cargo.toml          # cdylib crate depending on qwen3-tts (path = "..")
│   ├── pyproject.toml      # maturin build config
│   ├── src/
│   │   ├── lib.rs          # #[pymodule] entry point
│   │   ├── error.rs        # anyhow → PyErr conversion
│   │   ├── device.rs       # Device enum + auto_device/parse_device
│   │   ├── audio.rs        # AudioBuffer wrapper (numpy interop)
│   │   ├── options.rs      # SynthesisOptions wrapper
│   │   ├── enums.rs        # Speaker, Language, ModelType enums
│   │   ├── model.rs        # Qwen3TTS wrapper (main class)
│   │   └── streaming.rs    # StreamingSession wrapper
│   ├── python/
│   │   └── qwen3_tts/
│   │       ├── __init__.py   # Re-exports from native module
│   │       ├── __init__.pyi  # Type stubs for IDE support
│   │       └── py.typed      # PEP 561 marker
│   └── tests/
│       ├── test_types.py     # Unit tests for enums, options, audio buffer
│       └── test_synthesis.py # Integration tests (requires model)
```

## Key Design Decisions

### 1. Workspace without moving files

Add `[workspace] members = ["qwen3-tts-py"]` to root `Cargo.toml`. The existing library crate stays at root — zero disruption to current code.

### 2. StreamingSession lifetime

`StreamingSession<'a>` borrows `&'a Qwen3TTS` which PyO3 can't express. Two options:

- **Option A (safe, simple):** Wrap `Qwen3TTS` in `Arc`, clone into `StreamingSession`. Small overhead but zero unsafe code.
- **Option B (unsafe):** Transmute lifetime to `'static`, rely on Python refcount to keep model alive.

**Recommend Option A** — Arc overhead is negligible vs. inference cost, and it avoids unsafe.

### 3. GIL release

All synthesis methods call `py.allow_threads(|| ...)` to release the GIL during inference. This lets Python threads (e.g., audio playback) run concurrently.

### 4. Device as string

Rather than a Python enum, accept `device: str` (`"cpu"`, `"cuda"`, `"cuda:0"`, `"metal"`, `"auto"`) and delegate to `parse_device()`. Simpler, matches PyTorch convention.

### 5. numpy interop

`AudioBuffer.samples` returns `numpy.ndarray[float32]` via `rust-numpy` crate. Copies on access (audio buffers are small enough this is fine).

## Files to Create/Modify

### Modified: `Cargo.toml` (root)

Add workspace declaration:

```toml
[workspace]
members = ["qwen3-tts-py"]
```

### New: `qwen3-tts-py/Cargo.toml`

```toml
[package]
name = "qwen3-tts-py"
version = "0.1.0"
edition = "2021"

[lib]
name = "qwen3_tts"
crate-type = ["cdylib"]

[dependencies]
qwen3-tts = { path = "..", default-features = false }
pyo3 = { version = "0.23", features = ["extension-module"] }
numpy = "0.23"
anyhow = "1.0"

[features]
default = ["cpu"]
cpu = ["qwen3-tts/cpu"]
cuda = ["qwen3-tts/cuda"]
metal = ["qwen3-tts/metal"]
flash-attn = ["qwen3-tts/flash-attn"]
hub = ["qwen3-tts/hub"]
```

### New: `qwen3-tts-py/pyproject.toml`

```toml
[build-system]
requires = ["maturin>=1.7,<2.0"]
build-backend = "maturin"

[project]
name = "qwen3-tts"
version = "0.1.0"
description = "Python bindings for qwen3-tts Rust TTS inference"
requires-python = ">=3.9"
license = { text = "MIT" }
dependencies = ["numpy>=1.21"]

[project.optional-dependencies]
dev = ["pytest", "soundfile"]

[tool.maturin]
python-source = "python"
module-name = "qwen3_tts._qwen3_tts"
features = ["pyo3/extension-module"]
```

### New: `qwen3-tts-py/src/*.rs` (~400-500 lines total)

**`lib.rs`** — Module registration, `#[pymodule]` entry point

**`error.rs`** — `anyhow::Error` → `PyRuntimeError` / `PyValueError` conversion

**`device.rs`** — `parse_device(s: &str) -> Device` and `auto_device() -> Device` as `#[pyfunction]`

**`audio.rs`** — `AudioBuffer` pyclass:

- `__init__(samples, sample_rate)`
- `load(path)` / `save(path)` staticmethod/method
- `samples` property → numpy float32 array
- `sample_rate` property → int
- `duration()` → float
- `__len__()`, `normalize()`, `normalize_db(target_db)`

**`options.rs`** — `SynthesisOptions` pyclass with keyword-only defaults:

```rust
#[pyo3(signature = (*, max_length=2048, temperature=0.9, top_k=50, top_p=0.9,
                    repetition_penalty=1.05, eos_token_id=Some(2150),
                    chunk_frames=10, min_new_tokens=2, seed=None))]
```

**`enums.rs`** — Python enums for `Speaker` (9 variants), `Language` (10 variants), `ModelType` (3 variants). Each maps to/from Rust counterpart.

**`model.rs`** — `Qwen3TTS` pyclass:

- `from_pretrained(model_id: str, device: str = "auto")` — classmethod, releases GIL during load
- `synthesize(text, options=None)` → `AudioBuffer`
- `synthesize_with_voice(text, speaker, language, options=None)` → `AudioBuffer`
- `synthesize_voice_design(text, instruct, language, options=None)` → `AudioBuffer`
- `create_voice_clone_prompt(ref_audio, ref_text=None)` → `VoiceClonePrompt`
- `synthesize_voice_clone(text, prompt, language, options=None)` → `AudioBuffer`
- `synthesize_streaming(text, speaker, language, options=None)` → `StreamingSession`
- Properties: `model_type`, `supports_voice_cloning`, `supports_preset_speakers`, `supports_voice_design`

All synthesis methods release GIL via `py.allow_threads()`.

**`streaming.rs`** — `StreamingSession` pyclass:

- Internally holds `Arc<qwen3_tts::Qwen3TTS>` (cloned from model wrapper)
- `__iter__` / `__next__` protocol → yields `AudioBuffer` chunks
- `frames_generated()` → int
- `is_done()` → bool
- Releases GIL during each `next_chunk()` call

### New: `qwen3-tts-py/python/qwen3_tts/__init__.py`

```python
from ._qwen3_tts import (
    Qwen3TTS, AudioBuffer, SynthesisOptions, VoiceClonePrompt,
    StreamingSession, Speaker, Language, ModelType,
    auto_device, parse_device,
)

__version__ = "0.1.0"
__all__ = [
    "Qwen3TTS", "AudioBuffer", "SynthesisOptions", "VoiceClonePrompt",
    "StreamingSession", "Speaker", "Language", "ModelType",
    "auto_device", "parse_device",
]
```

### New: `qwen3-tts-py/python/qwen3_tts/__init__.pyi`

Full type stubs for all classes, methods, and functions (IDE autocompletion + mypy).

### New: `qwen3-tts-py/python/qwen3_tts/py.typed`

Empty file (PEP 561 marker).

## Target Python API

```python
from qwen3_tts import Qwen3TTS, SynthesisOptions, Speaker, Language, AudioBuffer

# Load model (device="auto" picks CUDA > Metal > CPU)
model = Qwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base", device="cuda")

# Simple synthesis
audio = model.synthesize("Hello world")
audio.save("out.wav")

# Preset voice
audio = model.synthesize_with_voice("Hello", Speaker.Ryan, Language.English)

# Custom options
audio = model.synthesize("Deterministic output",
    SynthesisOptions(temperature=0.7, seed=42))

# Streaming
for chunk in model.synthesize_streaming("Long text...", Speaker.Ryan, Language.English):
    play(chunk.samples)  # numpy float32 array, 24kHz

# Voice cloning
ref = AudioBuffer.load("reference.wav")
prompt = model.create_voice_clone_prompt(ref, "transcript text")
audio = model.synthesize_voice_clone("Clone this!", prompt, Language.English)

# numpy interop
import numpy as np
samples: np.ndarray = audio.samples  # shape: (N,), dtype: float32
print(f"{audio.duration():.1f}s at {audio.sample_rate}Hz")
```

## Implementation Order

1. Root `Cargo.toml` — add `[workspace]`
1. `qwen3-tts-py/Cargo.toml` + `pyproject.toml` — build scaffolding
1. `error.rs` — error conversion (everything depends on this)
1. `enums.rs` — Speaker, Language, ModelType
1. `options.rs` — SynthesisOptions
1. `audio.rs` — AudioBuffer + numpy
1. `device.rs` — device string parsing
1. `model.rs` — Qwen3TTS wrapper
1. `streaming.rs` — StreamingSession (Arc-based)
1. `lib.rs` — wire up pymodule
1. `python/` — __init__.py, stubs, py.typed
1. `tests/` — test_types.py, test_synthesis.py

## Build & Test

```bash
cd qwen3-tts-py
pip install maturin
maturin develop              # Debug build + install in venv
maturin develop --release    # Release build
pytest tests/                # Run tests
```

## Verification

1. `maturin develop` compiles without errors
1. `python -c "from qwen3_tts import Qwen3TTS, Speaker, Language; print('OK')"` imports cleanly
1. `pytest tests/test_types.py` passes (enum values, options defaults, audio buffer round-trip)
1. With a model downloaded: full synthesis produces valid WAV output
1. `cargo clippy` on workspace passes

## Open Questions

- PyPI package name: `qwen3-tts` (matches Rust) or `qwen3-tts-rs` (signals Rust backend)?
- Should we ship separate CPU/CUDA wheels or a single wheel with runtime detection?
- Include `hub` feature by default in the Python package for `from_pretrained` HF downloads?
