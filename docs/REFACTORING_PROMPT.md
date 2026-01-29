# Claude Code Refactoring Prompt for qwen3-tts-rs

Use this prompt to guide Claude Code through a systematic refactoring of the codebase.

______________________________________________________________________

## Master Prompt

````
You are refactoring qwen3-tts-rs, a pure Rust TTS library built on the Candle ML framework. Your goal is to transform this working codebase into idiomatic, maintainable Rust while preserving all functionality.

## Context

Read these files first to understand the current state:
1. `docs/CODE_IMPROVEMENTS.md` - Comprehensive catalog of 90+ issues
2. `src/lib.rs` - Main library (1,928 lines, needs splitting)
3. `src/models/transformer.rs` - Good reference for idiomatic patterns
4. `src/generation/sampling.rs` - Contains critical RNG issues
5. `Cargo.toml` - Dependencies and features

## Guiding Principles

1. **Preserve Functionality**: All existing tests must pass. Run `cargo test` after each change.
2. **Incremental Changes**: One logical change per commit. Never break the build.
3. **Idiomatic Rust**: Follow Rust API Guidelines (https://rust-lang.github.io/api-guidelines/)
4. **Idiomatic Candle**: Use VarBuilder::pp(), implement Module trait, leverage candle_nn utilities.
5. **No Regressions**: Maintain numerical equivalence with Python reference.

## Phase 1: Critical Fixes (Must Do First)

### Task 1.1: Fix Global RNG State
**Files:** `src/generation/sampling.rs`
**Issue:** GMS-001, GMS-002, GMS-003

The current global RNG breaks thread safety and determinism:
```rust
static RNG_STATE: AtomicU64 = AtomicU64::new(0);
static RNG_SEED: AtomicU64 = AtomicU64::new(0);
````

**Requirements:**

1. Remove all global static RNG state
1. Add `rng: Option<StdRng>` field to a new `SamplingContext` struct
1. Pass `SamplingContext` through all sampling functions
1. Update `SynthesisOptions` to accept an optional seed
1. Ensure deterministic output when seed is provided
1. Update all callers in `lib.rs`

**Test:** Same seed should produce identical output across runs and threads.

### Task 1.2: Extract Generation Loop

**Files:** `src/lib.rs` → new `src/generation/loop.rs`
**Issue:** DUP-001

The generation loop is duplicated 4 times (~300 lines each). Extract to shared implementation.

**Requirements:**

1. Create `src/generation/loop.rs` with:

```rust
pub struct GenerationLoop<'a> {
    talker: &'a TalkerModel,
    code_predictor: &'a CodePredictor,
    device: &'a Device,
    dtype: DType,
}

pub struct GenerationState {
    kv_caches: Vec<KVCache>,
    cp_kv_caches: Vec<KVCache>,
    offset: usize,
    all_codes: Vec<Vec<u32>>,
}

pub struct TrailingTextContext {
    hidden: Tensor,
    len: usize,
    pad_embed: Tensor,
}

impl<'a> GenerationLoop<'a> {
    pub fn new(...) -> Self;
    pub fn run(
        &self,
        initial_hidden: Tensor,
        initial_logits: Tensor,
        trailing: TrailingTextContext,
        config: &GenerationConfig,
        sampling_ctx: &mut SamplingContext,
    ) -> Result<Vec<Vec<u32>>>;
}
```

2. Replace all 4 duplicated loops with calls to `GenerationLoop::run()`
1. Ensure streaming still works (may need `GenerationLoop::step()` method)

**Test:** All synthesis methods produce identical output to before.

### Task 1.3: Create Token Constants Module

**Files:** `src/models/talker.rs`, `src/lib.rs` → new `src/tokens.rs`
**Issue:** SOC-003

**Requirements:**

1. Create `src/tokens.rs`:

```rust
//! Token IDs and constants for Qwen3-TTS

/// ChatML special tokens
pub mod chat {
    pub const IM_START: u32 = 151644;
    pub const IM_END: u32 = 151645;
    pub const ASSISTANT: u32 = 77091;
    pub const NEWLINE: u32 = 198;
}

/// TTS control tokens (text vocabulary)
pub mod tts {
    pub const PAD: u32 = 151671;
    pub const BOS: u32 = 151672;
    pub const EOS: u32 = 151673;
}

/// Codec tokens
pub mod codec {
    pub const SEMANTIC_OFFSET: u32 = 151936;
    pub const EOS: u32 = 151936 + 2150;  // CODEC_EOS_TOKEN_ID
    pub const VOCAB_SIZE: u32 = 3072;

    // Acoustic codebook
    pub const ACOUSTIC_VOCAB_SIZE: u32 = 2048;
}

/// Language token IDs
pub mod language {
    pub const ENGLISH: u32 = 151666;
    pub const CHINESE: u32 = 151667;
    // ... all languages
}

/// Speaker token IDs (CustomVoice models)
pub mod speaker {
    pub const RYAN: u32 = 151697;
    pub const VIVIAN: u32 = 151701;
    // ... all speakers
}
```

2. Replace all magic numbers with these constants
1. Update `Language` and `Speaker` enums to use these constants

## Phase 2: Module Restructuring

### Task 2.1: Split lib.rs

**Issue:** SOC-001

Create this structure:

```
src/
├── lib.rs                    # Re-exports only (~150 lines)
├── model.rs                  # Qwen3TTS struct, from_* methods
├── synthesis/
│   ├── mod.rs
│   ├── basic.rs             # synthesize(), synthesize_with_voice()
│   ├── voice_clone.rs       # synthesize_voice_clone(), VoiceClonePrompt
│   ├── voice_design.rs      # synthesize_voice_design()
│   └── streaming.rs         # StreamingSession
├── generation/
│   ├── mod.rs
│   ├── loop.rs              # GenerationLoop (from Task 1.2)
│   ├── sampling.rs          # Existing (updated in Task 1.1)
│   └── penalties.rs         # apply_generation_penalties, token suppression
├── device.rs                 # auto_device(), parse_device(), device_info()
├── options.rs                # SynthesisOptions, GenerationConfig
├── tokens.rs                 # From Task 1.3
└── error.rs                  # Typed errors
```

**Requirements:**

1. Move code in logical chunks, one commit per new file
1. Keep `pub use` re-exports in `lib.rs` for backward compatibility
1. Update all internal imports
1. Run `cargo test` after each move

### Task 2.2: Create Typed Errors

**Files:** new `src/error.rs`
**Issue:** ERR-001

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Qwen3TTSError {
    #[error("Model loading failed: {0}")]
    ModelLoad(#[from] ModelLoadError),

    #[error("Synthesis failed: {0}")]
    Synthesis(#[from] SynthesisError),

    #[error("Audio processing error: {0}")]
    Audio(#[from] AudioError),

    #[error("Device error: {0}")]
    Device(#[from] DeviceError),

    #[error("Tensor operation failed: {0}")]
    Tensor(#[from] candle_core::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Error, Debug)]
pub enum ModelLoadError {
    #[error("Missing weight: {name}")]
    MissingWeight { name: String },

    #[error("Config parse error at {path}: {source}")]
    ConfigParse {
        path: String,
        #[source] source: serde_json::Error
    },

    #[error("Incompatible model: expected {expected}, found {found}")]
    IncompatibleVariant { expected: ModelType, found: ModelType },

    #[error("Tokenizer load failed: {0}")]
    Tokenizer(String),
}

#[derive(Error, Debug)]
pub enum SynthesisError {
    #[error("Voice cloning requires Base model variant")]
    VoiceCloneUnsupported,

    #[error("Voice design requires VoiceDesign model variant")]
    VoiceDesignUnsupported,

    #[error("ICL mode requires speech encoder (not available in this model)")]
    SpeechEncoderRequired,

    #[error("Generation reached max length ({max}) without EOS")]
    MaxLengthReached { max: usize },

    #[error("Empty text provided")]
    EmptyText,
}

#[derive(Error, Debug)]
pub enum AudioError {
    #[error("Invalid sample rate: {rate} (expected {expected})")]
    InvalidSampleRate { rate: u32, expected: u32 },

    #[error("Audio too short: {duration_ms}ms (minimum {min_ms}ms)")]
    TooShort { duration_ms: u32, min_ms: u32 },

    #[error("Resampling failed: {0}")]
    Resample(String),
}

#[derive(Error, Debug)]
pub enum DeviceError {
    #[error("CUDA device {id} not available")]
    CudaNotAvailable { id: usize },

    #[error("Metal not available on this platform")]
    MetalNotAvailable,

    #[error("Unknown device: {spec}")]
    UnknownDevice { spec: String },
}

pub type Result<T> = std::result::Result<T, Qwen3TTSError>;
```

**Requirements:**

1. Create error types as shown
1. Replace `anyhow::Result` with `crate::error::Result` in library code
1. Keep `anyhow` for CLI only (`src/bin/generate_audio.rs`)
1. Add `.context()` calls for better error messages

## Phase 3: Idiomatic Candle Patterns

### Task 3.1: Fix VarBuilder Usage

**Files:** `src/lib.rs`, `src/models/*.rs`
**Issue:** IC-001, IC-002

**Current (bad):**

```rust
let cp_weights = Self::filter_weights(&weights, "talker.code_predictor.");
let cp_vb = VarBuilder::from_tensors(cp_weights, compute_dtype, &device);
```

**Target (good):**

```rust
let vb = VarBuilder::from_tensors(weights, dtype, &device);
let talker_vb = vb.pp("talker");
let cp_vb = talker_vb.pp("code_predictor");
```

**Requirements:**

1. Remove `filter_weights` helper function entirely
1. Create single VarBuilder at model load
1. Use `.pp()` for all component loading
1. Update TalkerModel, CodePredictor, SpeakerEncoder constructors

### Task 3.2: Implement Module Trait

**Files:** `src/models/talker.rs`, `src/models/code_predictor.rs`
**Issue:** IC-003

For structs with `forward` methods, implement `candle_nn::Module`:

```rust
use candle_nn::Module;

impl Module for TextProjection {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let hidden = self.fc1.forward(xs)?;
        let hidden = candle_nn::ops::silu(&hidden)?;
        self.fc2.forward(&hidden)
    }
}
```

**Apply to:**

- `TextProjection`
- `MLP`
- `DecoderLayer` (already close)
- `Attention` (needs signature change)

### Task 3.3: Cache Special Embeddings

**Files:** `src/models/talker.rs`
**Issue:** PERF-003, DUP-005

**Requirements:**

1. Add cached embeddings struct:

```rust
struct CachedEmbeddings {
    tts_pad: Tensor,
    tts_bos: Tensor,
    tts_eos: Tensor,
    role_prefix: Tensor,  // [IM_START, ASSISTANT, NEWLINE] projected
}
```

2. Compute once in `TalkerModel::new()`
1. Replace `get_tts_pad_embed()` etc. with cached lookups
1. Remove redundant computation in prefill methods

### Task 3.4: Fix KVCache Cloning

**Files:** `src/models/transformer.rs`
**Issue:** IC-007

**Current:**

```rust
pub fn update_k(&mut self, k: &Tensor) -> Result<Tensor> {
    let k = if let Some(prev_k) = &self.k {
        Tensor::cat(&[prev_k, k], 2)?
    } else {
        k.clone()
    };
    self.k = Some(k.clone());
    Ok(k)
}
```

**Target:**

```rust
pub fn update_k(&mut self, new_k: Tensor) -> Result<&Tensor> {
    self.k = Some(match self.k.take() {
        Some(prev) => Tensor::cat(&[&prev, &new_k], 2)?,
        None => new_k,
    });
    Ok(self.k.as_ref().unwrap())
}
```

Update all callers to work with references.

## Phase 4: Type Safety

### Task 4.1: Add Token Newtypes

**Files:** new `src/types.rs`
**Issue:** TS-001

```rust
/// Semantic token ID (codec vocabulary, offset by SEMANTIC_OFFSET)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct SemanticToken(pub u32);

impl SemanticToken {
    pub const EOS: Self = Self(tokens::codec::EOS - tokens::codec::SEMANTIC_OFFSET);

    pub fn new(id: u32) -> Option<Self> {
        (id < tokens::codec::VOCAB_SIZE).then_some(Self(id))
    }

    pub fn to_text_vocab(self) -> u32 {
        self.0 + tokens::codec::SEMANTIC_OFFSET
    }

    pub fn is_eos(self) -> bool {
        self == Self::EOS
    }
}

/// Acoustic token ID (codebook vocabulary, 0-2047)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct AcousticToken(pub u32);

impl AcousticToken {
    pub fn new(id: u32) -> Option<Self> {
        (id < tokens::codec::ACOUSTIC_VOCAB_SIZE).then_some(Self(id))
    }
}

/// A single frame of generated codes (1 semantic + 15 acoustic)
#[derive(Debug, Clone)]
pub struct CodecFrame {
    pub semantic: SemanticToken,
    pub acoustic: [AcousticToken; 15],
}

impl CodecFrame {
    pub fn to_u32_vec(&self) -> Vec<u32> {
        let mut v = vec![self.semantic.0];
        v.extend(self.acoustic.iter().map(|t| t.0));
        v
    }
}

/// Collection of generated frames
#[derive(Debug, Clone)]
pub struct GeneratedFrames(pub Vec<CodecFrame>);

impl GeneratedFrames {
    pub fn to_tensor(&self, device: &Device) -> Result<Tensor> {
        // Convert to [num_frames, 16] tensor
    }

    pub fn num_frames(&self) -> usize {
        self.0.len()
    }
}
```

### Task 4.2: Add Bounded Group Index

**Files:** `src/models/code_predictor.rs`
**Issue:** TS-002, ERR-005

```rust
/// Acoustic group index (0-14, representing the 15 acoustic codebooks)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AcousticGroupIdx(u8);

impl AcousticGroupIdx {
    pub const NUM_GROUPS: usize = 15;

    pub fn new(idx: usize) -> Option<Self> {
        (idx < Self::NUM_GROUPS).then_some(Self(idx as u8))
    }

    pub fn as_usize(self) -> usize {
        self.0 as usize
    }

    pub fn all() -> impl Iterator<Item = Self> {
        (0..Self::NUM_GROUPS).map(|i| Self(i as u8))
    }
}
```

## Phase 5: API Improvements

### Task 5.1: Add Builder for SynthesisOptions

**Files:** `src/options.rs`
**Issue:** API-002

```rust
#[derive(Debug, Clone)]
pub struct SynthesisOptions {
    pub(crate) max_length: usize,
    pub(crate) temperature: f64,
    pub(crate) top_k: usize,
    pub(crate) top_p: f64,
    pub(crate) repetition_penalty: f64,
    pub(crate) seed: Option<u64>,
}

impl SynthesisOptions {
    pub fn builder() -> SynthesisOptionsBuilder {
        SynthesisOptionsBuilder::default()
    }
}

#[derive(Debug, Clone, Default)]
pub struct SynthesisOptionsBuilder {
    max_length: Option<usize>,
    temperature: Option<f64>,
    top_k: Option<usize>,
    top_p: Option<f64>,
    repetition_penalty: Option<f64>,
    seed: Option<u64>,
}

impl SynthesisOptionsBuilder {
    pub fn max_length(mut self, v: usize) -> Self {
        self.max_length = Some(v);
        self
    }

    pub fn temperature(mut self, v: f64) -> Self {
        self.temperature = Some(v);
        self
    }

    pub fn top_k(mut self, v: usize) -> Self {
        self.top_k = Some(v);
        self
    }

    pub fn top_p(mut self, v: f64) -> Self {
        self.top_p = Some(v);
        self
    }

    pub fn repetition_penalty(mut self, v: f64) -> Self {
        self.repetition_penalty = Some(v);
        self
    }

    pub fn seed(mut self, v: u64) -> Self {
        self.seed = Some(v);
        self
    }

    pub fn build(self) -> SynthesisOptions {
        SynthesisOptions {
            max_length: self.max_length.unwrap_or(2048),
            temperature: self.temperature.unwrap_or(0.7),
            top_k: self.top_k.unwrap_or(50),
            top_p: self.top_p.unwrap_or(0.9),
            repetition_penalty: self.repetition_penalty.unwrap_or(1.0),
            seed: self.seed,
        }
    }
}

impl Default for SynthesisOptions {
    fn default() -> Self {
        Self::builder().build()
    }
}
```

### Task 5.2: Add Prelude Module

**Files:** new `src/prelude.rs`, update `src/lib.rs`
**Issue:** API-003

````rust
//! Convenient re-exports for common usage
//!
//! ```rust
//! use qwen3_tts::prelude::*;
//! ```

pub use crate::{
    // Main types
    Qwen3TTS,
    AudioBuffer,

    // Options and config
    SynthesisOptions,
    SynthesisOptionsBuilder,

    // Enums
    Speaker,
    Language,
    ModelType,

    // Voice cloning
    VoiceClonePrompt,
    VoiceCloneMode,

    // Device utilities
    auto_device,
    parse_device,

    // Errors
    Qwen3TTSError,
    Result,
};
````

### Task 5.3: Add Standard Trait Implementations

**Files:** `src/models/talker.rs` (Language, Speaker enums)
**Issue:** IR-011, IR-014

```rust
use std::str::FromStr;
use std::fmt::Display;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    English,
    Chinese,
    Japanese,
    Korean,
    German,
    French,
    Russian,
    Portuguese,
    Spanish,
    Italian,
}

impl Language {
    pub fn token_id(self) -> u32 {
        match self {
            Self::English => tokens::language::ENGLISH,
            // ...
        }
    }

    pub fn code(self) -> &'static str {
        match self {
            Self::English => "en",
            Self::Chinese => "zh",
            // ...
        }
    }
}

impl Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.code())
    }
}

impl FromStr for Language {
    type Err = ParseLanguageError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "en" | "english" => Ok(Self::English),
            "zh" | "chinese" => Ok(Self::Chinese),
            // ...
            _ => Err(ParseLanguageError(s.to_string())),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ParseLanguageError(String);

impl std::error::Error for ParseLanguageError {}
impl Display for ParseLanguageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Unknown language: {}", self.0)
    }
}
```

Do the same for `Speaker`.

## Phase 6: Additional Improvements

### Task 6.1: Deduplicate Mask Creation

**Files:** `src/models/talker.rs`, `src/models/code_predictor.rs` → `src/models/transformer.rs`
**Issue:** DUP-004

Move to transformer.rs:

```rust
/// Create a causal attention mask
pub fn create_causal_mask(
    seq_len: usize,
    offset: usize,
    device: &Device,
) -> Result<Tensor> {
    let total_len = offset + seq_len;
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| {
            (0..total_len).map(move |j| {
                if j <= offset + i { 0.0 } else { f32::NEG_INFINITY }
            })
        })
        .collect();

    Tensor::from_vec(mask, (1, 1, seq_len, total_len), device)
}

/// Create an ICL (in-context learning) mask
pub fn create_icl_mask(
    icl_len: usize,
    offset: usize,
    device: &Device,
) -> Result<Tensor> {
    let total_len = offset + icl_len;
    let mut mask = vec![0.0f32; icl_len * total_len];

    for i in 0..icl_len {
        for j in (offset + i + 1)..total_len {
            mask[i * total_len + j] = f32::NEG_INFINITY;
        }
    }

    Tensor::from_vec(mask, (1, 1, icl_len, total_len), device)
}
```

### Task 6.2: Add Benchmarks

**Files:** new `benches/synthesis.rs`
**Issue:** Missing performance tracking

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_sampling(c: &mut Criterion) {
    let device = Device::Cpu;
    let logits = Tensor::randn(0f32, 1.0, (1, 3072), &device).unwrap();
    let config = GenerationConfig::default();
    let mut ctx = SamplingContext::new(Some(42));

    c.bench_function("sample_top_k_50", |b| {
        b.iter(|| sample(&logits, &config, &mut ctx))
    });
}

fn bench_generation_step(c: &mut Criterion) {
    // Load small test model
    // Benchmark single generation step
}

criterion_group!(benches, bench_sampling, bench_generation_step);
criterion_main!(benches);
```

Add to Cargo.toml:

```toml
[[bench]]
name = "synthesis"
harness = false
```

## Verification Checklist

After each phase, verify:

1. **Compilation:** `cargo build --all-features`
1. **Tests:** `cargo test`
1. **Clippy:** `cargo clippy --all-features -- -D warnings`
1. **Format:** `cargo fmt --check`
1. **Docs:** `cargo doc --no-deps`

## Finding More Issues

After completing the above, run these commands to find additional issues:

```bash
# Find remaining unwraps
rg "\.unwrap\(\)" src/ --type rust

# Find remaining expects
rg "\.expect\(" src/ --type rust

# Find magic numbers
rg "\b\d{4,}\b" src/ --type rust | grep -v test

# Find TODO/FIXME comments
rg "TODO|FIXME|HACK|XXX" src/ --type rust

# Find large functions (>50 lines)
cargo install tokei
tokei src/ --files

# Find complex functions
cargo install cargo-complexity
cargo complexity

# Find missing docs
cargo doc --no-deps 2>&1 | grep "missing documentation"

# Run additional lints
cargo clippy --all-features -- -W clippy::pedantic -W clippy::nursery
```

## Commit Message Format

Use conventional commits:

```
type(scope): description

- Detailed bullet points
- Reference issues: Fixes #123

https://claude.ai/code/session_ID
```

Types: `feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `chore`
Scopes: `sampling`, `generation`, `model`, `audio`, `api`, `error`

````

---

## Quick Start Commands

```bash
# Start with Phase 1, Task 1.1
# Read the relevant files first:
cat src/generation/sampling.rs
cat docs/CODE_IMPROVEMENTS.md | grep -A 30 "GMS-001"

# After making changes:
cargo test
cargo clippy
git add -p  # Stage incrementally
git commit -m "refactor(sampling): remove global RNG state

- Add SamplingContext with optional StdRng
- Update sample() to take context parameter
- Add seed field to SynthesisOptions
- Ensures deterministic output when seeded

Fixes GMS-001, GMS-002, GMS-003"
````

______________________________________________________________________

## Notes for Claude

1. **Read before writing**: Always read the current implementation before making changes
1. **Small commits**: One logical change per commit, run tests between commits
1. **Preserve behavior**: The goal is refactoring, not adding features
1. **Ask if unclear**: If a requirement seems to conflict with existing behavior, ask
1. **Document decisions**: Add comments explaining non-obvious choices
1. **Update tests**: Add tests for new code, update tests for changed code
