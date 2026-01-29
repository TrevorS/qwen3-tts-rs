# Comprehensive Code Improvements Catalog

This document catalogs all non-idiomatic patterns and improvement opportunities in `qwen3-tts-rs`. Issues are organized by category with specific file locations for systematic fixing.

## Quick Reference

| Category | Issue Count | Priority |
|----------|-------------|----------|
| [1. Separation of Concerns](#1-separation-of-concerns) | 8 | High |
| [2. Global Mutable State](#2-global-mutable-state) | 3 | High |
| [3. Error Handling](#3-error-handling) | 12 | Medium |
| [4. Idiomatic Rust](#4-idiomatic-rust) | 18 | Medium |
| [5. Idiomatic Candle](#5-idiomatic-candle) | 14 | Medium |
| [6. Code Duplication](#6-code-duplication) | 7 | High |
| [7. Type Safety](#7-type-safety) | 6 | Medium |
| [8. API Design](#8-api-design) | 9 | Low |
| [9. Performance](#9-performance) | 8 | Low |
| [10. Documentation](#10-documentation) | 5 | Low |

______________________________________________________________________

## 1. Separation of Concerns

### SOC-001: `lib.rs` is a God file (1,928 lines)

**File:** `src/lib.rs`
**Lines:** 1-1928
**Problem:** Single file handles model loading, synthesis (3 variants), streaming, voice cloning, ICL, device management, and codec utilities.

**Suggested Split:**

```
src/
├── lib.rs                    # Public API re-exports only (~100 lines)
├── model.rs                  # Qwen3TTS struct + from_* constructors
├── synthesis/
│   ├── mod.rs
│   ├── basic.rs             # synthesize(), synthesize_with_voice()
│   ├── voice_clone.rs       # synthesize_voice_clone(), VoiceClonePrompt
│   ├── voice_design.rs      # synthesize_voice_design()
│   └── streaming.rs         # StreamingSession
├── device.rs                 # auto_device(), parse_device(), device_info()
└── options.rs                # SynthesisOptions, GenerationConfig bridge
```

### SOC-002: Mixed config systems in single file

**File:** `src/models/config.rs`
**Lines:** 1-755
**Problem:** `Qwen3TTSConfig` (serde-based) and `ParsedModelConfig` (HF parser) serve different purposes but share a file.

**Fix:** Split into `config/generic.rs` and `config/hf_parser.rs`.

### SOC-003: Token constants scattered across modules

**Files:** `src/models/talker.rs:28-50`, `src/lib.rs:1494-1502`
**Problem:** Token IDs defined in multiple places.

**Fix:** Create `src/tokens.rs` with all token constants organized by category:

```rust
pub mod chat { pub const IM_START: u32 = 151644; ... }
pub mod tts { pub const PAD: u32 = 151671; ... }
pub mod codec { pub const THINK: u32 = 2154; ... }
```

### SOC-004: StreamingSession defined in lib.rs

**File:** `src/lib.rs`
**Lines:** 1504-1714
**Problem:** 210-line struct embedded in main lib file.

**Fix:** Move to `src/synthesis/streaming.rs`.

### SOC-005: Helper methods on Qwen3TTS should be standalone

**File:** `src/lib.rs`
**Lines:** 1296-1350, 1358-1397, 1470-1491
**Problem:** `tensor_to_frame_codes`, `sum_ref_codec_embeddings`, `build_default_trailing_text`, `apply_generation_penalties`, `filter_weights` are generic utilities.

**Fix:** Move to appropriate modules (`codec_utils.rs`, `generation/penalties.rs`).

### SOC-006: VoiceClonePrompt defined in lib.rs

**File:** `src/lib.rs`
**Lines:** 114-125
**Problem:** Data structure mixed with main API.

**Fix:** Move to `src/synthesis/voice_clone.rs`.

### SOC-007: SynthesisOptions defined in lib.rs

**File:** `src/lib.rs`
**Lines:** 1716-1750
**Problem:** Options struct should be standalone.

**Fix:** Move to `src/options.rs`.

### SOC-008: Device utilities in lib.rs

**File:** `src/lib.rs`
**Lines:** 1752-1849
**Problem:** `auto_device()`, `parse_device()`, `device_info()` are utilities, not model logic.

**Fix:** Move to `src/device.rs`.

______________________________________________________________________

## 2. Global Mutable State

### GMS-001: Global RNG state with atomics

**File:** `src/generation/sampling.rs`
**Lines:** 10-13

```rust
static RNG_STATE: AtomicU64 = AtomicU64::new(0);
static RNG_SEED: AtomicU64 = AtomicU64::new(0);
static RNG_SEEDED: AtomicU64 = AtomicU64::new(0);
```

**Problem:** Global mutable state causes:

- Non-deterministic behavior across threads
- Testing isolation issues
- Library composability problems

**Fix:** Use explicit RNG passed through `GenerationConfig`:

```rust
pub struct GenerationConfig {
    pub rng: Option<StdRng>,  // Or Box<dyn RngCore + Send>
    // ... other fields
}
```

### GMS-002: COUNTER atomic in rand_f32

**File:** `src/generation/sampling.rs`
**Lines:** 284

```rust
static COUNTER: AtomicU64 = AtomicU64::new(0);
```

**Problem:** Secondary global state for unseeded RNG.

**Fix:** Part of GMS-001 fix.

### GMS-003: set_seed/clear_seed/reset_rng global functions

**File:** `src/generation/sampling.rs`
**Lines:** 22-52
**Problem:** Global seed management functions.

**Fix:** Replace with `GenerationConfig::with_seed(u64)` builder method.

______________________________________________________________________

## 3. Error Handling

### ERR-001: Mixed anyhow/thiserror usage

**Files:** All source files
**Problem:** `anyhow::Result` used everywhere, losing type information.

**Fix:** Use `thiserror` for library errors, reserve `anyhow` for CLI:

```rust
// src/error.rs
#[derive(Error, Debug)]
pub enum Qwen3TTSError {
    #[error("Model loading failed: {0}")]
    ModelLoad(#[source] ModelLoadError),
    #[error("Tensor operation failed")]
    Tensor(#[from] candle_core::Error),
    // ...
}
```

### ERR-002: Inconsistent bail!/anyhow! usage

**File:** `src/lib.rs`
**Lines:** 195-198, 244-248, 296-298
**Problem:** Mix of `anyhow::bail!` and `Err(anyhow::anyhow!(...))`.

**Fix:** Consistently use `bail!` for early returns, `anyhow!` for constructing errors.

### ERR-003: Missing context on errors

**File:** `src/lib.rs`
**Lines:** 296-298

```rust
.ok_or_else(|| anyhow::anyhow!("Missing talker.model.norm.weight"))?;
```

**Fix:** Use `.context()` for better error messages:

```rust
.context("Missing talker.model.norm.weight in model weights")?;
```

### ERR-004: Unwrap in debugging code

**File:** `src/models/transformer.rs`
**Lines:** 544-558 (k_sum, v_sum)

```rust
.unwrap()  // Multiple unwraps in debug methods
```

**Fix:** Return `Result` or use `?` with `Option::ok_or()`.

### ERR-005: Panic on invalid group_idx

**File:** `src/models/code_predictor.rs`
**Lines:** 404-409, 427-432

```rust
anyhow::bail!("Invalid group_idx...")
```

**Problem:** Runtime check for compile-time invariant.

**Fix:** Use newtype with bounds checking at construction:

```rust
pub struct AcousticGroupIdx(usize);
impl AcousticGroupIdx {
    pub fn new(idx: usize) -> Option<Self> {
        (idx < 15).then_some(Self(idx))
    }
}
```

### ERR-006: Silent fallback on speech encoder load failure

**File:** `src/lib.rs`
**Lines:** 1455-1466

```rust
Err(e) => {
    tracing::debug!("Speech encoder not available...");
    Ok(None)
}
```

**Problem:** Error silently swallowed; user may expect ICL to work.

**Fix:** Return a warning-level log or store the error for later query.

### ERR-007: Panic-prone vector indexing

**File:** `src/generation/sampling.rs`
**Lines:** 149, 195

```rust
let threshold = sorted[k - 1];  // Panics if k > vocab
```

**Fix:** Use `.get()` with proper bounds checking.

### ERR-008: Missing error type for device parsing

**File:** `src/lib.rs`
**Lines:** 1804-1839
**Problem:** Device parsing errors are strings.

**Fix:** Create `DeviceParseError` enum.

### ERR-009: Inconsistent error messages

**Files:** Multiple
**Problem:** Some errors start with capital, some don't; inconsistent punctuation.

**Fix:** Standardize: lowercase start, no trailing period, include context.

### ERR-010: No recovery strategy for max_length reached

**File:** `src/lib.rs`
**Lines:** 560-613
**Problem:** Generation silently stops at max_length.

**Fix:** Return a `GenerationResult` enum with `Completed` vs `Truncated` variants.

### ERR-011: Clone in error path

**File:** `src/lib.rs`
**Lines:** 1373

```rust
logits.clone()  // Unnecessary clone when penalty is 1.0
```

**Fix:** Return borrowed reference or use Cow.

### ERR-012: Error message includes internal details

**File:** `src/models/config.rs`
**Lines:** 238-242
**Problem:** Exposes JSON parsing internals to users.

**Fix:** Wrap with user-friendly message.

______________________________________________________________________

## 4. Idiomatic Rust

### IR-001: Manual Option handling instead of combinators

**File:** `src/lib.rs`
**Lines:** 527-533, 672-678

```rust
let trailing_text_hidden = if input_ids.len() > 1 {
    let remaining_proj = self.talker.get_projected_text_embeddings(&input_ids[1..])?;
    // ...
} else {
    self.talker.get_tts_eos_embed()?
};
```

**Fix:** Use `Option` combinators:

```rust
let trailing = (input_ids.len() > 1)
    .then(|| self.build_trailing(&input_ids[1..]))
    .transpose()?
    .unwrap_or_else(|| self.talker.get_tts_eos_embed())?;
```

### IR-002: Redundant clone() calls

**File:** `src/lib.rs`
**Lines:** 591, 733, 944, 1248

```rust
tts_pad_embed.clone()  // In loop, cloned every iteration
```

**Fix:** Clone once outside loop or use reference.

### IR-003: Unnecessary to_string() on &str

**File:** `src/lib.rs`
**Lines:** 1485

```rust
k.strip_prefix(prefix).unwrap().to_string()
```

**Fix:** Use `String::from()` or `.into()` for clarity.

### IR-004: Manual iteration instead of iterators

**File:** `src/models/code_predictor.rs`
**Lines:** 455-463

```rust
let mut sum: Option<Tensor> = None;
for (i, &code) in acoustic_codes.iter().enumerate() {
    let embed = self.get_acoustic_embedding(code, i, device)?;
    sum = Some(match sum { ... });
}
```

**Fix:** Use `try_fold`:

```rust
acoustic_codes.iter()
    .enumerate()
    .try_fold(None, |acc, (i, &code)| -> Result<Option<Tensor>> {
        let embed = self.get_acoustic_embedding(code, i, device)?;
        Ok(Some(match acc { ... }))
    })?
```

### IR-005: pub fields on data structs

**File:** `src/audio/io.rs`
**Lines:** 28-33

```rust
pub struct AudioBuffer {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}
```

**Problem:** Direct field access prevents future invariant enforcement.

**Fix:** Use getters/setters or make fields private with constructor.

### IR-006: Missing #[must_use] on pure functions

**Files:** Multiple
**Lines:** `audio/io.rs:57-59`, `lib.rs:428-430`

```rust
pub fn duration(&self) -> f32 { ... }
pub fn model_type(&self) -> Option<&ModelType> { ... }
```

**Fix:** Add `#[must_use]` attribute.

### IR-007: Type alias would improve readability

**File:** `src/lib.rs`
**Lines:** Multiple

```rust
Vec<Vec<u32>>  // Used for frame codes everywhere
```

**Fix:** Create type alias:

```rust
pub type FrameCodes = Vec<Vec<u32>>;
// Or better, a newtype:
pub struct FrameCodes(Vec<CodecFrame>);
```

### IR-008: Unused fields with underscore prefix

**File:** `src/models/transformer.rs`
**Lines:** 19, 91-95

```rust
_dim: usize,
_mrope_section: [usize; 3],
_h_mask: Vec<bool>,
_w_mask: Vec<bool>,
```

**Problem:** Dead code that should either be used or removed.

**Fix:** Remove if truly unused, or implement functionality.

### IR-009: Manual Default impl could be derived

**File:** `src/models/config.rs`
**Lines:** 403-425
**Problem:** Manual `Default` implementation duplicates field defaults.

**Fix:** Use `#[derive(Default)]` with `#[serde(default = "...")]`.

### IR-010: Inconsistent method naming

**Files:** Multiple

```rust
has_speaker_encoder()  // vs
supports_voice_cloning()  // vs
is_seeded()
```

**Fix:** Standardize on `has_*` for presence, `supports_*` for capability, `is_*` for state.

### IR-011: Missing From/Into implementations

**File:** `src/models/talker.rs`
**Lines:** 52-83, 86-128
**Problem:** `Language` and `Speaker` enums lack `FromStr`, `Display`.

**Fix:** Implement standard conversion traits:

```rust
impl FromStr for Language { ... }
impl Display for Language { ... }
impl TryFrom<u32> for Language { ... }
```

### IR-012: Raw loop instead of collect

**File:** `src/models/code_predictor.rs`
**Lines:** 174-181, 207-213

```rust
let mut codec_embeddings = Vec::with_capacity(num_acoustic_groups);
for i in 0..num_acoustic_groups {
    codec_embeddings.push(embedding(...)?);
}
```

**Fix:** Use iterator:

```rust
let codec_embeddings: Vec<_> = (0..num_acoustic_groups)
    .map(|i| embedding(...))
    .collect::<Result<_>>()?;
```

### IR-013: Inconsistent visibility

**File:** `src/models/speaker.rs`
**Problem:** Helper structs like `TimeDelayNetBlock`, `Res2NetBlock` are private but could be useful.

**Fix:** Make `pub(crate)` for internal reuse or fully document public API.

### IR-014: Missing Copy derive where applicable

**File:** `src/models/talker.rs`
**Lines:** 53, 86

```rust
pub enum Language { ... }  // No Copy
pub enum Speaker { ... }   // No Copy
```

**Fix:** Add `#[derive(Copy)]` to small enums.

### IR-015: Inefficient string formatting

**File:** `src/models/code_predictor.rs`
**Lines:** 179, 199, 209

```rust
vb.pp(format!("model.codec_embedding.{}", i))
```

**Fix:** Use `vb.pp(format_args!(...))` or pre-format once.

### IR-016: Magic numbers

**File:** `src/lib.rs`
**Lines:** 814, 1111

```rust
options.max_length.min(75.max(input_ids.len() * 6))
```

**Problem:** Magic numbers `75` and `6` unexplained.

**Fix:** Extract to named constants:

```rust
const ICL_MIN_TOKENS: usize = 75;
const ICL_TOKENS_PER_TEXT: usize = 6;
```

### IR-017: Hardcoded array sizes

**File:** `src/models/config.rs`
**Lines:** 276-283

```rust
if arr.len() == 3 { Some([arr[0]..., arr[1]..., arr[2]...]) }
```

**Fix:** Use `TryFrom<Vec<_>>` for `[T; N]` conversion.

### IR-018: Inconsistent use of Self

**Files:** Multiple
**Problem:** Mix of `Self::method()` and direct calls.

**Fix:** Prefer `Self::` for associated functions, direct for methods.

______________________________________________________________________

## 5. Idiomatic Candle

### IC-001: Manual weight filtering instead of VarBuilder::pp

**File:** `src/lib.rs`
**Lines:** 227-229, 322-324, 400-402, 1479-1491

```rust
let cp_weights = Self::filter_weights(&weights, "talker.code_predictor.");
let cp_vb = VarBuilder::from_tensors(cp_weights, compute_dtype, &device);
```

**Fix:** Use `VarBuilder::pp()` throughout:

```rust
let vb = VarBuilder::from_tensors(weights, dtype, device);
let cp_vb = vb.pp("talker").pp("code_predictor");
```

### IC-002: Repeated VarBuilder creation

**File:** `src/lib.rs`
**Lines:** 228, 323, 401, 1432
**Problem:** Creating new VarBuilder for each component.

**Fix:** Create once and use `.pp()` for scoping.

### IC-003: Not using candle_nn::Module trait

**File:** `src/models/talker.rs`
**Lines:** 248-277

```rust
impl TextProjection {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> { ... }
}
```

**Fix:** Implement `candle_nn::Module`:

```rust
impl Module for TextProjection {
    fn forward(&self, x: &Tensor) -> Result<Tensor> { ... }
}
```

### IC-004: Repeated dtype conversions

**File:** `src/lib.rs`
**Lines:** 1366

```rust
let logits = logits.to_dtype(DType::F32)?;  // Every generation step
```

**Fix:** Convert once after model forward, or keep dtype field.

### IC-005: Manual causal mask creation

**File:** `src/models/talker.rs`
**Lines:** 940-955

```rust
fn create_causal_mask(&self, seq_len: usize, offset: usize) -> Result<Tensor> {
    let mask: Vec<f32> = (0..seq_len).flat_map(|i| { ... }).collect();
    // ...
}
```

**Fix:** Use `candle_transformers::utils::get_causal_mask` or cache masks.

### IC-006: Repeated mask creation in generation loop

**File:** `src/lib.rs`
**Lines:** 596-598 (inside loop)
**Problem:** Mask created every iteration.

**Fix:** Use KV cache length to determine mask, create once.

### IC-007: Tensor cloning in KVCache::update

**File:** `src/models/transformer.rs`
**Lines:** 523-541

```rust
self.k = Some(k.clone());
Ok(k)
```

**Fix:** Return reference to cached tensor:

```rust
pub fn update_k(&mut self, new_k: Tensor) -> Result<&Tensor> {
    self.k = Some(match self.k.take() { ... });
    Ok(self.k.as_ref().unwrap())
}
```

### IC-008: Missing contiguous() before matmul

**File:** `src/models/transformer.rs`
**Lines:** 388-392
**Problem:** Transpose creates non-contiguous view.

**Status:** Already fixed with `.contiguous()`, but pattern repeated elsewhere.

### IC-009: Inefficient embedding lookup in loop

**File:** `src/models/code_predictor.rs`
**Lines:** 346-356

```rust
for group_idx in 1..num_acoustic {
    let code_tensor = Tensor::new(&[prev_code], device)?;
    let code_embed = self.codec_embeddings[group_idx - 1].forward(&code_tensor)?;
    // ...
}
```

**Fix:** Batch embedding lookups where possible.

### IC-010: Not using candle's built-in activations

**File:** `src/models/speaker.rs`
**Lines:** 53-60

```rust
fn relu(x: &Tensor) -> Result<Tensor> {
    Ok(x.maximum(&x.zeros_like()?)?)
}
fn sigmoid(x: &Tensor) -> Result<Tensor> { ... }
```

**Fix:** Use `candle_nn::ops::relu`, `x.sigmoid()`.

### IC-011: Manual softmax reimplementation

**File:** `src/generation/sampling.rs`
**Lines:** 175-180

```rust
let max_val = row[indices[0]];
let mut exp_sorted: Vec<f32> = indices.iter().map(|&i| (row[i] - max_val).exp()).collect();
let sum: f32 = exp_sorted.iter().sum();
```

**Fix:** Use `candle_nn::ops::softmax_last_dim`.

### IC-012: Inconsistent use of D::Minus1 vs -1

**Files:** Multiple
**Problem:** Mix of `D::Minus1` and magic index `-1`.

**Fix:** Always use `D::Minus1`, `D::Minus2` for clarity.

### IC-013: Not leveraging tensor broadcasting

**File:** `src/models/talker.rs`
**Lines:** 488-489

```rust
let tts_pad_expanded = tts_pad_proj.broadcast_as((1, 5, self.config.hidden_size))?;
```

**Fix:** Could use `.expand()` with shape inference.

### IC-014: Repeated device references

**Files:** Multiple

```rust
&self.device  // Passed around constantly
```

**Fix:** Store device in config or use tensor's device.

______________________________________________________________________

## 6. Code Duplication

### DUP-001: Generation loop duplicated 4 times

**File:** `src/lib.rs`
**Lines:** 559-613, 708-753, 918-964, 1222-1268
**Problem:** Nearly identical generation loops in:

- `synthesize_with_voice`
- `synthesize_voice_design`
- `synthesize_voice_clone` (twice for debug variant)

**Fix:** Extract to `GenerationLoop` struct:

```rust
struct GenerationLoop<'a> {
    model: &'a Qwen3TTS,
    config: &'a GenerationConfig,
    kv_caches: Vec<KVCache>,
    trailing_text: TrailingText,
}

impl GenerationLoop<'_> {
    fn run(&mut self, initial_logits: Tensor) -> Result<Vec<Vec<u32>>> { ... }
}
```

### DUP-002: Trailing text construction duplicated

**File:** `src/lib.rs`
**Lines:** 527-535, 672-680, 1342-1350, 1545-1556

```rust
let trailing_text_hidden = if input_ids.len() > 1 {
    let remaining_proj = self.talker.get_projected_text_embeddings(&input_ids[1..])?;
    let tts_eos_embed = self.talker.get_tts_eos_embed()?;
    Tensor::cat(&[&remaining_proj, &tts_eos_embed], 1)?
} else {
    self.talker.get_tts_eos_embed()?
};
```

**Fix:** Already has `build_default_trailing_text` - use it everywhere.

### DUP-003: ICL mask creation duplicated

**File:** `src/lib.rs`
**Lines:** 862-870, 1163-1170

```rust
let mut mask_data = vec![0.0f32; icl_len * (offset + icl_len)];
for i in 0..icl_len {
    for j in (offset + i + 1)..(offset + icl_len) {
        mask_data[i * (offset + icl_len) + j] = f32::NEG_INFINITY;
    }
}
```

**Fix:** Extract to helper function `create_icl_mask(icl_len, offset, device)`.

### DUP-004: Causal mask creation duplicated

**Files:** `src/models/talker.rs:940-955`, `src/models/code_predictor.rs:380-392`
**Problem:** Same mask creation logic in two files.

**Fix:** Move to `transformer.rs` as shared utility.

### DUP-005: TTS token embedding pattern repeated

**File:** `src/models/talker.rs`
**Lines:** 477-485, 595-604, 696-708

```rust
let tts_pad_id = Tensor::new(&[TTS_PAD], &self.device)?;
let tts_pad_embed = self.text_embedding.forward(&tts_pad_id)?;
let tts_pad_embed = tts_pad_embed.unsqueeze(0)?;
let tts_pad_proj = self.text_projection.forward(&tts_pad_embed)?;
```

**Fix:** Cache these embeddings at construction time.

### DUP-006: Config conversion methods duplicated

**File:** `src/models/config.rs` vs `src/models/talker.rs`
**Problem:** `to_layer_config()` in multiple places.

**Fix:** Single conversion point in config module.

### DUP-007: Role prefix construction duplicated

**File:** `src/models/talker.rs`
**Lines:** 456-459, 563-566, 676-679

```rust
let role_prefix_ids = Tensor::new(&[IM_START, ASSISTANT, NEWLINE], &self.device)?;
let role_prefix_embed = self.text_embedding.forward(&role_prefix_ids)?;
let role_prefix_embed = role_prefix_embed.unsqueeze(0)?;
let role_prefix_hidden = self.text_projection.forward(&role_prefix_embed)?;
```

**Fix:** Cache at construction or extract helper method.

______________________________________________________________________

## 7. Type Safety

### TS-001: Raw u32 for all token types

**Files:** Multiple
**Problem:** Semantic, acoustic, and text tokens all use `u32`.

**Fix:** Create newtype wrappers:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemanticToken(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AcousticToken(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextToken(pub u32);
```

### TS-002: Unchecked acoustic group index

**File:** `src/models/code_predictor.rs`
**Lines:** 398-413
**Problem:** `group_idx: usize` can be any value.

**Fix:** Use bounded newtype (see ERR-005).

### TS-003: Model type runtime checks

**File:** `src/lib.rs`
**Lines:** 433-455

```rust
pub fn supports_voice_cloning(&self) -> bool { ... }
pub fn supports_preset_speakers(&self) -> bool { ... }
pub fn supports_voice_design(&self) -> bool { ... }
```

**Problem:** Runtime capability checks could be compile-time.

**Fix:** Use type-state pattern:

```rust
pub struct Qwen3TTS<V: Variant> { ... }
impl Qwen3TTS<BaseVariant> {
    pub fn synthesize_voice_clone(&self, ...) { ... }
}
```

### TS-004: Optional fields that should be required

**File:** `src/lib.rs`
**Lines:** 143

```rust
speaker_encoder: Option<SpeakerEncoder>,
```

**Problem:** Option used where type-state would be clearer.

**Fix:** Part of TS-003 fix.

### TS-005: Stringly-typed device selection

**File:** `src/lib.rs`
**Lines:** 1804

```rust
pub fn parse_device(device_str: &str) -> Result<Device>
```

**Fix:** Create `DeviceSpec` enum:

```rust
pub enum DeviceSpec {
    Auto,
    Cpu,
    Cuda(usize),
    Metal,
}
```

### TS-006: Magic token IDs scattered in code

**File:** `src/lib.rs`
**Lines:** 1377

```rust
generation::apply_token_suppression(&logits, 3072, CODEC_EOS_TOKEN_ID)?;
```

**Problem:** Magic number `3072` inline.

**Fix:** Use named constant from tokens module.

______________________________________________________________________

## 8. API Design

### API-001: Inconsistent Option<T> parameter patterns

**File:** `src/lib.rs`
**Lines:** 460, 494-496, 1091-1094

```rust
pub fn synthesize(&self, text: &str, options: Option<SynthesisOptions>)
```

**Fix:** Use `impl Into<Option<SynthesisOptions>>` for ergonomics.

### API-002: Missing builder pattern for SynthesisOptions

**File:** `src/lib.rs`
**Lines:** 1716-1750
**Problem:** Many fields with defaults.

**Fix:** Add builder:

```rust
SynthesisOptions::builder()
    .temperature(0.8)
    .top_k(30)
    .build()
```

### API-003: No prelude module

**Problem:** Users must import many items individually.

**Fix:** Create `src/prelude.rs`:

```rust
pub use crate::{
    Qwen3TTS, SynthesisOptions, Speaker, Language,
    AudioBuffer, auto_device, VoiceClonePrompt,
};
```

### API-004: Inconsistent return types

**Files:** Multiple

```rust
pub fn has_speaker_encoder(&self) -> bool
pub fn supports_voice_cloning(&self) -> bool  // Same meaning, different name
```

**Fix:** Pick one naming convention.

### API-005: Missing async support

**Problem:** All operations are synchronous.

**Fix:** Add optional async feature:

```rust
#[cfg(feature = "async")]
pub async fn synthesize_async(&self, ...) -> Result<AudioBuffer>
```

### API-006: StreamingSession lacks ergonomic methods

**File:** `src/lib.rs`
**Lines:** 1504-1714
**Problem:** Only Iterator impl, no `collect_all()` helper.

**Fix:** Add convenience methods:

```rust
impl StreamingSession<'_> {
    pub fn collect_all(&mut self) -> Result<AudioBuffer> { ... }
    pub fn with_callback<F>(&mut self, f: F) -> Result<()> { ... }
}
```

### API-007: No way to query model info

**Problem:** Users can't easily inspect loaded model.

**Fix:** Add info methods:

```rust
pub fn model_info(&self) -> ModelInfo {
    ModelInfo {
        variant: self.model_type,
        hidden_size: self.talker.config().hidden_size,
        // ...
    }
}
```

### API-008: AudioBuffer lacks common operations

**File:** `src/audio/io.rs`
**Problem:** No trim, concat, resample methods on AudioBuffer.

**Fix:** Add ergonomic methods:

```rust
impl AudioBuffer {
    pub fn trim(&mut self, start: Duration, end: Duration) { ... }
    pub fn concat(&self, other: &AudioBuffer) -> Result<Self> { ... }
    pub fn resample(&self, target_rate: u32) -> Result<Self> { ... }
}
```

### API-009: Constants not exported at crate root

**File:** `src/lib.rs`
**Lines:** 1494-1502
**Problem:** `CODEC_EOS_TOKEN_ID`, `SAMPLES_PER_FRAME` buried in lib.

**Fix:** Re-export in prelude or at crate root.

______________________________________________________________________

## 9. Performance

### PERF-001: Unnecessary tensor clones in loop

**File:** `src/lib.rs`
**Lines:** 591, 733, 944, 1248

```rust
tts_pad_embed.clone()  // Every iteration
```

**Fix:** Clone once before loop.

### PERF-002: Vec allocation in hot path

**File:** `src/generation/sampling.rs`
**Lines:** 141-160, 165-201

```rust
let mut result_data = Vec::with_capacity(batch * vocab);
// ... build new vec every call
```

**Fix:** Pre-allocate buffer or use in-place operations.

### PERF-003: Repeated special embedding computation

**File:** `src/models/talker.rs`
**Lines:** 871-877, 882-888
**Problem:** `get_tts_pad_embed()`, `get_tts_eos_embed()` compute every call.

**Fix:** Cache at construction:

```rust
struct CachedEmbeddings {
    tts_pad: Tensor,
    tts_eos: Tensor,
    role_prefix: Tensor,
}
```

### PERF-004: No mask caching

**File:** `src/models/talker.rs`
**Lines:** 940-955
**Problem:** Causal mask rebuilt every call.

**Fix:** Cache common mask sizes.

### PERF-005: Sequential embedding lookups

**File:** `src/lib.rs`
**Lines:** 1321-1336

```rust
for group in 1..16 {
    let group_embed = self.code_predictor.embed_codes_for_group(...)?;
    summed = summed.add(&group_embed)?;
}
```

**Fix:** Batch into single embedding lookup.

### PERF-006: Lazy component loading not implemented

**File:** `src/lib.rs`
**Problem:** Speaker encoder loaded even if never used.

**Fix:** Use `OnceLock` for lazy initialization.

### PERF-007: No tensor fusion

**File:** `src/models/transformer.rs`
**Problem:** Multiple small tensor ops instead of fused kernels.

**Note:** Candle limitation, but could add custom kernels for hot paths.

### PERF-008: Allocations in multinomial_sample

**File:** `src/generation/sampling.rs`
**Lines:** 225-254
**Problem:** Multiple Vec allocations per sample.

**Fix:** Pre-allocate or use candle's multinomial when available.

______________________________________________________________________

## 10. Documentation

### DOC-001: Missing crate-level examples

**File:** `src/lib.rs`
**Problem:** Doc comments show examples but `cargo test --doc` may not run them.

**Fix:** Add `# Example` sections with `rust,ignore` or proper test setup.

### DOC-002: Undocumented public items

**Files:** Multiple
**Problem:** Some public structs/functions lack doc comments.

**Fix:** Add `/// ...` documentation to all public items.

### DOC-003: Missing CHANGELOG.md

**Problem:** No version history.

**Fix:** Create `CHANGELOG.md` following Keep a Changelog format.

### DOC-004: Internal docs mixed with public docs

**File:** `docs/CONTINUATION.md`, `docs/VALIDATION.md`
**Problem:** Development notes in public docs folder.

**Fix:** Move to `.github/` or internal wiki.

### DOC-005: README examples may be outdated

**File:** `README.md`
**Problem:** No CI to validate README code snippets.

**Fix:** Use `skeptic` or `doc-comment` crate to test README.

______________________________________________________________________

## Implementation Order

Suggested order for fixing (dependencies considered):

### Phase 1: Foundation (High Impact)

1. GMS-001/002/003 - Fix global RNG state
1. DUP-001 - Extract generation loop
1. SOC-001 - Split lib.rs
1. TS-001 - Add token newtypes

### Phase 2: Candle Patterns (Medium Impact)

5. IC-001/002 - Fix VarBuilder usage
1. IC-007 - Fix KVCache cloning
1. DUP-004/005 - Deduplicate mask and embedding creation
1. PERF-003 - Cache special embeddings

### Phase 3: Error Handling (Medium Impact)

9. ERR-001 - Add typed errors
1. ERR-002/003 - Standardize error messages
1. API-002 - Add SynthesisOptions builder

### Phase 4: Polish (Low Impact)

12. IR-011/014 - Add standard trait impls
01. API-003 - Add prelude
01. DOC-001/002 - Complete documentation
01. PERF-001/002 - Optimize hot paths

______________________________________________________________________

## Notes

- Issues are interconnected; fixing SOC-001 enables many other fixes
- GMS-001 is critical for library correctness
- IC-001 alone would simplify many files significantly
- Consider feature flags for breaking changes
