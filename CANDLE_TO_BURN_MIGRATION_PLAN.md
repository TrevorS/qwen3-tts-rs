# Candle to Burn Migration Plan: Qwen3-TTS-RS

## Executive Summary

This document outlines a comprehensive plan to migrate the `qwen3-tts-rs` codebase from the Candle ML framework to Burn. The migration will modernize the inference stack, improve maintainability through Burn's derive macro system, and enable broader backend support including WebGPU.

**Current State**: The codebase uses Candle v0.9 for all tensor operations, neural network layers, and device management across a three-stage TTS pipeline (TalkerModel → CodePredictor → Decoder12Hz).

**Target State**: Idiomatic Burn code with generic backend support, compile-time dimension checking, and the Config/Module derive pattern.

---

## 1. Framework Comparison

### Core Philosophical Differences

| Aspect | Candle | Burn |
|--------|--------|------|
| **Design Philosophy** | Minimalist, PyTorch-like API | Full-featured framework with training infrastructure |
| **Error Handling** | `Result<T, E>` throughout | Panics on errors (simpler API) |
| **Tensor Dimensions** | Runtime checked | Compile-time const generics `Tensor<B, D>` |
| **Module System** | Manual struct + impl | `#[derive(Module)]` macro |
| **Configuration** | Manual constructors | `#[derive(Config)]` with builder pattern |
| **Backend Abstraction** | Direct implementations | Generic `Backend` trait with decorators |
| **Weight Loading** | `VarBuilder` from safetensors | `Record` system with multiple recorders |

### API Mapping

| Candle | Burn |
|--------|------|
| `Tensor` | `Tensor<B, D, K>` |
| `Device::{Cpu, Cuda, Metal}` | `B::Device` (backend-specific) |
| `DType::{F32, BF16, ...}` | `FloatElem<B>`, `IntElem<B>` |
| `tensor.reshape((a, b))?` | `tensor.reshape([a, b])` |
| `tensor.transpose(0, 1)?` | `tensor.swap_dims(0, 1)` |
| `tensor.i((.., 0, ..))?` | `tensor.slice([0..batch, 0..1, 0..seq])` |
| `tensor.matmul(&other)?` | `tensor.matmul(other)` |
| `Tensor::cat(&[&a, &b], dim)?` | `Tensor::cat(vec![a, b], dim)` |
| `tensor.to_dtype(DType::F32)?` | `tensor.float()` or backend handles |
| `candle_nn::linear(...)` | `LinearConfig::new(...).init(device)` |
| `candle_nn::rms_norm(...)` | `RmsNormConfig::new(...).init(device)` |
| `candle_nn::ops::silu(&x)?` | `activation::silu(x)` |
| `candle_nn::ops::softmax_last_dim(&x)?` | `activation::softmax(x, D-1)` |

---

## 2. Migration Strategy

### Phase Overview

```
Phase 1: Foundation & Infrastructure (Week 1-2)
    ├── Set up Burn dependencies and feature flags
    ├── Create backend abstraction layer
    ├── Implement Config structs for all models
    └── Port tensor utility functions

Phase 2: Core Layers (Week 2-3)
    ├── Port transformer building blocks (Attention, MLP, RoPE)
    ├── Implement KVCache for Burn
    └── Port normalization layers

Phase 3: Model Components (Week 3-5)
    ├── Port TalkerModel
    ├── Port CodePredictor
    ├── Port Decoder12Hz codec
    ├── Port SpeakerEncoder (ECAPA-TDNN)
    └── Port Encoder12Hz (Mimi integration)

Phase 4: Inference Pipeline (Week 5-6)
    ├── Port generation/sampling logic
    ├── Port streaming session
    └── Integrate audio processing

Phase 5: Testing & Optimization (Week 6-7)
    ├── Validate output parity with Candle version
    ├── Performance benchmarking
    └── Backend-specific optimizations
```

### Guiding Principles

1. **Backend-Generic Code**: All model code should be generic over `B: Backend`
2. **Compile-Time Safety**: Leverage const generic dimensions where possible
3. **Config-Driven**: Use `#[derive(Config)]` for all configurable components
4. **Incremental Migration**: Port component by component with tests at each step
5. **Preserve API**: Keep the public API (`synthesize()`, `stream()`, etc.) unchanged

---

## 3. Dependency Changes

### Cargo.toml Updates

```toml
[dependencies]
# Remove Candle dependencies
# candle-core = "0.9"
# candle-nn = "0.9"
# candle-transformers = "0.9"
# candle-flash-attn = { version = "0.9", optional = true }

# Add Burn dependencies
burn = { version = "0.16", features = ["std"] }
burn-ndarray = { version = "0.16", optional = true }
burn-tch = { version = "0.16", optional = true }
burn-wgpu = { version = "0.16", optional = true }
burn-cuda = { version = "0.16", optional = true }
burn-import = { version = "0.16" }  # For safetensors loading

# Keep existing
tokenizers = "0.22"
hound = "3.5"
rubato = "1.0"
rustfft = "6.2"
safetensors = "0.7"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
rayon = "1.10"

[features]
default = ["ndarray"]
ndarray = ["dep:burn-ndarray"]
tch = ["dep:burn-tch"]
wgpu = ["dep:burn-wgpu"]
cuda = ["dep:burn-cuda"]
```

---

## 4. Detailed Component Migration

### 4.1 Foundation Layer

#### 4.1.1 Backend Type Aliases (`src/backend.rs`)

```rust
use burn::prelude::*;

#[cfg(feature = "ndarray")]
pub type DefaultBackend = burn_ndarray::NdArray<f32>;

#[cfg(feature = "tch")]
pub type DefaultBackend = burn_tch::TchBackend<f32>;

#[cfg(feature = "wgpu")]
pub type DefaultBackend = burn_wgpu::Wgpu;

#[cfg(feature = "cuda")]
pub type DefaultBackend = burn_cuda::Cuda;

/// Get default device for the configured backend
pub fn default_device<B: Backend>() -> B::Device {
    Default::default()
}
```

#### 4.1.2 Config Structs (`src/models/config.rs`)

**Current Candle:**
```rust
pub struct Qwen3TTSConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    // ... more fields
}
```

**Burn Migration:**
```rust
use burn::config::Config;

#[derive(Config, Debug, Clone)]
pub struct Qwen3TTSConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub codec_vocab_size: usize,
    #[config(default = "1e-6")]
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub mrope_section: Vec<usize>,
}

#[derive(Config, Debug, Clone)]
pub struct CodePredictorConfig {
    pub hidden_size: usize,
    #[config(default = "5")]
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub codec_vocab_size: usize,
    #[config(default = "15")]
    pub num_codebooks: usize,
}
```

### 4.2 Transformer Building Blocks

#### 4.2.1 RMSNorm

**Current Candle (`transformer.rs`):**
```rust
use candle_nn::{rms_norm, RmsNorm};

let norm = rms_norm(hidden_size, eps, vb.pp("norm"))?;
let output = norm.forward(&hidden_states)?;
```

**Burn Migration:**
```rust
use burn::nn::rms_norm::{RmsNorm, RmsNormConfig};

#[derive(Module, Debug)]
pub struct MyRmsNorm<B: Backend> {
    norm: RmsNorm<B>,
}

impl<B: Backend> MyRmsNorm<B> {
    pub fn new(hidden_size: usize, eps: f64, device: &B::Device) -> Self {
        Self {
            norm: RmsNormConfig::new(hidden_size)
                .with_epsilon(eps)
                .init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.norm.forward(x)
    }
}
```

#### 4.2.2 Linear Layers

**Current Candle:**
```rust
use candle_nn::{linear, linear_no_bias, Linear};

let proj = linear_no_bias(in_features, out_features, vb.pp("proj"))?;
let output = proj.forward(&input)?;
```

**Burn Migration:**
```rust
use burn::nn::{Linear, LinearConfig};

#[derive(Module, Debug)]
pub struct Projection<B: Backend> {
    proj: Linear<B>,
}

impl<B: Backend> Projection<B> {
    pub fn new(in_features: usize, out_features: usize, device: &B::Device) -> Self {
        Self {
            proj: LinearConfig::new(in_features, out_features)
                .with_bias(false)
                .init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.proj.forward(x)
    }
}
```

#### 4.2.3 Embedding

**Current Candle:**
```rust
use candle_nn::{embedding, Embedding};

let embed = embedding(vocab_size, embed_dim, vb.pp("embed"))?;
let output = embed.forward(&input_ids)?;
```

**Burn Migration:**
```rust
use burn::nn::{Embedding, EmbeddingConfig};

#[derive(Module, Debug)]
pub struct TokenEmbedding<B: Backend> {
    embed: Embedding<B>,
}

impl<B: Backend> TokenEmbedding<B> {
    pub fn new(vocab_size: usize, embed_dim: usize, device: &B::Device) -> Self {
        Self {
            embed: EmbeddingConfig::new(vocab_size, embed_dim).init(device),
        }
    }

    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.embed.forward(input_ids)
    }
}
```

#### 4.2.4 Rotary Position Embedding (RoPE)

**Current Candle:**
```rust
pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, theta: f64, device: &Device) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::new(inv_freq, device)?;
        let positions = Tensor::arange(0u32, max_seq_len as u32, device)?
            .to_dtype(DType::F32)?;
        let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        Ok(Self {
            cos: freqs.cos()?,
            sin: freqs.sin()?,
        })
    }

    pub fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;
        let cos = self.cos.i(offset..offset + seq_len)?;
        let sin = self.sin.i(offset..offset + seq_len)?;
        let q_rotated = apply_rope_rotation(q, &cos, &sin)?;
        let k_rotated = apply_rope_rotation(k, &cos, &sin)?;
        Ok((q_rotated, k_rotated))
    }
}
```

**Burn Migration:**
```rust
use burn::prelude::*;
use burn::nn::RotaryEncoding;  // Burn has built-in support!

#[derive(Module, Debug)]
pub struct RotaryEmbedding<B: Backend> {
    rotary: RotaryEncoding<B>,
}

impl<B: Backend> RotaryEmbedding<B> {
    pub fn new(
        dim: usize,
        max_seq_len: usize,
        theta: f64,
        device: &B::Device,
    ) -> Self {
        Self {
            rotary: RotaryEncodingConfig::new(max_seq_len, dim)
                .with_theta(theta)
                .init(device),
        }
    }

    pub fn apply(
        &self,
        q: Tensor<B, 4>,  // [batch, heads, seq, head_dim]
        k: Tensor<B, 4>,
        offset: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        // Burn's RotaryEncoding expects [batch, seq, heads, head_dim]
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);

        let q_rotated = self.rotary.forward(q, offset);
        let k_rotated = self.rotary.forward(k, offset);

        (q_rotated.swap_dims(1, 2), k_rotated.swap_dims(1, 2))
    }
}
```

**Note**: For MRoPE (multimodal RoPE with interleaved sections [24, 20, 20]), we'll need a custom implementation since Burn's built-in RotaryEncoding doesn't support section-based interleaving.

#### 4.2.5 KVCache

**Current Candle:**
```rust
pub struct KVCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
}

impl KVCache {
    pub fn update_k(&mut self, k: &Tensor) -> Result<Tensor> {
        let k = if let Some(prev_k) = &self.k {
            Tensor::cat(&[prev_k, k], 2)?
        } else {
            k.clone()
        };
        self.k = Some(k.clone());
        Ok(k)
    }
}
```

**Burn Migration:**
```rust
#[derive(Debug, Clone)]
pub struct KVCache<B: Backend> {
    k: Option<Tensor<B, 4>>,  // [batch, heads, seq, head_dim]
    v: Option<Tensor<B, 4>>,
}

impl<B: Backend> KVCache<B> {
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    pub fn update_k(&mut self, k: Tensor<B, 4>) -> Tensor<B, 4> {
        let k = match &self.k {
            Some(prev_k) => Tensor::cat(vec![prev_k.clone(), k], 2),
            None => k,
        };
        self.k = Some(k.clone());
        k
    }

    pub fn update_v(&mut self, v: Tensor<B, 4>) -> Tensor<B, 4> {
        let v = match &self.v {
            Some(prev_v) => Tensor::cat(vec![prev_v.clone(), v], 2),
            None => v,
        };
        self.v = Some(v.clone());
        v
    }

    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }

    pub fn offset(&self) -> usize {
        self.k.as_ref().map(|k| k.dims()[2]).unwrap_or(0)
    }
}
```

#### 4.2.6 Multi-Head Attention with GQA

**Current Candle (simplified):**
```rust
pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
}
```

**Burn Migration:**
```rust
use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};
use burn::nn::rms_norm::{RmsNorm, RmsNormConfig};

#[derive(Config, Debug)]
pub struct AttentionConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    #[config(default = "1e-6")]
    pub rms_norm_eps: f64,
}

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    q_norm: RmsNorm<B>,
    k_norm: RmsNorm<B>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl AttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Attention<B> {
        let hidden_size = self.hidden_size;
        let head_dim = self.head_dim;

        Attention {
            q_proj: LinearConfig::new(hidden_size, self.num_heads * head_dim)
                .with_bias(false)
                .init(device),
            k_proj: LinearConfig::new(hidden_size, self.num_kv_heads * head_dim)
                .with_bias(false)
                .init(device),
            v_proj: LinearConfig::new(hidden_size, self.num_kv_heads * head_dim)
                .with_bias(false)
                .init(device),
            o_proj: LinearConfig::new(self.num_heads * head_dim, hidden_size)
                .with_bias(false)
                .init(device),
            q_norm: RmsNormConfig::new(head_dim)
                .with_epsilon(self.rms_norm_eps)
                .init(device),
            k_norm: RmsNormConfig::new(head_dim)
                .with_epsilon(self.rms_norm_eps)
                .init(device),
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        }
    }
}

impl<B: Backend> Attention<B> {
    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,  // [batch, seq, hidden]
        rope: &RotaryEmbedding<B>,
        attention_mask: Option<Tensor<B, 4>>,
        kv_cache: Option<&mut KVCache<B>>,
        offset: usize,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _] = hidden_states.dims();

        // Project Q, K, V
        let q = self.q_proj.forward(hidden_states.clone());
        let k = self.k_proj.forward(hidden_states.clone());
        let v = self.v_proj.forward(hidden_states);

        // Reshape to [batch, seq, heads, head_dim]
        let q = q.reshape([batch, seq_len, self.num_heads, self.head_dim]);
        let k = k.reshape([batch, seq_len, self.num_kv_heads, self.head_dim]);
        let v = v.reshape([batch, seq_len, self.num_kv_heads, self.head_dim]);

        // Apply QK normalization (per-head)
        let q = self.apply_head_norm(&self.q_norm, q);
        let k = self.apply_head_norm(&self.k_norm, k);

        // Transpose to [batch, heads, seq, head_dim]
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        // Apply rotary embeddings
        let (q, k) = rope.apply(q, k, offset);

        // Update KV cache if provided
        let (k, v) = match kv_cache {
            Some(cache) => {
                let k = cache.update_k(k);
                let v = cache.update_v(v);
                (k, v)
            }
            None => (k, v),
        };

        // Repeat KV heads for GQA
        let k = self.repeat_kv(k);
        let v = self.repeat_kv(v);

        // Scaled dot-product attention
        let attn_weights = q.matmul(k.swap_dims(2, 3)) * self.scale;

        let attn_weights = match attention_mask {
            Some(mask) => attn_weights + mask,
            None => attn_weights,
        };

        let attn_weights = burn::tensor::activation::softmax(attn_weights, 3);
        let attn_output = attn_weights.matmul(v);

        // Reshape back to [batch, seq, hidden]
        let attn_output = attn_output
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.num_heads * self.head_dim]);

        self.o_proj.forward(attn_output)
    }

    fn repeat_kv(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let n_rep = self.num_heads / self.num_kv_heads;
        if n_rep == 1 {
            return x;
        }
        let [batch, num_kv_heads, seq_len, head_dim] = x.dims();
        x.unsqueeze_dim(2)
            .repeat(&[1, 1, n_rep, 1, 1])
            .reshape([batch, self.num_heads, seq_len, head_dim])
    }

    fn apply_head_norm(&self, norm: &RmsNorm<B>, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // x: [batch, seq, heads, head_dim]
        let [batch, seq, heads, head_dim] = x.dims();
        let x = x.reshape([batch * seq * heads, head_dim]);
        let x = norm.forward(x.unsqueeze_dim(1)).squeeze(1);
        x.reshape([batch, seq, heads, head_dim])
    }
}
```

#### 4.2.7 MLP with SwiGLU

**Current Candle:**
```rust
pub struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MLP {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let gate = candle_nn::ops::silu(&gate)?;
        let up = self.up_proj.forward(x)?;
        Ok(self.down_proj.forward(&(gate * up)?)?)
    }
}
```

**Burn Migration:**
```rust
use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::silu;

#[derive(Config, Debug)]
pub struct MLPConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
}

#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    gate_proj: Linear<B>,
    up_proj: Linear<B>,
    down_proj: Linear<B>,
}

impl MLPConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MLP<B> {
        MLP {
            gate_proj: LinearConfig::new(self.hidden_size, self.intermediate_size)
                .with_bias(false)
                .init(device),
            up_proj: LinearConfig::new(self.hidden_size, self.intermediate_size)
                .with_bias(false)
                .init(device),
            down_proj: LinearConfig::new(self.intermediate_size, self.hidden_size)
                .with_bias(false)
                .init(device),
        }
    }
}

impl<B: Backend> MLP<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}
```

#### 4.2.8 Decoder Layer

**Burn Migration:**
```rust
#[derive(Config, Debug)]
pub struct DecoderLayerConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    #[config(default = "1e-6")]
    pub rms_norm_eps: f64,
}

#[derive(Module, Debug)]
pub struct DecoderLayer<B: Backend> {
    self_attn: Attention<B>,
    mlp: MLP<B>,
    input_layernorm: RmsNorm<B>,
    post_attention_layernorm: RmsNorm<B>,
}

impl DecoderLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DecoderLayer<B> {
        DecoderLayer {
            self_attn: AttentionConfig {
                hidden_size: self.hidden_size,
                num_heads: self.num_heads,
                num_kv_heads: self.num_kv_heads,
                head_dim: self.head_dim,
                rms_norm_eps: self.rms_norm_eps,
            }.init(device),
            mlp: MLPConfig {
                hidden_size: self.hidden_size,
                intermediate_size: self.intermediate_size,
            }.init(device),
            input_layernorm: RmsNormConfig::new(self.hidden_size)
                .with_epsilon(self.rms_norm_eps)
                .init(device),
            post_attention_layernorm: RmsNormConfig::new(self.hidden_size)
                .with_epsilon(self.rms_norm_eps)
                .init(device),
        }
    }
}

impl<B: Backend> DecoderLayer<B> {
    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        attention_mask: Option<Tensor<B, 4>>,
        kv_cache: Option<&mut KVCache<B>>,
        offset: usize,
    ) -> Tensor<B, 3> {
        // Self-attention with residual
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states);
        let hidden_states = self.self_attn.forward(
            hidden_states,
            rope,
            attention_mask.clone(),
            kv_cache,
            offset,
        );
        let hidden_states = residual + hidden_states;

        // MLP with residual
        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(hidden_states);
        let hidden_states = self.mlp.forward(hidden_states);
        residual + hidden_states
    }
}
```

### 4.3 Model Components

#### 4.3.1 TalkerModel

The TalkerModel is the largest component with 28 transformer layers. Key migration considerations:

```rust
#[derive(Config, Debug)]
pub struct TalkerModelConfig {
    pub qwen3_config: Qwen3TTSConfig,
    pub model_type: ModelType,  // Base, CustomVoice, VoiceDesign
}

#[derive(Module, Debug)]
pub struct TalkerModel<B: Backend> {
    text_embed_tokens: Embedding<B>,
    codec_embed_tokens: Embedding<B>,
    text_projection: Option<TextProjection<B>>,  // SwiGLU
    layers: Vec<DecoderLayer<B>>,
    norm: RmsNorm<B>,
    codec_head: Linear<B>,
    rope: MRoPE<B>,  // Custom MRoPE implementation needed
    config: Qwen3TTSConfig,
}
```

**Special Handling Required:**
1. **MRoPE**: Custom implementation for interleaved [24, 20, 20] sections
2. **Dual Embeddings**: Text vocab (151936) + Codec vocab (3072)
3. **KV Cache Management**: Per-layer cache for autoregressive generation
4. **Multiple Prefill Variants**: `prefill_custom_voice()`, `prefill_voice_clone()`, `prefill_voice_design()`

#### 4.3.2 CodePredictor

```rust
#[derive(Module, Debug)]
pub struct CodePredictor<B: Backend> {
    codec_embeddings: Vec<Embedding<B>>,  // 15 embeddings
    small_to_mtp_projection: Option<Linear<B>>,
    layers: Vec<DecoderLayer<B>>,  // 5 layers
    norm: RmsNorm<B>,
    lm_heads: Vec<Linear<B>>,  // 15 heads
    rope: RoPE<B>,
    config: CodePredictorConfig,
}
```

#### 4.3.3 Decoder12Hz (Codec)

**Current Candle uses custom Conv1d/ConvTranspose1d layers:**

```rust
#[derive(Module, Debug)]
pub struct Decoder12Hz<B: Backend> {
    embeddings: Vec<Embedding<B>>,  // 16 codebook embeddings
    prenet: Linear<B>,
    layers: Vec<ConvNeXtBlock<B>>,
    norm: LayerNorm<B>,
    final_conv: ConvTranspose1d<B>,
}
```

**Custom Layers Needed:**
1. `ConvNeXtBlock` - Depthwise separable conv + pointwise conv
2. `SnakeBeta` activation - `x + (1/beta) * sin²(beta * x)`
3. Transposed convolutions with specific padding

#### 4.3.4 SpeakerEncoder (ECAPA-TDNN)

```rust
#[derive(Module, Debug)]
pub struct SpeakerEncoder<B: Backend> {
    layer1: TimeDelayNetBlock<B>,
    layer2: Res2NetBlock<B>,
    layer3: Res2NetBlock<B>,
    layer4: Res2NetBlock<B>,
    mfa: Linear<B>,  // Multi-layer Feature Aggregation
    asp: AttentiveStatisticsPooling<B>,
    asp_bn: BatchNorm1d<B>,
    fc: Linear<B>,
}
```

**Custom Components:**
1. `TimeDelayNetBlock` - Conv1d with reflect padding
2. `Res2NetBlock` - Cascaded TDNNs with scale splits
3. `SqueezeExcitationBlock` - Channel attention
4. `AttentiveStatisticsPooling` - Attention-weighted mean/std

### 4.4 Weight Loading Strategy

#### SafeTensors Loading

```rust
use burn_import::safetensors::SafetensorsFileRecorder;
use burn::record::FullPrecisionSettings;

pub fn load_talker_weights<B: Backend>(
    model: TalkerModel<B>,
    weights_path: &Path,
    device: &B::Device,
) -> TalkerModel<B> {
    // Option 1: Direct SafeTensors loading (if weight names match)
    let recorder = SafetensorsFileRecorder::<FullPrecisionSettings>::default();
    let record = recorder.load(weights_path.into(), device)
        .expect("Failed to load weights");
    model.load_record(record)
}

// Option 2: Manual weight mapping for complex cases
pub fn load_with_remapping<B: Backend>(
    config: &TalkerModelConfig,
    weights: HashMap<String, TensorData>,
    device: &B::Device,
) -> TalkerModel<B> {
    // Build model
    let mut model = config.init(device);

    // Manual weight assignment using tensor data
    // This approach needed when Candle weight names don't match Burn's expected names

    model
}
```

**Weight Name Mapping** (Candle → Burn):

| Candle Path | Burn Path |
|-------------|-----------|
| `talker.model.embed_tokens.weight` | `text_embed_tokens.weight` |
| `talker.model.layers.0.self_attn.q_proj.weight` | `layers.0.self_attn.q_proj.weight` |
| `talker.model.layers.0.input_layernorm.weight` | `layers.0.input_layernorm.weight` |
| `talker.model.norm.weight` | `norm.weight` |

### 4.5 Inference Pipeline

#### 4.5.1 Sampling Logic

**Current Candle:**
```rust
pub fn sample_top_k_top_p(
    logits: &Tensor,
    top_k: usize,
    top_p: f64,
    temperature: f64,
    rng: &mut impl Rng,
) -> Result<u32> {
    let logits = (logits / temperature)?;
    // ... sampling logic
}
```

**Burn Migration:**
```rust
pub fn sample_top_k_top_p<B: Backend>(
    logits: Tensor<B, 2>,  // [batch=1, vocab]
    top_k: usize,
    top_p: f64,
    temperature: f64,
    rng: &mut impl Rng,
) -> u32 {
    let logits = logits / temperature;

    // Convert to Vec for sampling (need to pull to CPU)
    let logits_vec: Vec<f32> = logits.into_data().to_vec().unwrap();

    // Top-K filtering
    let mut indexed: Vec<(usize, f32)> = logits_vec.iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.truncate(top_k);

    // Top-P filtering
    let max_logit = indexed[0].1;
    let probs: Vec<f64> = indexed.iter()
        .map(|(_, v)| ((v - max_logit) as f64).exp())
        .collect();
    let sum: f64 = probs.iter().sum();
    let probs: Vec<f64> = probs.iter().map(|p| p / sum).collect();

    // Nucleus sampling
    let mut cumsum = 0.0;
    let mut cutoff_idx = probs.len();
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum >= top_p {
            cutoff_idx = i + 1;
            break;
        }
    }

    // Sample from filtered distribution
    let filtered: Vec<(usize, f64)> = indexed.iter()
        .zip(probs.iter())
        .take(cutoff_idx)
        .map(|((idx, _), &p)| (*idx, p))
        .collect();

    let sum: f64 = filtered.iter().map(|(_, p)| p).sum();
    let r: f64 = rng.gen::<f64>() * sum;

    let mut cumsum = 0.0;
    for (idx, p) in filtered {
        cumsum += p;
        if cumsum >= r {
            return idx as u32;
        }
    }

    filtered.last().map(|(idx, _)| *idx as u32).unwrap_or(0)
}
```

#### 4.5.2 Main Synthesis Loop

```rust
impl<B: Backend> Qwen3TTS<B> {
    pub fn synthesize(
        &mut self,
        text: &str,
        speaker: Speaker,
        language: Language,
        config: &GenerationConfig,
    ) -> AudioBuffer {
        // 1. Tokenize
        let input_ids = self.tokenizer.encode(text);
        let input_ids = Tensor::<B, 2, Int>::from_data(
            TensorData::from(input_ids.as_slice()),
            &self.device,
        );

        // 2. Prefill
        let (mut last_hidden, mut logits) = self.talker.prefill_custom_voice(
            input_ids,
            speaker,
            language,
            &mut self.talker_kv_caches,
        );

        // 3. Sample first token
        let mut current_token = sample_top_k_top_p(
            logits,
            config.top_k,
            config.top_p,
            config.temperature,
            &mut self.rng,
        );

        // 4. Autoregressive loop
        let mut all_codes = Vec::new();
        for _ in 0..config.max_new_tokens {
            if current_token == config.eos_token_id.unwrap_or(2150) {
                break;
            }

            // Generate acoustic codes
            let acoustic_codes = self.code_predictor.generate_acoustic_codes(
                last_hidden.clone(),
                &mut self.predictor_kv_caches,
            );

            // Combine semantic + acoustic
            let frame_codes = std::iter::once(current_token)
                .chain(acoustic_codes)
                .collect::<Vec<_>>();
            all_codes.push(frame_codes);

            // Get embeddings for next step
            let code_embed = self.code_predictor.get_acoustic_embeddings_sum(
                &acoustic_codes,
                current_token,
            );

            // Step talker
            (last_hidden, logits) = self.talker.generate_step_with_embed(
                code_embed,
                &mut self.talker_kv_caches,
            );

            current_token = sample_top_k_top_p(
                logits,
                config.top_k,
                config.top_p,
                config.temperature,
                &mut self.rng,
            );
        }

        // 5. Decode to audio
        let codes_tensor = self.codes_to_tensor(&all_codes);
        let waveform = self.decoder.forward(codes_tensor);

        self.tensor_to_audio(waveform)
    }
}
```

---

## 5. Custom Components Requiring Special Implementation

### 5.1 MRoPE (Multimodal RoPE)

The Qwen3-TTS model uses MRoPE with interleaved sections [24, 20, 20] for 3D positions. This requires a custom implementation:

```rust
#[derive(Module, Debug)]
pub struct MRoPE<B: Backend> {
    inv_freq: Tensor<B, 1>,
    sections: [usize; 3],  // [24, 20, 20]
    device: B::Device,
}

impl<B: Backend> MRoPE<B> {
    pub fn new(head_dim: usize, theta: f64, sections: [usize; 3], device: &B::Device) -> Self {
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / theta.powf(i as f64 / head_dim as f64) as f32)
            .collect();

        Self {
            inv_freq: Tensor::from_floats(inv_freq.as_slice(), device),
            sections,
            device: device.clone(),
        }
    }

    pub fn apply(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        positions: Tensor<B, 2>,  // [batch, seq] or 3D positions
        offset: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        // Split head_dim into sections
        // Apply different position encodings to each section
        // Interleave the results
        todo!("Custom MRoPE implementation")
    }
}
```

### 5.2 SnakeBeta Activation

```rust
pub fn snake_beta<B: Backend>(x: Tensor<B, 3>, beta: f32) -> Tensor<B, 3> {
    // x + (1/beta) * sin²(beta * x)
    let scaled = x.clone() * beta;
    let sin_squared = scaled.sin().powf_scalar(2.0);
    x + sin_squared / beta
}
```

### 5.3 Reflect Padding for Conv1d

```rust
pub fn reflect_pad_1d<B: Backend>(
    x: Tensor<B, 3>,  // [batch, channels, length]
    pad_left: usize,
    pad_right: usize,
) -> Tensor<B, 3> {
    let [batch, channels, length] = x.dims();

    if pad_left == 0 && pad_right == 0 {
        return x;
    }

    // Extract left padding (reversed)
    let left_pad = if pad_left > 0 {
        let indices: Vec<i64> = (1..=pad_left as i64).rev().collect();
        let indices = Tensor::<B, 1, Int>::from_ints(indices.as_slice(), x.device());
        x.clone().select(2, indices).flip([2])
    } else {
        Tensor::empty([batch, channels, 0], x.device())
    };

    // Extract right padding (reversed)
    let right_pad = if pad_right > 0 {
        let start = length as i64 - pad_right as i64 - 1;
        let indices: Vec<i64> = (start..length as i64 - 1).rev().collect();
        let indices = Tensor::<B, 1, Int>::from_ints(indices.as_slice(), x.device());
        x.clone().select(2, indices).flip([2])
    } else {
        Tensor::empty([batch, channels, 0], x.device())
    };

    Tensor::cat(vec![left_pad, x, right_pad], 2)
}
```

### 5.4 Causal Mask Creation

```rust
pub fn create_causal_mask<B: Backend>(
    seq_len: usize,
    offset: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let total_len = offset + seq_len;
    let mut mask_data = vec![0.0f32; seq_len * total_len];

    for i in 0..seq_len {
        for j in 0..total_len {
            if j > offset + i {
                mask_data[i * total_len + j] = f32::NEG_INFINITY;
            }
        }
    }

    Tensor::from_floats(mask_data.as_slice(), device)
        .reshape([1, 1, seq_len, total_len])
}
```

---

## 6. Testing Strategy

### 6.1 Unit Tests (Per Component)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_attention_forward() {
        let device = Default::default();
        let config = AttentionConfig {
            hidden_size: 1024,
            num_heads: 16,
            num_kv_heads: 8,
            head_dim: 64,
            rms_norm_eps: 1e-6,
        };
        let attn = config.init::<TestBackend>(&device);

        let input = Tensor::random([1, 10, 1024], Distribution::Normal(0.0, 1.0), &device);
        let rope = RotaryEmbedding::new(64, 2048, 10000.0, &device);

        let output = attn.forward(input.clone(), &rope, None, None, 0);

        assert_eq!(output.dims(), [1, 10, 1024]);
    }

    #[test]
    fn test_mlp_forward() {
        let device = Default::default();
        let config = MLPConfig {
            hidden_size: 1024,
            intermediate_size: 4096,
        };
        let mlp = config.init::<TestBackend>(&device);

        let input = Tensor::random([1, 10, 1024], Distribution::Normal(0.0, 1.0), &device);
        let output = mlp.forward(input);

        assert_eq!(output.dims(), [1, 10, 1024]);
    }
}
```

### 6.2 Integration Tests (Model Outputs)

```rust
#[test]
fn test_talker_prefill() {
    let device = Default::default();
    let config = TalkerModelConfig::default();
    let model = config.init::<TestBackend>(&device);

    // Load test weights
    let model = load_test_weights(model, &device);

    let input_ids = Tensor::<TestBackend, 2, Int>::from_ints(&[[1, 2, 3, 4, 5]], &device);
    let (hidden, logits) = model.prefill(input_ids);

    assert_eq!(hidden.dims(), [1, 1, 1024]);
    assert_eq!(logits.dims(), [1, 1, 3072]);
}
```

### 6.3 Reference Validation Tests

Compare outputs between Candle and Burn implementations:

```rust
#[test]
fn test_output_parity_with_candle() {
    // Load reference outputs from Candle version
    let reference_hidden: Vec<f32> = load_reference("test_data/hidden_states.bin");
    let reference_logits: Vec<f32> = load_reference("test_data/logits.bin");

    // Run Burn model
    let (hidden, logits) = run_burn_model();

    // Compare with tolerance
    let hidden_vec: Vec<f32> = hidden.into_data().to_vec().unwrap();
    for (a, b) in hidden_vec.iter().zip(reference_hidden.iter()) {
        assert!((a - b).abs() < 1e-4, "Hidden state mismatch");
    }
}
```

### 6.4 End-to-End Tests

```rust
#[test]
fn test_full_synthesis() {
    let mut tts = Qwen3TTS::<TestBackend>::load("path/to/model", &device);

    let audio = tts.synthesize(
        "Hello world",
        Speaker::Default,
        Language::English,
        &GenerationConfig::default(),
    );

    assert!(audio.samples.len() > 0);
    assert_eq!(audio.sample_rate, 24000);
}
```

---

## 7. Performance Considerations

### 7.1 Backend Selection Guidelines

| Use Case | Recommended Backend |
|----------|---------------------|
| Development/Testing | NdArray (pure Rust, easy debugging) |
| NVIDIA GPU | CUDA backend or Tch (LibTorch) |
| Apple Silicon | Tch with MPS or WGPU |
| WebAssembly | WGPU (WebGPU) or Candle backend |
| Cross-platform | WGPU (Vulkan/Metal/DX12) |

### 7.2 Kernel Fusion

For WGPU and CUDA backends, enable kernel fusion:

```rust
type OptimizedBackend = Fusion<Wgpu>;
```

### 7.3 Memory Optimization

1. **KV Cache Preallocation**: Allocate max sequence length upfront
2. **Streaming**: Process audio in chunks to limit memory
3. **Weight Sharing**: Decoder weights shared across inference sessions

### 7.4 Flash Attention

Burn doesn't have built-in Flash Attention, but can use:
- Tch backend with PyTorch's native Flash Attention
- Custom CUDA kernel integration (advanced)

---

## 8. Migration Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Numerical Differences** | Audio quality changes | Extensive reference testing, tolerance tuning |
| **Missing Mimi Codec** | Can't use candle_transformers::mimi | Port Mimi to Burn or keep Candle for encoder only |
| **Performance Regression** | Slower inference | Benchmark at each phase, optimize hot paths |
| **Weight Loading Issues** | Model fails to load | Build weight remapping utility, validate shapes |
| **Custom Op Implementation** | Bugs in MRoPE, SnakeBeta | Port carefully with unit tests, compare against PyTorch |

---

## 9. File-by-File Migration Checklist

### Phase 1: Foundation
- [ ] `src/backend.rs` - New file for backend abstraction
- [ ] `Cargo.toml` - Update dependencies
- [ ] `src/models/config.rs` - Convert to `#[derive(Config)]`

### Phase 2: Core Layers
- [ ] `src/models/transformer.rs` - Port all transformer components
  - [ ] `RotaryEmbedding`
  - [ ] `MRoPE`
  - [ ] `KVCache`
  - [ ] `Attention`
  - [ ] `MLP`
  - [ ] `DecoderLayer`
  - [ ] `create_causal_mask`

### Phase 3: Models
- [ ] `src/models/talker.rs` - Port TalkerModel
- [ ] `src/models/code_predictor.rs` - Port CodePredictor
- [ ] `src/models/speaker.rs` - Port SpeakerEncoder
- [ ] `src/models/codec/` - Port all codec components
  - [ ] `decoder_12hz.rs`
  - [ ] `encoder_12hz.rs`
  - [ ] `causal_conv.rs`
  - [ ] `causal_trans_conv.rs`
  - [ ] `convnext_block.rs`
  - [ ] `quantizer.rs`
  - [ ] `snake_beta.rs`
  - [ ] `decoder_block.rs`

### Phase 4: Inference
- [ ] `src/generation/sampling.rs` - Port sampling logic
- [ ] `src/generation/tts.rs` - Port TTS-specific token handling
- [ ] `src/lib.rs` - Port main synthesis API

### Phase 5: CLI & Tests
- [ ] `src/bin/generate_audio.rs` - Update CLI
- [ ] `tests/integration.rs` - Port integration tests
- [ ] `tests/reference_validation.rs` - Add parity tests

---

## 10. Estimated Effort

| Phase | Components | Estimated LOC | Complexity |
|-------|------------|---------------|------------|
| 1. Foundation | Config, Backend | ~500 | Low |
| 2. Core Layers | Transformer blocks | ~1,500 | Medium |
| 3. Models | All model components | ~3,000 | High |
| 4. Inference | Sampling, pipeline | ~800 | Medium |
| 5. Testing | All tests | ~1,000 | Medium |

**Total Estimated Effort**: ~6,800 lines of Rust code

---

## 11. Open Questions

1. **Mimi Codec Dependency**: Should we port Mimi to Burn or keep Candle for the encoder only?
2. **Flash Attention**: How critical is Flash Attention performance? Worth custom kernel?
3. **Backward Compatibility**: Should we maintain Candle as an optional backend during transition?
4. **WebAssembly Priority**: Is WASM deployment a near-term goal affecting backend choice?

---

## 12. Conclusion

Migrating from Candle to Burn offers:
- **Better type safety** through compile-time dimension checking
- **Cleaner model definitions** via derive macros
- **Broader backend support** including WebGPU
- **Training capability** if fine-tuning is ever desired

The migration is substantial but well-structured. The modular architecture of the existing codebase maps cleanly to Burn's patterns. Success depends on careful attention to numerical parity and thorough testing at each phase.

**Recommended Approach**: Incremental migration with reference tests, starting with the simpler components (MLP, normalization) before tackling the complex attention and model orchestration layers.
