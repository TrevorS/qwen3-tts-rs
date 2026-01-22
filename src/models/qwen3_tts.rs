//! Qwen3-TTS main model implementation
//!
//! This implements the "Talker" model that generates audio codec tokens
//! from text input autoregressively.

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{embedding, linear, linear_no_bias, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};
use std::sync::Arc;

use super::config::Qwen3TTSConfig;
use crate::generation::GenerationConfig;

/// Rotary position embedding
pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    dim: usize,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, theta: f64, device: &Device) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / (theta as f32).powf(i as f32 / dim as f32))
            .collect();

        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;
        let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?.unsqueeze(1)?;

        let freqs = positions.matmul(&inv_freq.unsqueeze(0)?)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        Ok(Self { cos, sin, dim })
    }

    pub fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;
        let cos = self.cos.i(offset..offset + seq_len)?;
        let sin = self.sin.i(offset..offset + seq_len)?;

        let q_rot = self.rotate(q, &cos, &sin)?;
        let k_rot = self.rotate(k, &cos, &sin)?;

        Ok((q_rot, k_rot))
    }

    fn rotate(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (b, h, seq, d) = x.dims4()?;
        let x1 = x.narrow(D::Minus1, 0, d / 2)?;
        let x2 = x.narrow(D::Minus1, d / 2, d / 2)?;

        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

        let rotated = Tensor::cat(&[
            &((&x1 * &cos)? - (&x2 * &sin)?)?,
            &((&x1 * &sin)? + (&x2 * &cos)?)?,
        ], D::Minus1)?;

        Ok(rotated)
    }
}

/// Multi-head attention with grouped-query attention support
pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    pub fn new(config: &Qwen3TTSConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads();
        let head_dim = config.head_dim();

        let q_proj = linear_no_bias(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        rope: &RotaryEmbedding,
        attention_mask: Option<&Tensor>,
        kv_cache: Option<&mut KVCache>,
        offset: usize,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = hidden_states.dims3()?;

        // Project Q, K, V
        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        // Reshape to [batch, heads, seq, head_dim]
        let q = q.reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply rotary embeddings
        let (q, k) = rope.apply(&q, &k, offset)?;

        // Update KV cache
        let (k, v) = if let Some(cache) = kv_cache {
            let k = cache.update_k(&k)?;
            let v = cache.update_v(&v)?;
            (k, v)
        } else {
            (k, v)
        };

        // Repeat KV heads for GQA
        let k = self.repeat_kv(&k)?;
        let v = self.repeat_kv(&v)?;

        // Scaled dot-product attention
        let attn_weights = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * self.scale)?;

        let attn_weights = if let Some(mask) = attention_mask {
            (attn_weights + mask)?
        } else {
            attn_weights
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((batch, seq_len, self.num_heads * self.head_dim))?;

        Ok(self.o_proj.forward(&attn_output)?)
    }

    fn repeat_kv(&self, x: &Tensor) -> Result<Tensor> {
        let n_rep = self.num_heads / self.num_kv_heads;
        if n_rep == 1 {
            return Ok(x.clone());
        }

        let (batch, num_kv_heads, seq_len, head_dim) = x.dims4()?;
        let x = x.unsqueeze(2)?
            .expand((batch, num_kv_heads, n_rep, seq_len, head_dim))?
            .reshape((batch, num_kv_heads * n_rep, seq_len, head_dim))?;
        Ok(x)
    }
}

/// MLP block with SwiGLU activation
pub struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MLP {
    pub fn new(config: &Qwen3TTSConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;

        Ok(Self {
            gate_proj: linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let gate = candle_nn::ops::silu(&gate)?;
        let up = self.up_proj.forward(x)?;
        Ok(self.down_proj.forward(&(gate * up)?)?)
    }
}

/// Transformer decoder layer
pub struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    pub fn new(config: &Qwen3TTSConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::new(config, vb.pp("self_attn"))?,
            mlp: MLP::new(config, vb.pp("mlp"))?,
            input_layernorm: rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_attention_layernorm: rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        rope: &RotaryEmbedding,
        attention_mask: Option<&Tensor>,
        kv_cache: Option<&mut KVCache>,
        offset: usize,
    ) -> Result<Tensor> {
        // Self-attention with residual
        let residual = hidden_states;
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let hidden_states = self.self_attn.forward(
            &hidden_states,
            rope,
            attention_mask,
            kv_cache,
            offset,
        )?;
        let hidden_states = (residual + hidden_states)?;

        // MLP with residual
        let residual = &hidden_states;
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        let hidden_states = (residual + hidden_states)?;

        Ok(hidden_states)
    }
}

/// KV cache for efficient autoregressive generation
pub struct KVCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
}

impl KVCache {
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    pub fn update_k(&mut self, k: &Tensor) -> Result<Tensor> {
        let k = if let Some(prev_k) = &self.k {
            Tensor::cat(&[prev_k, k], 2)?
        } else {
            k.clone()
        };
        self.k = Some(k.clone());
        Ok(k)
    }

    pub fn update_v(&mut self, v: &Tensor) -> Result<Tensor> {
        let v = if let Some(prev_v) = &self.v {
            Tensor::cat(&[prev_v, v], 2)?
        } else {
            v.clone()
        };
        self.v = Some(v.clone());
        Ok(v)
    }

    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }
}

/// Main Qwen3-TTS model
pub struct Qwen3TTSModel {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    rope: RotaryEmbedding,
    config: Qwen3TTSConfig,
    device: Device,
}

impl Qwen3TTSModel {
    /// Load model from pretrained weights
    pub fn from_pretrained(
        model_id: &str,
        config: &Qwen3TTSConfig,
        device: &Device,
    ) -> Result<Self> {
        // TODO: Implement weight loading from safetensors
        // For now, just return uninitialized error
        anyhow::bail!(
            "Weight loading not yet implemented. Model ID: {}",
            model_id
        )
    }

    /// Create model with given weights
    pub fn new(config: Qwen3TTSConfig, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();

        let embed_tokens = embedding(config.vocab_size, config.hidden_size, vb.pp("model.embed_tokens"))?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(DecoderLayer::new(&config, vb.pp(format!("model.layers.{}", i)))?);
        }

        let norm = rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("model.norm"))?;
        let lm_head = linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?;

        let rope = RotaryEmbedding::new(
            config.head_dim(),
            config.max_position_embeddings,
            config.rope_theta,
            &device,
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope,
            config,
            device,
        })
    }

    /// Forward pass returning logits
    pub fn forward(
        &self,
        input_ids: &Tensor,
        kv_caches: &mut [KVCache],
        offset: usize,
    ) -> Result<Tensor> {
        let attention_mask = self.create_causal_mask(input_ids.dim(1)?, offset)?;

        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        for (i, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(
                &hidden_states,
                &self.rope,
                Some(&attention_mask),
                Some(&mut kv_caches[i]),
                offset,
            )?;
        }

        hidden_states = self.norm.forward(&hidden_states)?;
        let logits = self.lm_head.forward(&hidden_states)?;

        Ok(logits)
    }

    /// Generate codec tokens autoregressively
    pub fn generate(
        &self,
        input_ids: &Tensor,
        speaker_embedding: Option<&Tensor>,
        config: &GenerationConfig,
    ) -> Result<Tensor> {
        let batch_size = input_ids.dim(0)?;
        let mut kv_caches: Vec<KVCache> = (0..self.layers.len())
            .map(|_| KVCache::new())
            .collect();

        // Process input prompt
        let mut logits = self.forward(input_ids, &mut kv_caches, 0)?;
        let mut generated = input_ids.clone();
        let mut offset = input_ids.dim(1)?;

        // Generate tokens
        for _ in 0..config.max_new_tokens {
            // Get last token logits
            let last_logits = logits.i((.., logits.dim(1)? - 1, ..))?;

            // Sample next token
            let next_token = crate::generation::sample(&last_logits, config)?;

            // Check for EOS
            // TODO: Implement proper EOS detection

            // Append to generated sequence
            let next_token = next_token.unsqueeze(1)?;
            generated = Tensor::cat(&[&generated, &next_token], 1)?;

            // Forward with just the new token
            logits = self.forward(&next_token, &mut kv_caches, offset)?;
            offset += 1;
        }

        Ok(generated)
    }

    fn create_causal_mask(&self, seq_len: usize, offset: usize) -> Result<Tensor> {
        let total_len = offset + seq_len;
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..total_len).map(move |j| {
                    if j <= offset + i {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect();

        Ok(Tensor::new(mask.as_slice(), &self.device)?
            .reshape((1, 1, seq_len, total_len))?)
    }
}
