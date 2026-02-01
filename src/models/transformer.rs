//! Shared transformer building blocks for Qwen3-TTS
//!
//! Contains `KVCache`, `RotaryEmbedding`, `MRoPE`, `Attention`, `MLP`,
//! and `DecoderLayer` — used by both `TalkerModel` and `CodePredictor`.

use anyhow::Result;
use candle_core::{Device, IndexOp, Module, Tensor, D};
use candle_nn::{linear_no_bias, rms_norm, Linear, RmsNorm, VarBuilder};

use super::fused_ops::FusedRmsNorm;

#[cfg(feature = "flash-attn")]
use candle_flash_attn::flash_attn;

use super::config::Qwen3TTSConfig;

/// Create a causal attention mask.
///
/// Returns a `[1, 1, seq_len, offset + seq_len]` tensor where position `(i, j)`
/// is `0.0` if `j <= offset + i` (allowed) and `NEG_INFINITY` (masked).
pub fn create_causal_mask(seq_len: usize, offset: usize, device: &Device) -> Result<Tensor> {
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

    Ok(Tensor::new(mask.as_slice(), device)?.reshape((1, 1, seq_len, total_len))?)
}

/// Apply RoPE rotation to a tensor.
///
/// `x` has shape `[batch, heads, seq_len, head_dim]`.
/// `cos` and `sin` have shape `[seq_len, head_dim/2]`.
fn apply_rope_rotation(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_b, _h, _seq, d) = x.dims4()?;
    let x1 = x.narrow(D::Minus1, 0, d / 2)?;
    let x2 = x.narrow(D::Minus1, d / 2, d / 2)?;

    // Broadcast cos/sin from [seq_len, half_dim] to [1, 1, seq_len, half_dim]
    let cos = cos
        .unsqueeze(0)?
        .unsqueeze(0)?
        .to_dtype(x.dtype())?
        .broadcast_as(x1.shape())?;
    let sin = sin
        .unsqueeze(0)?
        .unsqueeze(0)?
        .to_dtype(x.dtype())?
        .broadcast_as(x1.shape())?;

    // Standard RoPE: [x1*cos - x2*sin, x2*cos + x1*sin]
    let rotated = Tensor::cat(
        &[
            &(x1.mul(&cos)? - x2.mul(&sin)?)?,
            &(x2.mul(&cos)? + x1.mul(&sin)?)?,
        ],
        D::Minus1,
    )?;

    Ok(rotated)
}

/// Rotary position embedding (standard RoPE)
pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
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

        Ok(Self { cos, sin })
    }

    pub fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;
        let cos = self.cos.i(offset..offset + seq_len)?;
        let sin = self.sin.i(offset..offset + seq_len)?;

        let q_rot = apply_rope_rotation(q, &cos, &sin)?;
        let k_rot = apply_rope_rotation(k, &cos, &sin)?;

        Ok((q_rot, k_rot))
    }
}

/// Multimodal Rotary Embedding (MRoPE) for 3D positions (temporal, height, width)
///
/// Uses interleaved layout matching Qwen3-TTS's rope_scaling configuration.
/// For TTS, all 3 position dimensions use the same value, but the interleaving
/// still affects how frequencies are distributed across the head dimension.
pub struct MRoPE {
    inv_freq: Tensor,
    device: Device,
}

impl MRoPE {
    /// Create MRoPE with specified mrope_section
    ///
    /// mrope_section = [24, 20, 20] means:
    /// - 24 frequency pairs for temporal (T)
    /// - 20 frequency pairs for height (H)
    /// - 20 frequency pairs for width (W)
    ///
    /// Total = 64 = head_dim / 2
    pub fn new(
        dim: usize,
        theta: f64,
        _mrope_section: [usize; 3],
        device: &Device,
    ) -> Result<Self> {
        // Compute inverse frequencies
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / (theta as f32).powf(i as f32 / dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;

        Ok(Self {
            inv_freq,
            device: device.clone(),
        })
    }

    /// Apply MRoPE to query and key tensors
    ///
    /// For TTS, all 3 position dimensions (T, H, W) use the same position value,
    /// but the interleaving still changes how frequencies are distributed.
    ///
    /// Arguments:
    /// - q, k: [batch, heads, seq_len, head_dim]
    /// - offset: position offset for KV cache
    /// - seq_len: sequence length
    pub fn apply(
        &self,
        q: &Tensor,
        k: &Tensor,
        offset: usize,
        seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        // Generate position IDs for all 3 dimensions (same values for TTS)
        let positions: Vec<f32> = (offset..offset + seq_len).map(|i| i as f32).collect();
        let pos = Tensor::new(positions.as_slice(), &self.device)?; // [seq_len]

        // Compute frequencies for each dimension
        // freqs = pos[:, None] * inv_freq[None, :] -> [seq_len, head_dim/2]
        let pos_col = pos.unsqueeze(1)?; // [seq_len, 1]
        let inv_freq_row = self.inv_freq.unsqueeze(0)?; // [1, head_dim/2]
        let freqs = pos_col.matmul(&inv_freq_row)?; // [seq_len, head_dim/2]

        // For TTS, all 3 dimensions have the same position, so freqs_t = freqs_h = freqs_w.
        // The interleaving is a no-op when all positions are identical.
        // freqs is already [seq_len, head_dim/2] — compute cos/sin directly.
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        let q_rot = apply_rope_rotation(q, &cos, &sin)?;
        let k_rot = apply_rope_rotation(k, &cos, &sin)?;

        Ok((q_rot, k_rot))
    }
}

/// Either standard RoPE or MRoPE (multimodal)
pub enum RoPEType {
    Standard(RotaryEmbedding),
    Multimodal(MRoPE),
}

impl RoPEType {
    /// Apply rotary embedding to Q and K tensors
    pub fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        match self {
            RoPEType::Standard(rope) => rope.apply(q, k, offset),
            RoPEType::Multimodal(mrope) => {
                let seq_len = q.dim(2)?;
                mrope.apply(q, k, offset, seq_len)
            }
        }
    }
}

/// Multi-head attention with grouped-query attention and QK normalization
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

        // QK normalization: RMSNorm applied per-head after projection
        let q_norm = rms_norm(head_dim, config.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(head_dim, config.rms_norm_eps, vb.pp("k_norm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        rope: &RoPEType,
        attention_mask: Option<&Tensor>,
        kv_cache: Option<&mut AnyKVCache>,
        offset: usize,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = hidden_states.dims3()?;

        // Project Q, K, V
        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        // Reshape to [batch, seq, heads, head_dim] for QK norm
        let q = q.reshape((batch, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?;

        // Apply QK normalization (per-head RMSNorm)
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // Transpose to [batch, heads, seq, head_dim]
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // Apply rotary embeddings
        let (q, k) = rope.apply(&q, &k, offset)?;

        // Update KV cache
        let (k, v) = if let Some(cache) = kv_cache {
            cache.update(&k, &v)?
        } else {
            (k, v)
        };

        // ---- Attention computation ----
        // When compiled with flash-attn, use it on CUDA and fall back to
        // standard attention on CPU. Without the feature, always use standard.

        #[cfg(feature = "flash-attn")]
        let use_flash = q.device().is_cuda();
        #[cfg(not(feature = "flash-attn"))]
        let use_flash = false;

        let attn_output = if use_flash {
            #[cfg(feature = "flash-attn")]
            {
                // Flash Attention 2: handles GQA natively (no repeat_kv needed),
                // uses causal=true instead of an explicit attention mask.
                // Requires half-precision — cast f32→bf16 for the kernel, cast back after.
                let _ = attention_mask;
                let input_dtype = q.dtype();
                // flash_attn expects [B, S, H, D] — transpose back from [B, H, S, D]
                let q = q
                    .transpose(1, 2)?
                    .to_dtype(candle_core::DType::BF16)?
                    .contiguous()?;
                let k = k
                    .transpose(1, 2)?
                    .to_dtype(candle_core::DType::BF16)?
                    .contiguous()?;
                let v = v
                    .transpose(1, 2)?
                    .to_dtype(candle_core::DType::BF16)?
                    .contiguous()?;
                let softmax_scale = self.scale as f32;
                let attn_output = flash_attn(&q, &k, &v, softmax_scale, /* causal */ true)?;
                // [B, S_q, H_q, D] → cast back → [B, S_q, hidden]
                attn_output.to_dtype(input_dtype)?.reshape((
                    batch,
                    seq_len,
                    self.num_heads * self.head_dim,
                ))?
            }
            #[cfg(not(feature = "flash-attn"))]
            unreachable!()
        } else {
            // Manual scaled dot-product attention with GQA repeat
            let k = self.repeat_kv(&k)?;
            let v = self.repeat_kv(&v)?;
            // Ensure contiguous layout for CUDA matmul (transpose creates non-contiguous views)
            let q = q.contiguous()?;
            let k = k.contiguous()?;
            let v = v.contiguous()?;
            let attn_weights =
                (q.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)? * self.scale)?;
            let attn_weights = if let Some(mask) = attention_mask {
                // Cast mask to match attn_weights dtype (e.g. f32 mask with bf16 weights)
                let mask = mask.to_dtype(attn_weights.dtype())?;
                attn_weights.broadcast_add(&mask)?
            } else {
                attn_weights
            };
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            let attn_output = attn_weights.matmul(&v)?;
            attn_output.transpose(1, 2)?.reshape((
                batch,
                seq_len,
                self.num_heads * self.head_dim,
            ))?
        };

        Ok(self.o_proj.forward(&attn_output)?)
    }

    fn repeat_kv(&self, x: &Tensor) -> Result<Tensor> {
        let n_rep = self.num_heads / self.num_kv_heads;
        if n_rep == 1 {
            return Ok(x.clone());
        }

        let (batch, num_kv_heads, seq_len, head_dim) = x.dims4()?;
        let x = x
            .unsqueeze(2)?
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
    post_attention_layernorm: FusedRmsNorm,
}

impl DecoderLayer {
    pub fn new(config: &Qwen3TTSConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::new(config, vb.pp("self_attn"))?,
            mlp: MLP::new(config, vb.pp("mlp"))?,
            input_layernorm: rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: FusedRmsNorm::load(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        rope: &RoPEType,
        attention_mask: Option<&Tensor>,
        kv_cache: Option<&mut AnyKVCache>,
        offset: usize,
    ) -> Result<Tensor> {
        // Self-attention with residual
        let residual = hidden_states;
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let hidden_states =
            self.self_attn
                .forward(&hidden_states, rope, attention_mask, kv_cache, offset)?;

        // Fused: residual add + post_attention_layernorm in one kernel (CUDA)
        let (normed, hidden_states) = self
            .post_attention_layernorm
            .forward_residual(&hidden_states, residual)?;

        // MLP with residual
        let mlp_out = self.mlp.forward(&normed)?;
        let hidden_states = (hidden_states + mlp_out)?;

        Ok(hidden_states)
    }
}

// KVCache, PreAllocKVCache, and AnyKVCache are defined in kv_cache.rs
pub use super::kv_cache::{AnyKVCache, KVCache, PreAllocKVCache};

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use candle_nn::VarMap;

    fn create_mock_vb(device: &Device) -> VarBuilder<'static> {
        let varmap = VarMap::new();
        VarBuilder::from_varmap(&varmap, DType::F32, device)
    }

    fn small_config() -> Qwen3TTSConfig {
        Qwen3TTSConfig {
            vocab_size: 1000,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            max_position_embeddings: 512,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            ..Default::default()
        }
    }

    #[test]
    fn test_rotary_embedding_creation() {
        let device = Device::Cpu;
        let rope = RotaryEmbedding::new(64, 512, 10000.0, &device).unwrap();
        // Verify cos/sin shapes match expected dim
        assert_eq!(rope.cos.dims()[1], 32); // dim / 2
    }

    #[test]
    fn test_rotary_embedding_shape() {
        let device = Device::Cpu;
        let rope = RotaryEmbedding::new(64, 512, 10000.0, &device).unwrap();

        // cos and sin should be [max_seq_len, dim/2]
        assert_eq!(rope.cos.dims()[0], 512);
        assert_eq!(rope.cos.dims()[1], 32); // dim / 2
        assert_eq!(rope.sin.dims()[0], 512);
        assert_eq!(rope.sin.dims()[1], 32);
    }

    #[test]
    fn test_rotary_embedding_apply() {
        let device = Device::Cpu;
        let rope = RotaryEmbedding::new(16, 512, 10000.0, &device).unwrap();

        // q, k: [batch, heads, seq, head_dim]
        let q = Tensor::randn(0.0f32, 1.0, (2, 4, 10, 16), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (2, 4, 10, 16), &device).unwrap();

        let (q_rot, k_rot) = rope.apply(&q, &k, 0).unwrap();

        assert_eq!(q_rot.dims(), q.dims());
        assert_eq!(k_rot.dims(), k.dims());
    }

    #[test]
    fn test_rotary_embedding_with_offset() {
        let device = Device::Cpu;
        let rope = RotaryEmbedding::new(16, 512, 10000.0, &device).unwrap();

        let q = Tensor::randn(0.0f32, 1.0, (1, 2, 5, 16), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (1, 2, 5, 16), &device).unwrap();

        let (q_rot, k_rot) = rope.apply(&q, &k, 100).unwrap();

        assert_eq!(q_rot.dims(), &[1, 2, 5, 16]);
        assert_eq!(k_rot.dims(), &[1, 2, 5, 16]);
    }

    #[test]
    fn test_kv_cache_new() {
        let cache = KVCache::new();
        assert!(cache.k.is_none());
        assert!(cache.v.is_none());
    }

    #[test]
    fn test_kv_cache_update() {
        let device = Device::Cpu;
        let mut cache = KVCache::new();

        let k1 = Tensor::randn(0.0f32, 1.0, (1, 2, 4, 16), &device).unwrap();
        let k_out = cache.update_k(&k1).unwrap();
        assert_eq!(k_out.dims(), &[1, 2, 4, 16]);

        let k2 = Tensor::randn(0.0f32, 1.0, (1, 2, 3, 16), &device).unwrap();
        let k_out = cache.update_k(&k2).unwrap();
        assert_eq!(k_out.dims(), &[1, 2, 7, 16]); // 4 + 3 = 7
    }

    #[test]
    fn test_kv_cache_reset() {
        let device = Device::Cpu;
        let mut cache = KVCache::new();

        let k = Tensor::randn(0.0f32, 1.0, (1, 2, 4, 16), &device).unwrap();
        cache.update_k(&k).unwrap();
        assert!(cache.k.is_some());

        cache.reset();
        assert!(cache.k.is_none());
        assert!(cache.v.is_none());
    }

    #[test]
    fn test_mlp() {
        let device = Device::Cpu;
        let config = small_config();
        let vb = create_mock_vb(&device);

        let mlp = MLP::new(&config, vb).unwrap();

        // Input: [batch=2, seq=10, hidden=64]
        let input = Tensor::randn(0.0f32, 1.0, (2, 10, 64), &device).unwrap();
        let output = mlp.forward(&input).unwrap();

        assert_eq!(output.dims(), &[2, 10, 64]);
    }

    #[test]
    fn test_attention() {
        let device = Device::Cpu;
        let config = small_config();
        let vb = create_mock_vb(&device);

        let attn = Attention::new(&config, vb).unwrap();

        assert_eq!(attn.num_heads, 4);
        assert_eq!(attn.num_kv_heads, 2);
        assert_eq!(attn.head_dim, 16); // 64 / 4 = 16
    }

    #[test]
    fn test_attention_forward() {
        let device = Device::Cpu;
        let config = small_config();
        let vb = create_mock_vb(&device);

        let attn = Attention::new(&config, vb).unwrap();
        let rope = RoPEType::Standard(RotaryEmbedding::new(16, 512, 10000.0, &device).unwrap());

        // Input: [batch=1, seq=10, hidden=64]
        let input = Tensor::randn(0.0f32, 1.0, (1, 10, 64), &device).unwrap();
        let output = attn.forward(&input, &rope, None, None, 0).unwrap();

        assert_eq!(output.dims(), &[1, 10, 64]);
    }

    #[test]
    fn test_attention_with_cache() {
        let device = Device::Cpu;
        let config = small_config();
        let vb = create_mock_vb(&device);

        let attn = Attention::new(&config, vb).unwrap();
        let rope = RoPEType::Standard(RotaryEmbedding::new(16, 512, 10000.0, &device).unwrap());
        let mut cache = AnyKVCache::Concat(KVCache::new());

        // First forward
        let input1 = Tensor::randn(0.0f32, 1.0, (1, 5, 64), &device).unwrap();
        let _out1 = attn
            .forward(&input1, &rope, None, Some(&mut cache), 0)
            .unwrap();

        // Second forward with cache
        let input2 = Tensor::randn(0.0f32, 1.0, (1, 3, 64), &device).unwrap();
        let out2 = attn
            .forward(&input2, &rope, None, Some(&mut cache), 5)
            .unwrap();

        assert_eq!(out2.dims(), &[1, 3, 64]);
    }

    #[test]
    fn test_decoder_layer() {
        let device = Device::Cpu;
        let config = small_config();
        let vb = create_mock_vb(&device);

        let layer = DecoderLayer::new(&config, vb).unwrap();
        let rope = RoPEType::Standard(RotaryEmbedding::new(16, 512, 10000.0, &device).unwrap());
        let mut cache = AnyKVCache::Concat(KVCache::new());

        let input = Tensor::randn(0.0f32, 1.0, (1, 8, 64), &device).unwrap();
        let output = layer
            .forward(&input, &rope, None, Some(&mut cache), 0)
            .unwrap();

        assert_eq!(output.dims(), &[1, 8, 64]);
    }

    #[test]
    fn test_qwen3_tts_kv_caches_creation() {
        // Test that KV caches can be created for a model
        let kv_caches: Vec<AnyKVCache> =
            (0..2).map(|_| AnyKVCache::Concat(KVCache::new())).collect();
        // Just verify KV caches can be created
        assert_eq!(kv_caches.len(), 2);
    }

    #[test]
    fn test_repeat_kv_no_repeat() {
        let device = Device::Cpu;
        let config = Qwen3TTSConfig {
            num_attention_heads: 4,
            num_key_value_heads: Some(4), // Same as num_heads
            hidden_size: 64,
            ..small_config()
        };
        let vb = create_mock_vb(&device);

        let attn = Attention::new(&config, vb).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (1, 4, 10, 16), &device).unwrap();
        let repeated = attn.repeat_kv(&x).unwrap();

        // Should be unchanged when n_rep = 1
        assert_eq!(repeated.dims(), x.dims());
    }

    #[test]
    fn test_repeat_kv_with_repeat() {
        let device = Device::Cpu;
        let config = Qwen3TTSConfig {
            num_attention_heads: 8,
            num_key_value_heads: Some(2), // 8/2 = 4x repeat
            hidden_size: 128,
            ..small_config()
        };
        let vb = create_mock_vb(&device);

        let attn = Attention::new(&config, vb).unwrap();

        // [batch=1, kv_heads=2, seq=10, head_dim=16]
        let x = Tensor::randn(0.0f32, 1.0, (1, 2, 10, 16), &device).unwrap();
        let repeated = attn.repeat_kv(&x).unwrap();

        // Should expand to [1, 8, 10, 16]
        assert_eq!(repeated.dims(), &[1, 8, 10, 16]);
    }
}
