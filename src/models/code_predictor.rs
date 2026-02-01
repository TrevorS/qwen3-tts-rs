//! Code Predictor for Qwen3-TTS
//!
//! The code predictor generates acoustic tokens (groups 2-16) given the
//! semantic token (group 1) and the hidden state from the talker model.
//!
//! Architecture:
//! - 5 transformer layers with same structure as talker
//! - 15 codec embeddings (one per acoustic group)
//! - 15 lm_heads (one per acoustic group)

use anyhow::Result;
use candle_core::{IndexOp, Module, Tensor, D};
use candle_nn::{embedding, linear_no_bias, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};

use super::config::Qwen3TTSConfig;
use super::kv_cache::{AnyKVCache, KVCache, PreAllocKVCache};
use super::transformer::{DecoderLayer, RoPEType, RotaryEmbedding};
use candle_core::DType;

/// Code predictor configuration
#[derive(Debug, Clone)]
pub struct CodePredictorConfig {
    /// Hidden dimension
    pub hidden_size: usize,
    /// Intermediate size for MLP
    pub intermediate_size: usize,
    /// Number of transformer layers
    pub num_hidden_layers: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Number of KV heads (for GQA)
    pub num_key_value_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// RMS norm epsilon
    pub rms_norm_eps: f64,
    /// RoPE theta
    pub rope_theta: f64,
    /// Vocabulary size for codec tokens
    pub vocab_size: usize,
    /// Number of code groups (total, including semantic)
    pub num_code_groups: usize,
    /// Codec embedding dimension (may differ from hidden_size for CustomVoice models)
    /// When different from hidden_size, a small_to_mtp_projection is used
    pub codec_embed_dim: Option<usize>,
}

impl Default for CodePredictorConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            intermediate_size: 3072,
            num_hidden_layers: 5,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 1000000.0,
            vocab_size: 2048,
            num_code_groups: 16,
            codec_embed_dim: None, // When None, uses hidden_size
        }
    }
}

impl CodePredictorConfig {
    /// Create config from parsed HuggingFace config.json.
    ///
    /// When the talker hidden_size differs from the code predictor hidden_size
    /// (e.g. 1.7B models: talker=2048, CP=1024), `codec_embed_dim` is set to
    /// the talker's hidden_size so the `small_to_mtp_projection` layer is created.
    pub fn from_parsed(parsed: &super::config::ParsedModelConfig) -> Self {
        let codec_embed_dim = if parsed.talker_hidden_size != parsed.cp_hidden_size {
            Some(parsed.talker_hidden_size)
        } else {
            None
        };
        Self {
            hidden_size: parsed.cp_hidden_size,
            intermediate_size: parsed.cp_intermediate_size,
            num_hidden_layers: parsed.cp_num_hidden_layers,
            num_attention_heads: parsed.cp_num_attention_heads,
            num_key_value_heads: parsed.cp_num_key_value_heads,
            head_dim: parsed.cp_head_dim,
            rms_norm_eps: parsed.cp_rms_norm_eps,
            rope_theta: parsed.cp_rope_theta,
            vocab_size: parsed.cp_vocab_size,
            num_code_groups: parsed.cp_num_code_groups,
            codec_embed_dim,
        }
    }

    /// Get the codec embedding dimension (defaults to hidden_size)
    pub fn codec_embed_dim(&self) -> usize {
        self.codec_embed_dim.unwrap_or(self.hidden_size)
    }

    /// Create config for CustomVoice model
    pub fn custom_voice() -> Self {
        Self {
            hidden_size: 1024,
            intermediate_size: 3072,
            num_hidden_layers: 5,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 1000000.0,
            vocab_size: 2048,
            num_code_groups: 16,
            codec_embed_dim: Some(2048), // CustomVoice uses 2048-dim codec embeddings
        }
    }

    /// Create a Qwen3TTSConfig for building decoder layers
    fn to_layer_config(&self) -> Qwen3TTSConfig {
        Qwen3TTSConfig {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: Some(self.num_key_value_heads),
            head_dim_override: Some(self.head_dim),
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            vocab_size: self.vocab_size,
            ..Default::default()
        }
    }
}

/// Code predictor model
pub struct CodePredictor {
    /// Codec embeddings for each acoustic group (0-14 for groups 2-16)
    codec_embeddings: Vec<Embedding>,
    /// Projection from codec_embed_dim to hidden_size (for CustomVoice models)
    small_to_mtp_projection: Option<Linear>,
    /// Transformer layers
    layers: Vec<DecoderLayer>,
    /// Final normalization
    norm: RmsNorm,
    /// LM heads for each acoustic group (0-14 for groups 2-16)
    lm_heads: Vec<Linear>,
    /// Rotary embeddings
    rope: RoPEType,
    /// Configuration
    config: CodePredictorConfig,
    /// Cached causal mask for prefill (always 2×2, created once)
    prefill_mask: Tensor,
    /// Device (needed for PreAllocKVCache creation)
    device: candle_core::Device,
    /// Compute dtype (needed for PreAllocKVCache creation)
    dtype: DType,
}

impl CodePredictor {
    /// Create new code predictor
    pub fn new(config: CodePredictorConfig, vb: VarBuilder) -> Result<Self> {
        let layer_config = config.to_layer_config();
        let num_acoustic_groups = config.num_code_groups - 1;
        let codec_embed_dim = config.codec_embed_dim();

        // Create codec embeddings (one per acoustic group)
        // Note: for CustomVoice, codec_embed_dim (2048) differs from hidden_size (1024)
        let mut codec_embeddings = Vec::with_capacity(num_acoustic_groups);
        for i in 0..num_acoustic_groups {
            codec_embeddings.push(embedding(
                config.vocab_size,
                codec_embed_dim,
                vb.pp(format!("model.codec_embedding.{}", i)),
            )?);
        }

        // Projection layer for CustomVoice models (2048 -> 1024)
        let small_to_mtp_projection = if codec_embed_dim != config.hidden_size {
            Some(candle_nn::linear(
                codec_embed_dim,
                config.hidden_size,
                vb.pp("small_to_mtp_projection"),
            )?)
        } else {
            None
        };

        // Create transformer layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(DecoderLayer::new(
                &layer_config,
                vb.pp(format!("model.layers.{}", i)),
            )?);
        }

        // Final norm
        let norm = rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("model.norm"))?;

        // LM heads (one per acoustic group)
        let mut lm_heads = Vec::with_capacity(num_acoustic_groups);
        for i in 0..num_acoustic_groups {
            lm_heads.push(linear_no_bias(
                config.hidden_size,
                config.vocab_size,
                vb.pp(format!("lm_head.{}", i)),
            )?);
        }

        // Rotary embeddings
        let rope = RoPEType::Standard(RotaryEmbedding::new(
            config.head_dim,
            1024, // Max sequence length for code predictor
            config.rope_theta,
            vb.device(),
        )?);

        // Pre-build the 2×2 causal mask for prefill (talker_hidden + semantic_embed).
        // This never changes, so building it once avoids per-frame allocation.
        let prefill_mask = super::transformer::create_causal_mask(2, 0, vb.device())?;

        let device = vb.device().clone();
        let dtype = vb.dtype();

        Ok(Self {
            codec_embeddings,
            small_to_mtp_projection,
            layers,
            norm,
            lm_heads,
            rope,
            config,
            prefill_mask,
            device,
            dtype,
        })
    }

    /// Generate next token logits for a specific group
    ///
    /// # Arguments
    /// * `hidden` - Hidden states from forward pass, shape [batch, seq, hidden]
    /// * `group_idx` - Which acoustic group (0-14 for groups 2-16)
    /// * `position` - Which position to use for prediction
    pub fn get_logits(&self, hidden: &Tensor, group_idx: usize, position: usize) -> Result<Tensor> {
        let pos_hidden = hidden.i((.., position..position + 1, ..))?;
        Ok(self.lm_heads[group_idx].forward(&pos_hidden)?)
    }

    /// Run a prefill pass through the code predictor transformer layers.
    ///
    /// Takes pre-built hidden states (e.g. talker_hidden concatenated with code
    /// embeddings), runs through all layers with KV caches, and returns the
    /// normed hidden states. Use `get_logits` to extract per-group predictions.
    ///
    /// This is a low-level method for reference validation.
    pub fn forward_prefill(
        &self,
        hidden: &Tensor,
        _prev_codes: &[u32],
        kv_caches: &mut [AnyKVCache],
    ) -> Result<Tensor> {
        let device = hidden.device();
        let input = if let Some(proj) = &self.small_to_mtp_projection {
            proj.forward(hidden)?
        } else {
            hidden.clone()
        };

        let seq_len = input.dim(1)?;
        let mask = self.create_causal_mask(seq_len, device)?;

        let mut h = input;
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &self.rope, Some(&mask), Some(&mut kv_caches[i]), 0)?;
        }
        Ok(self.norm.forward(&h)?)
    }

    /// Create a set of KV caches for the code predictor (one per layer).
    ///
    /// Callers should create this once and pass it to [`CodePredictor::generate_acoustic_codes`]
    /// on each frame — the method resets the caches internally, avoiding
    /// per-frame allocation.
    pub fn new_kv_caches(&self) -> Vec<AnyKVCache> {
        // Code predictor: 2 prefill + 15 decode = 17 max tokens
        const CP_MAX_SEQ: usize = 17;

        (0..self.config.num_hidden_layers)
            .map(|_| {
                if self.device.is_cuda() {
                    PreAllocKVCache::new(
                        1, // batch
                        self.config.num_key_value_heads,
                        CP_MAX_SEQ,
                        self.config.head_dim,
                        self.dtype,
                        &self.device,
                    )
                    .map(AnyKVCache::PreAlloc)
                    .unwrap_or_else(|_| AnyKVCache::Concat(KVCache::new()))
                } else {
                    AnyKVCache::Concat(KVCache::new())
                }
            })
            .collect()
    }

    /// Generate all 15 acoustic tokens autoregressively.
    ///
    /// Each acoustic code is predicted conditioned on the talker hidden state,
    /// the semantic token embedding, and all previously generated acoustic codes.
    /// Uses KV caching for sequential generation.
    ///
    /// # Arguments
    /// * `talker_hidden` - Hidden state from talker model, shape `[batch, 1, hidden]`
    /// * `semantic_embed` - Embedding of semantic token, shape `[batch, 1, hidden]`
    /// * `cp_kv_caches` - Reusable KV caches (created via [`CodePredictor::new_kv_caches`]). Reset internally each call.
    ///
    /// # Returns
    /// GPU tensor of shape `[num_acoustic]` containing the 15 acoustic code IDs.
    /// Stays on device to avoid GPU→CPU sync; callers should use tensor ops directly.
    pub fn generate_acoustic_codes(
        &self,
        talker_hidden: &Tensor,
        semantic_embed: &Tensor,
        cp_kv_caches: &mut [AnyKVCache],
    ) -> Result<Tensor> {
        #[cfg(feature = "profiling")]
        let _span = tracing::info_span!("code_predictor_inner").entered();

        // Reset caches from previous frame
        for cache in cp_kv_caches.iter_mut() {
            cache.reset();
        }

        let device = talker_hidden.device();
        let num_acoustic = self.config.num_code_groups - 1; // 15 acoustic codes

        // Step 1: Prefill with [talker_hidden, semantic_embed]
        let input = Tensor::cat(&[talker_hidden, semantic_embed], 1)?;

        // Apply projection if needed (CustomVoice: 2048 -> 1024)
        let input = if let Some(proj) = &self.small_to_mtp_projection {
            proj.forward(&input)?
        } else {
            input
        };

        let seq_len = input.dim(1)?;
        // Use cached mask for the standard 2-token prefill, create on-the-fly otherwise
        let dynamic_mask;
        let mask = if seq_len == 2 {
            &self.prefill_mask
        } else {
            dynamic_mask = self.create_causal_mask(seq_len, device)?;
            &dynamic_mask
        };

        let mut hidden = input;
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(
                &hidden,
                &self.rope,
                Some(mask),
                Some(&mut cp_kv_caches[i]),
                0,
            )?;
        }
        hidden = self.norm.forward(&hidden)?;

        // Step 2: Predict first acoustic code from last position
        // Keep codes as GPU tensors to avoid per-step GPU→CPU syncs.
        // Pre-allocate a single [num_acoustic] tensor and write each code into it
        // to avoid Tensor::cat overhead on many small tensors.
        let last_hidden = hidden.i((.., seq_len - 1..seq_len, ..))?;
        let logits = self.lm_heads[0].forward(&last_hidden)?;
        let first_code = logits.argmax(D::Minus1)?.flatten_all()?; // [1] tensor on GPU

        let mut all_codes = Tensor::zeros(num_acoustic, candle_core::DType::U32, device)?;
        let range = 0..1;
        all_codes = all_codes.slice_assign(&[range], &first_code)?;

        // Also keep a reference to the latest code for embedding lookup
        let mut prev_code = first_code;

        // Step 3: Autoregressively generate remaining 14 codes
        let mut offset = seq_len;
        for group_idx in 1..num_acoustic {
            // Embed previous code using the previous group's embedding (stays on GPU)
            let code_embed = self.codec_embeddings[group_idx - 1].forward(&prev_code)?;
            let code_embed = code_embed.unsqueeze(0)?; // [1, 1, codec_embed_dim]

            // Apply projection if needed
            let code_embed = if let Some(proj) = &self.small_to_mtp_projection {
                proj.forward(&code_embed)?
            } else {
                code_embed
            };

            // Single token attending to all previous positions via KV cache —
            // no masking needed (all-zeros mask is a no-op).
            let mut h = code_embed;
            for (i, layer) in self.layers.iter().enumerate() {
                h = layer.forward(&h, &self.rope, None, Some(&mut cp_kv_caches[i]), offset)?;
            }
            h = self.norm.forward(&h)?;

            // Predict next code (stays on GPU)
            let logits = self.lm_heads[group_idx].forward(&h)?;
            let next_code = logits.argmax(D::Minus1)?.flatten_all()?; // [1] tensor on GPU
            let range = group_idx..group_idx + 1;
            all_codes = all_codes.slice_assign(&[range], &next_code)?;
            prev_code = next_code;
            offset += 1;
        }

        Ok(all_codes)
    }

    fn create_causal_mask(&self, seq_len: usize, device: &candle_core::Device) -> Result<Tensor> {
        super::transformer::create_causal_mask(seq_len, 0, device)
    }

    /// Get acoustic code embedding for a specific group
    ///
    /// group_idx: 0-14 for acoustic groups 2-16
    /// Returns: [1, 1, codec_embed_dim] tensor
    pub fn get_acoustic_embedding(
        &self,
        code: u32,
        group_idx: usize,
        device: &candle_core::Device,
    ) -> Result<Tensor> {
        if group_idx >= self.codec_embeddings.len() {
            anyhow::bail!(
                "Invalid group_idx {} (max {})",
                group_idx,
                self.codec_embeddings.len() - 1
            );
        }
        let code_tensor = Tensor::new(&[code], device)?;
        let embed = self.codec_embeddings[group_idx].forward(&code_tensor)?;
        Ok(embed.unsqueeze(0)?) // [1, 1, codec_embed_dim]
    }

    /// Embed a sequence of codes for a specific acoustic group.
    ///
    /// Used by ICL voice cloning to build reference codec embeddings.
    ///
    /// # Arguments
    /// * `group_idx` — acoustic group (0–14 for codebook groups 2–16)
    /// * `codes` — 1-D i64 tensor of codec token IDs, shape `[T]`
    ///
    /// # Returns
    /// Tensor of shape `[1, T, codec_embed_dim]`
    pub fn embed_codes_for_group(&self, group_idx: usize, codes: &Tensor) -> Result<Tensor> {
        if group_idx >= self.codec_embeddings.len() {
            anyhow::bail!(
                "Invalid group_idx {} (max {})",
                group_idx,
                self.codec_embeddings.len() - 1
            );
        }
        let embed = self.codec_embeddings[group_idx].forward(codes)?; // [T, codec_embed_dim]
        Ok(embed.unsqueeze(0)?) // [1, T, codec_embed_dim]
    }

    /// Get sum of all acoustic code embeddings
    ///
    /// acoustic_codes: 15 acoustic codes for groups 2-16
    /// Returns: [1, 1, codec_embed_dim] tensor with summed embeddings
    pub fn get_acoustic_embeddings_sum(
        &self,
        acoustic_codes: &[u32],
        device: &candle_core::Device,
    ) -> Result<Tensor> {
        if acoustic_codes.len() != self.codec_embeddings.len() {
            anyhow::bail!(
                "Expected {} acoustic codes, got {}",
                self.codec_embeddings.len(),
                acoustic_codes.len()
            );
        }

        let first = self.get_acoustic_embedding(acoustic_codes[0], 0, device)?;
        acoustic_codes[1..]
            .iter()
            .enumerate()
            .try_fold(first, |acc, (i, &code)| {
                let embed = self.get_acoustic_embedding(code, i + 1, device)?;
                acc.add(&embed).map_err(Into::into)
            })
    }

    /// Get sum of all acoustic code embeddings from a GPU tensor.
    ///
    /// Like `get_acoustic_embeddings_sum` but takes codes as a \[num_acoustic\] tensor
    /// already on device, avoiding 15 small CPU→GPU transfers.
    pub fn get_acoustic_embeddings_sum_from_tensor(
        &self,
        acoustic_codes: &Tensor,
    ) -> Result<Tensor> {
        let n = acoustic_codes.dim(0)?;
        if n != self.codec_embeddings.len() {
            anyhow::bail!(
                "Expected {} acoustic codes, got {}",
                self.codec_embeddings.len(),
                n
            );
        }

        let first_code = acoustic_codes.narrow(0, 0, 1)?;
        let first = self.codec_embeddings[0]
            .forward(&first_code)?
            .unsqueeze(0)?;
        (1..n).try_fold(first, |acc, i| {
            let code = acoustic_codes.narrow(0, i, 1)?;
            let embed = self.codec_embeddings[i].forward(&code)?.unsqueeze(0)?;
            acc.add(&embed).map_err(Into::into)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    fn create_mock_vb(device: &Device) -> VarBuilder<'static> {
        let varmap = VarMap::new();
        VarBuilder::from_varmap(&varmap, DType::F32, device)
    }

    #[test]
    fn test_config_default() {
        let config = CodePredictorConfig::default();
        assert_eq!(config.num_hidden_layers, 5);
        assert_eq!(config.num_code_groups, 16);
        assert_eq!(config.hidden_size, 1024);
    }

    #[test]
    fn test_code_predictor_construction() {
        let device = Device::Cpu;
        let vb = create_mock_vb(&device);

        let config = CodePredictorConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            vocab_size: 64,
            num_code_groups: 4,
            ..Default::default()
        };

        let predictor = CodePredictor::new(config, vb);
        assert!(predictor.is_ok());

        let predictor = predictor.unwrap();
        assert_eq!(predictor.codec_embeddings.len(), 3); // 4-1 acoustic groups
        assert_eq!(predictor.layers.len(), 2);
        assert_eq!(predictor.lm_heads.len(), 3);
    }
}
