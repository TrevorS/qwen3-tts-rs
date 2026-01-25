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
use super::qwen3_tts::{DecoderLayer, KVCache, RotaryEmbedding};

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
    /// Create config from Qwen3TTS config
    pub fn from_qwen3_tts(config: &Qwen3TTSConfig) -> Self {
        Self {
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_hidden_layers: 5, // Fixed for code predictor
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_kv_heads(),
            head_dim: config.head_dim(),
            rms_norm_eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            vocab_size: config.codebook_size,
            num_code_groups: config.num_codebook_groups,
            codec_embed_dim: None,
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
    rope: RotaryEmbedding,
    /// Configuration
    config: CodePredictorConfig,
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
        let rope = RotaryEmbedding::new(
            config.head_dim,
            1024, // Max sequence length for code predictor
            config.rope_theta,
            vb.device(),
        )?;

        Ok(Self {
            codec_embeddings,
            small_to_mtp_projection,
            layers,
            norm,
            lm_heads,
            rope,
            config,
        })
    }

    /// Forward pass for prefill
    ///
    /// Takes the talker hidden state and previously generated codes,
    /// returns hidden states for all positions.
    pub fn forward_prefill(
        &self,
        talker_hidden: &Tensor,
        codes: &[u32],
        kv_caches: &mut [KVCache],
    ) -> Result<Tensor> {
        // Build input: [talker_hidden, embed(code_0), embed(code_1), ...]
        let mut inputs = vec![talker_hidden.clone()];

        for (i, &code) in codes.iter().enumerate() {
            let code_tensor = Tensor::new(&[code], talker_hidden.device())?;
            let embed = self.codec_embeddings[i].forward(&code_tensor)?;
            inputs.push(embed.unsqueeze(0)?);
        }

        let hidden = Tensor::cat(&inputs, 1)?;

        // Apply projection if needed (CustomVoice: 2048 -> 1024)
        let hidden = if let Some(proj) = &self.small_to_mtp_projection {
            proj.forward(&hidden)?
        } else {
            hidden
        };

        let seq_len = hidden.dim(1)?;

        // Create causal mask
        let mask = self.create_causal_mask(seq_len, talker_hidden.device())?;

        // Run through layers
        let mut hidden = hidden;
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &self.rope, Some(&mask), Some(&mut kv_caches[i]), 0)?;
        }

        // Final norm
        Ok(self.norm.forward(&hidden)?)
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

    /// Generate all 15 acoustic tokens given talker hidden state and semantic token
    ///
    /// The code predictor uses autoregressive decoding: each acoustic token
    /// depends on the previous one.
    ///
    /// Flow:
    /// 1. Prefill: `[talker_hidden, semantic_embed]` → `lm_head[0]` → acoustic_0
    /// 2. Step 1: embed(acoustic_0) via `codec_embedding[0]` → `lm_head[1]` → acoustic_1
    /// 3. Step 2: embed(acoustic_1) via `codec_embedding[1]` → `lm_head[2]` → acoustic_2
    ///
    /// ...and so on for all 15 acoustic tokens.
    ///
    /// # Arguments
    /// * `talker_hidden` - Hidden state from talker model, shape [batch, 1, hidden]
    /// * `semantic_embed` - Embedding of semantic token, shape [batch, 1, hidden]
    ///
    /// # Returns
    /// Vector of 15 acoustic token IDs
    pub fn generate_acoustic_codes(
        &self,
        talker_hidden: &Tensor,
        semantic_embed: &Tensor,
    ) -> Result<Vec<u32>> {
        let device = talker_hidden.device();
        let num_acoustic = self.config.num_code_groups - 1; // 15 acoustic codes

        // Create KV caches for autoregressive generation
        let mut kv_caches: Vec<KVCache> = (0..self.config.num_hidden_layers)
            .map(|_| KVCache::new())
            .collect();

        // === Prefill ===
        // Input: [talker_hidden, semantic_embed] → shape [1, 2, embed_dim]
        let input = Tensor::cat(&[talker_hidden, semantic_embed], 1)?;

        // Apply projection if needed (CustomVoice: 2048 -> 1024)
        let input = if let Some(proj) = &self.small_to_mtp_projection {
            proj.forward(&input)?
        } else {
            input
        };

        let seq_len = input.dim(1)?;
        let mask = self.create_causal_mask(seq_len, device)?;

        let mut hidden = input;
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &self.rope, Some(&mask), Some(&mut kv_caches[i]), 0)?;
        }
        hidden = self.norm.forward(&hidden)?;

        // First acoustic code from lm_head[0] at last position
        let hidden_last = hidden.i((.., seq_len - 1..seq_len, ..))?;
        let logits = self.lm_heads[0].forward(&hidden_last)?;
        let first_code: u32 = logits.argmax(D::Minus1)?.flatten_all()?.to_vec1::<u32>()?[0];

        let mut codes = Vec::with_capacity(num_acoustic);
        codes.push(first_code);

        // === Autoregressive generation for remaining 14 acoustic codes ===
        let mut prev_code = first_code;
        let mut offset = seq_len; // Current position for KV cache

        for group_idx in 1..num_acoustic {
            // Embed previous code with codec_embedding[group_idx - 1]
            let code_tensor = Tensor::new(&[prev_code], device)?;
            let code_embed = self.codec_embeddings[group_idx - 1].forward(&code_tensor)?;
            let code_embed = code_embed.unsqueeze(0)?; // [1, 1, embed_dim]

            // Apply projection if needed
            let code_embed = if let Some(proj) = &self.small_to_mtp_projection {
                proj.forward(&code_embed)?
            } else {
                code_embed
            };

            // Forward through layers (single token, using KV cache)
            let mut hidden = code_embed;
            for (i, layer) in self.layers.iter().enumerate() {
                hidden =
                    layer.forward(&hidden, &self.rope, None, Some(&mut kv_caches[i]), offset)?;
            }
            hidden = self.norm.forward(&hidden)?;

            // Get next acoustic code from lm_head[group_idx]
            let logits = self.lm_heads[group_idx].forward(&hidden)?;
            let next_code: u32 = logits.argmax(D::Minus1)?.flatten_all()?.to_vec1::<u32>()?[0];

            codes.push(next_code);
            prev_code = next_code;
            offset += 1;
        }

        Ok(codes)
    }

    fn create_causal_mask(&self, seq_len: usize, device: &candle_core::Device) -> Result<Tensor> {
        let mut mask_data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask_data[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
        Ok(Tensor::from_vec(
            mask_data,
            (1, 1, seq_len, seq_len),
            device,
        )?)
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

        let mut sum: Option<Tensor> = None;
        for (i, &code) in acoustic_codes.iter().enumerate() {
            let embed = self.get_acoustic_embedding(code, i, device)?;
            sum = Some(match sum {
                Some(s) => s.add(&embed)?,
                None => embed,
            });
        }

        Ok(sum.unwrap())
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
    fn test_config_from_qwen3_tts() {
        let qwen_config = Qwen3TTSConfig {
            hidden_size: 1024,
            num_attention_heads: 16,
            num_key_value_heads: Some(8),
            head_dim_override: Some(128),
            num_codebook_groups: 16,
            codebook_size: 2048,
            ..Default::default()
        };

        let config = CodePredictorConfig::from_qwen3_tts(&qwen_config);
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.num_code_groups, 16);
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
