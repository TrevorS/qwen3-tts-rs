//! TTS-specific generation logic
//!
//! This module handles the proper prefill construction and generation loop
//! for Qwen3-TTS, including:
//! - Dual-stream embedding fusion (text + codec)
//! - Special token handling (think, language, speaker)
//! - Token suppression during sampling
//! - Residual VQ pattern (summing all 16 embeddings)

use anyhow::Result;
use candle_core::{Device, Tensor};
use std::collections::HashMap;

/// TTS-specific token IDs
#[derive(Debug, Clone)]
pub struct TtsTokenIds {
    // Text special tokens (from Qwen tokenizer)
    pub im_start: u32,      // 151644
    pub im_end: u32,        // 151645
    pub assistant: u32,     // 77091
    pub newline: u32,       // 198 (\n)
    pub tts_bos: u32,       // 151672
    pub tts_eos: u32,       // 151673
    pub tts_pad: u32,       // 151671

    // Codec special tokens
    pub codec_pad: u32,     // 2148
    pub codec_bos: u32,     // 2149
    pub codec_eos: u32,     // 2150
    pub codec_think: u32,   // 2154
    pub codec_nothink: u32, // 2155
    pub codec_think_bos: u32, // 2156
    pub codec_think_eos: u32, // 2157

    // Language IDs (in codec embedding space)
    pub lang_english: u32,  // 2050
    pub lang_chinese: u32,  // 2055

    // Preset speakers (CustomVoice model)
    pub spk_vivian: u32,    // 3065
    pub spk_serena: u32,    // 3066
    pub spk_ryan: u32,      // 3061
    pub spk_aiden: u32,     // 2861
    pub spk_uncle_fu: u32,  // 3010
    pub spk_dylan: u32,     // 2878
    pub spk_eric: u32,      // 2875
    pub spk_ono_anna: u32,  // 2873
    pub spk_sohee: u32,     // 2864
}

impl Default for TtsTokenIds {
    fn default() -> Self {
        Self {
            // Text tokens
            im_start: 151644,
            im_end: 151645,
            assistant: 77091,
            newline: 198,
            tts_bos: 151672,
            tts_eos: 151673,
            tts_pad: 151671,
            // Codec tokens
            codec_pad: 2148,
            codec_bos: 2149,
            codec_eos: 2150,
            codec_think: 2154,
            codec_nothink: 2155,
            codec_think_bos: 2156,
            codec_think_eos: 2157,
            // Languages
            lang_english: 2050,
            lang_chinese: 2055,
            // Preset speakers
            spk_vivian: 3065,
            spk_serena: 3066,
            spk_ryan: 3061,
            spk_aiden: 2861,
            spk_uncle_fu: 3010,
            spk_dylan: 2878,
            spk_eric: 2875,
            spk_ono_anna: 2873,
            spk_sohee: 2864,
        }
    }
}

/// TTS Generation configuration
#[derive(Debug, Clone)]
pub struct TtsGenerationConfig {
    /// Maximum frames to generate
    pub max_frames: usize,
    /// Sampling temperature
    pub temperature: f64,
    /// Top-k sampling
    pub top_k: usize,
    /// Language ID to use
    pub language_id: u32,
    /// Speaker ID (None for base model auto)
    pub speaker_id: Option<u32>,
    /// Whether to use thinking mode
    pub use_think: bool,
    /// Token IDs
    pub token_ids: TtsTokenIds,
}

impl Default for TtsGenerationConfig {
    fn default() -> Self {
        let token_ids = TtsTokenIds::default();
        Self {
            max_frames: 100,
            temperature: 0.7,
            top_k: 50,
            language_id: token_ids.lang_english,
            speaker_id: None,
            use_think: true,
            token_ids,
        }
    }
}

/// Build the prefill embeddings for TTS generation
///
/// Structure:
/// 1. Role prefix: [im_start, assistant, im_end] -> text_projection
/// 2. Control sequence: text_pad/bos + codec_control (added together)
/// 3. First text token + codec_bos
///
/// Returns (prefill_embeddings, trailing_text_embeddings)
pub fn build_prefill_embeddings(
    text_token_ids: &[u32],
    text_embedding: &Tensor,
    codec_embedding: &Tensor,
    text_projection_weights: &HashMap<String, Tensor>,
    config: &TtsGenerationConfig,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let token_ids = &config.token_ids;

    // Helper to project text embeddings through text_projection MLP
    let project_text = |ids: &[u32]| -> Result<Tensor> {
        let ids_tensor = Tensor::new(ids, device)?;
        let embeds = text_embedding.index_select(&ids_tensor, 0)?;
        let embeds = embeds.unsqueeze(0)?; // [1, seq, 2048]
        text_projection_forward(&embeds, text_projection_weights)
    };

    // Helper to get codec embeddings
    let get_codec = |ids: &[u32]| -> Result<Tensor> {
        let ids_tensor = Tensor::new(ids, device)?;
        let embeds = codec_embedding.index_select(&ids_tensor, 0)?;
        Ok(embeds.unsqueeze(0)?) // [1, seq, hidden]
    };

    // Step 1: Role prefix [im_start, assistant, newline]
    // This matches Python: input_id[:, :3] from "<|im_start|>assistant\n..."
    let role_prefix_ids = [token_ids.im_start, token_ids.assistant, token_ids.newline];
    let role_prefix_embed = project_text(&role_prefix_ids)?;
    // [1, 3, hidden]

    // Step 2: Build codec control sequence
    let codec_control_ids = if config.use_think {
        vec![
            token_ids.codec_think,
            token_ids.codec_think_bos,
            config.language_id,
            token_ids.codec_think_eos,
            token_ids.codec_pad,
            token_ids.codec_bos,
        ]
    } else {
        vec![
            token_ids.codec_nothink,
            token_ids.codec_think_bos,
            token_ids.codec_think_eos,
            token_ids.codec_pad,
            token_ids.codec_bos,
        ]
    };

    // Add speaker ID if provided
    let codec_control_ids = if let Some(spk_id) = config.speaker_id {
        let mut ids = codec_control_ids[..codec_control_ids.len() - 2].to_vec();
        ids.push(spk_id);
        ids.push(token_ids.codec_pad);
        ids.push(token_ids.codec_bos);
        ids
    } else {
        codec_control_ids
    };

    let codec_control_embed = get_codec(&codec_control_ids)?;
    // [1, control_len, hidden]

    let control_len = codec_control_ids.len();

    // Step 3: Build text control (pad tokens + tts_bos)
    // We need (control_len - 2) pad tokens, then tts_bos
    let mut text_control_ids = vec![token_ids.tts_pad; control_len - 2];
    text_control_ids.push(token_ids.tts_bos);
    let text_control_embed = project_text(&text_control_ids)?;
    // [1, control_len - 1, hidden]

    // Step 4: Fuse text_control + codec_control[:-1] by ADDITION
    let codec_control_prefix = codec_control_embed.narrow(1, 0, control_len - 1)?;
    let fused_control = text_control_embed.add(&codec_control_prefix)?;
    // [1, control_len - 1, hidden]

    // Step 5: First text token + codec_bos
    let first_text_embed = if !text_token_ids.is_empty() {
        project_text(&[text_token_ids[0]])?
    } else {
        project_text(&[token_ids.tts_pad])?
    };
    let codec_bos_embed = codec_control_embed.narrow(1, control_len - 1, 1)?;
    let first_input = first_text_embed.add(&codec_bos_embed)?;
    // [1, 1, hidden]

    // Step 6: Concatenate everything
    let prefill_embed = Tensor::cat(&[&role_prefix_embed, &fused_control, &first_input], 1)?;
    // [1, 3 + (control_len - 1) + 1 = control_len + 3, hidden]

    // Step 7: Build trailing text embeddings
    let trailing_embed = if text_token_ids.len() > 1 {
        let mut trailing_ids: Vec<u32> = text_token_ids[1..].to_vec();
        trailing_ids.push(token_ids.tts_eos);
        project_text(&trailing_ids)?
    } else {
        project_text(&[token_ids.tts_eos])?
    };

    Ok((prefill_embed, trailing_embed))
}

/// Text projection MLP: fc1 -> silu -> fc2
fn text_projection_forward(
    x: &Tensor,
    weights: &HashMap<String, Tensor>,
) -> Result<Tensor> {
    let fc1_w = weights.get("talker.text_projection.linear_fc1.weight")
        .ok_or_else(|| anyhow::anyhow!("Missing text_projection fc1 weight"))?;
    let fc1_b = weights.get("talker.text_projection.linear_fc1.bias")
        .ok_or_else(|| anyhow::anyhow!("Missing text_projection fc1 bias"))?;
    let fc2_w = weights.get("talker.text_projection.linear_fc2.weight")
        .ok_or_else(|| anyhow::anyhow!("Missing text_projection fc2 weight"))?;
    let fc2_b = weights.get("talker.text_projection.linear_fc2.bias")
        .ok_or_else(|| anyhow::anyhow!("Missing text_projection fc2 bias"))?;

    let hidden = linear_3d(x, fc1_w, Some(fc1_b))?;
    let hidden = candle_nn::ops::silu(&hidden)?;
    linear_3d(&hidden, fc2_w, Some(fc2_b))
}

/// Linear projection for 3D tensors [batch, seq, features]
fn linear_3d(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let dims = x.dims();
    let (batch, seq, _features) = (dims[0], dims[1], dims[2]);
    let x_2d = x.reshape((batch * seq, x.dim(2)?))?;
    let out_2d = x_2d.matmul(&weight.t()?)?;
    let out_3d = out_2d.reshape((batch, seq, out_2d.dim(1)?))?;
    match bias {
        Some(b) => Ok(out_3d.broadcast_add(b)?),
        None => Ok(out_3d),
    }
}

/// Apply token suppression to logits
///
/// Masks out tokens in range [vocab_size - 1024, vocab_size) except for EOS token
pub fn apply_token_suppression(
    logits: &Tensor,
    vocab_size: usize,
    eos_token_id: u32,
) -> Result<Tensor> {
    let suppress_start = vocab_size - 1024;
    let logits_vec: Vec<f32> = logits.flatten_all()?.to_vec1()?;
    let batch_size = logits.dim(0)?;
    let vocab = logits.dim(1)?;

    let mut suppressed = logits_vec;
    for b in 0..batch_size {
        for v in suppress_start..vocab_size {
            if v as u32 != eos_token_id {
                suppressed[b * vocab + v] = f32::NEG_INFINITY;
            }
        }
    }

    Ok(Tensor::from_vec(suppressed, logits.shape(), logits.device())?)
}

/// Sum embeddings from all 16 codebooks (residual VQ pattern)
pub fn sum_residual_embeddings(embeddings: &[Tensor]) -> Result<Tensor> {
    if embeddings.is_empty() {
        return Err(anyhow::anyhow!("No embeddings to sum"));
    }

    let mut result = embeddings[0].clone();
    for embed in &embeddings[1..] {
        result = result.add(embed)?;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tts_token_ids_default() {
        let ids = TtsTokenIds::default();
        assert_eq!(ids.codec_bos, 2149);
        assert_eq!(ids.codec_eos, 2150);
        assert_eq!(ids.im_start, 151644);
    }

    #[test]
    fn test_tts_generation_config_default() {
        let config = TtsGenerationConfig::default();
        assert_eq!(config.max_frames, 100);
        assert!((config.temperature - 0.7).abs() < 1e-6);
        assert!(config.use_think);
    }
}
