//! Model configuration for Qwen3-TTS

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Main Qwen3-TTS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen3TTSConfig {
    /// Model architecture type
    #[serde(default = "default_model_type")]
    pub model_type: String,

    /// Vocabulary size for text tokens
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,

    /// Hidden dimension
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,

    /// Intermediate size in MLP
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,

    /// Number of transformer layers
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,

    /// Number of attention heads
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,

    /// Number of key-value heads (for GQA)
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    /// Maximum sequence length
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    /// RoPE theta base
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,

    /// RMS norm epsilon
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,

    /// Sliding window size for attention
    #[serde(default)]
    pub sliding_window: Option<usize>,

    /// Number of audio codec groups/quantizers
    #[serde(default = "default_num_codebook_groups")]
    pub num_codebook_groups: usize,

    /// Codebook size per quantizer
    #[serde(default = "default_codebook_size")]
    pub codebook_size: usize,

    /// Speaker embedding dimension
    #[serde(default = "default_speaker_embed_dim")]
    pub speaker_embed_dim: usize,

    /// Tokenizer model ID (for audio codec)
    #[serde(default)]
    pub tokenizer_model_id: Option<String>,

    /// Talker configuration
    #[serde(default)]
    pub talker: Option<TalkerConfig>,

    /// Use flash attention
    #[serde(default)]
    pub use_flash_attention: bool,
}

/// Talker (text-to-code) model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TalkerConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub vocab_size: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub sliding_window: Option<usize>,
}

// Default values matching Qwen3-TTS-12Hz-0.6B-Base
fn default_model_type() -> String {
    "qwen3_tts".to_string()
}

fn default_vocab_size() -> usize {
    151936
}

fn default_hidden_size() -> usize {
    896
}

fn default_intermediate_size() -> usize {
    4864
}

fn default_num_hidden_layers() -> usize {
    24
}

fn default_num_attention_heads() -> usize {
    14
}

fn default_max_position_embeddings() -> usize {
    32768
}

fn default_rope_theta() -> f64 {
    1000000.0
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}

fn default_num_codebook_groups() -> usize {
    16
}

fn default_codebook_size() -> usize {
    2048
}

fn default_speaker_embed_dim() -> usize {
    1024
}

impl Default for Qwen3TTSConfig {
    fn default() -> Self {
        Self {
            model_type: default_model_type(),
            vocab_size: default_vocab_size(),
            hidden_size: default_hidden_size(),
            intermediate_size: default_intermediate_size(),
            num_hidden_layers: default_num_hidden_layers(),
            num_attention_heads: default_num_attention_heads(),
            num_key_value_heads: None,
            max_position_embeddings: default_max_position_embeddings(),
            rope_theta: default_rope_theta(),
            rms_norm_eps: default_rms_norm_eps(),
            sliding_window: None,
            num_codebook_groups: default_num_codebook_groups(),
            codebook_size: default_codebook_size(),
            speaker_embed_dim: default_speaker_embed_dim(),
            tokenizer_model_id: None,
            talker: None,
            use_flash_attention: false,
        }
    }
}

impl Qwen3TTSConfig {
    /// Load configuration from a HuggingFace model ID or local path
    pub fn from_pretrained(model_id: &str) -> Result<Self> {
        // Try local path first
        let config_path = Path::new(model_id).join("config.json");
        if config_path.exists() {
            return Self::from_file(&config_path);
        }

        // TODO: Download from HuggingFace Hub
        // For now, return error
        anyhow::bail!(
            "Remote model loading not yet implemented. \
             Please download the model locally first: {}",
            model_id
        )
    }

    /// Load configuration from a local JSON file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config from {}", path.display()))?;

        let config: Self = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse config from {}", path.display()))?;

        Ok(config)
    }

    /// Get number of key-value heads, defaulting to num_attention_heads if not set
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// Audio codec configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioCodecConfig {
    /// Codec type: "12hz" or "25hz"
    #[serde(default = "default_codec_type")]
    pub codec_type: String,

    /// Sample rate
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,

    /// Number of quantizers
    #[serde(default = "default_num_quantizers")]
    pub num_quantizers: usize,

    /// Codebook size
    #[serde(default = "default_codebook_size")]
    pub codebook_size: usize,

    /// Frame rate (Hz)
    #[serde(default = "default_frame_rate")]
    pub frame_rate: f32,

    /// Decoder hidden size
    #[serde(default = "default_decoder_hidden_size")]
    pub decoder_hidden_size: usize,

    /// Decoder transformer layers
    #[serde(default = "default_decoder_num_layers")]
    pub decoder_num_layers: usize,
}

fn default_codec_type() -> String {
    "12hz".to_string()
}

fn default_sample_rate() -> u32 {
    24000
}

fn default_num_quantizers() -> usize {
    16
}

fn default_frame_rate() -> f32 {
    12.5
}

fn default_decoder_hidden_size() -> usize {
    1024
}

fn default_decoder_num_layers() -> usize {
    8
}

impl Default for AudioCodecConfig {
    fn default() -> Self {
        Self {
            codec_type: default_codec_type(),
            sample_rate: default_sample_rate(),
            num_quantizers: default_num_quantizers(),
            codebook_size: default_codebook_size(),
            frame_rate: default_frame_rate(),
            decoder_hidden_size: default_decoder_hidden_size(),
            decoder_num_layers: default_decoder_num_layers(),
        }
    }
}
