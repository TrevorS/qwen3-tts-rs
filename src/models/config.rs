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

    /// Head dimension (if not specified, computed as hidden_size / num_attention_heads)
    #[serde(default)]
    pub head_dim_override: Option<usize>,

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

/// Speaker encoder (ECAPA-TDNN) configuration
///
/// Matches `Qwen3TTSSpeakerEncoderConfig` from the Python reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerEncoderConfig {
    /// Input mel spectrogram channels
    #[serde(default = "default_mel_dim")]
    pub mel_dim: usize,
    /// Output embedding dimension
    #[serde(default = "default_enc_dim")]
    pub enc_dim: usize,
    /// Per-block channel counts: [initial_out, se_res2net_0, se_res2net_1, se_res2net_2, mfa_out]
    #[serde(default = "default_enc_channels")]
    pub enc_channels: Vec<usize>,
    /// Kernel sizes per block
    #[serde(default = "default_enc_kernel_sizes")]
    pub enc_kernel_sizes: Vec<usize>,
    /// Dilation per block
    #[serde(default = "default_enc_dilations")]
    pub enc_dilations: Vec<usize>,
    /// Attention channels in ASP
    #[serde(default = "default_enc_attention_channels")]
    pub enc_attention_channels: usize,
    /// Res2Net scale factor
    #[serde(default = "default_enc_res2net_scale")]
    pub enc_res2net_scale: usize,
    /// SE block bottleneck channels
    #[serde(default = "default_enc_se_channels")]
    pub enc_se_channels: usize,
    /// Audio sample rate for mel extraction
    #[serde(default = "default_speaker_sample_rate")]
    pub sample_rate: u32,
}

fn default_mel_dim() -> usize {
    128
}
fn default_enc_dim() -> usize {
    1024
}
fn default_enc_channels() -> Vec<usize> {
    vec![512, 512, 512, 512, 1536]
}
fn default_enc_kernel_sizes() -> Vec<usize> {
    vec![5, 3, 3, 3, 1]
}
fn default_enc_dilations() -> Vec<usize> {
    vec![1, 2, 3, 4, 1]
}
fn default_enc_attention_channels() -> usize {
    128
}
fn default_enc_res2net_scale() -> usize {
    8
}
fn default_enc_se_channels() -> usize {
    128
}
fn default_speaker_sample_rate() -> u32 {
    24000
}

impl Default for SpeakerEncoderConfig {
    fn default() -> Self {
        Self {
            mel_dim: default_mel_dim(),
            enc_dim: default_enc_dim(),
            enc_channels: default_enc_channels(),
            enc_kernel_sizes: default_enc_kernel_sizes(),
            enc_dilations: default_enc_dilations(),
            enc_attention_channels: default_enc_attention_channels(),
            enc_res2net_scale: default_enc_res2net_scale(),
            enc_se_channels: default_enc_se_channels(),
            sample_rate: default_speaker_sample_rate(),
        }
    }
}

/// Model variant type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    /// Voice cloning from reference audio (includes speaker encoder)
    Base,
    /// 9 preset speakers with optional instruction control
    CustomVoice,
    /// Novel voice creation from text descriptions
    VoiceDesign,
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Base => write!(f, "base"),
            Self::CustomVoice => write!(f, "custom_voice"),
            Self::VoiceDesign => write!(f, "voice_design"),
        }
    }
}

/// Parsed model configuration from HuggingFace config.json
///
/// Extracts the subset of fields needed for Rust model construction.
/// All 5 model variants (0.6B-Base, 0.6B-CustomVoice, 1.7B-Base,
/// 1.7B-CustomVoice, 1.7B-VoiceDesign) can be auto-detected from
/// their config.json.
#[derive(Debug, Clone)]
pub struct ParsedModelConfig {
    pub model_type: ModelType,
    pub model_size: String,
    // Talker
    pub talker_hidden_size: usize,
    pub talker_intermediate_size: usize,
    pub talker_num_hidden_layers: usize,
    pub talker_num_attention_heads: usize,
    pub talker_num_key_value_heads: usize,
    pub talker_head_dim: usize,
    pub talker_vocab_size: usize,
    pub talker_text_vocab_size: usize,
    pub talker_text_hidden_size: usize,
    pub talker_rms_norm_eps: f64,
    pub talker_rope_theta: f64,
    pub talker_max_position_embeddings: usize,
    pub mrope_section: Option<[usize; 3]>,
    // Code predictor
    pub cp_hidden_size: usize,
    pub cp_intermediate_size: usize,
    pub cp_num_hidden_layers: usize,
    pub cp_num_attention_heads: usize,
    pub cp_num_key_value_heads: usize,
    pub cp_head_dim: usize,
    pub cp_vocab_size: usize,
    pub cp_num_code_groups: usize,
    pub cp_rms_norm_eps: f64,
    pub cp_rope_theta: f64,
    // Speaker encoder
    pub speaker_encoder_config: Option<SpeakerEncoderConfig>,
}

impl ParsedModelConfig {
    /// Parse from a config.json file
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config from {}", path.display()))?;
        let v: serde_json::Value = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse config from {}", path.display()))?;

        let model_type = match v["tts_model_type"].as_str().unwrap_or("base") {
            "custom_voice" => ModelType::CustomVoice,
            "voice_design" => ModelType::VoiceDesign,
            _ => ModelType::Base,
        };
        let model_size = v["tts_model_size"]
            .as_str()
            .unwrap_or("unknown")
            .to_string();

        let t = &v["talker_config"];
        let cp = &t["code_predictor_config"];

        let talker_hidden_size = t["hidden_size"].as_u64().unwrap_or(1024) as usize;
        let talker_intermediate_size = t["intermediate_size"].as_u64().unwrap_or(3072) as usize;
        let talker_num_hidden_layers = t["num_hidden_layers"].as_u64().unwrap_or(28) as usize;
        let talker_num_attention_heads = t["num_attention_heads"].as_u64().unwrap_or(16) as usize;
        let talker_num_key_value_heads = t["num_key_value_heads"].as_u64().unwrap_or(8) as usize;
        let talker_head_dim = t["head_dim"].as_u64().unwrap_or(128) as usize;
        let talker_vocab_size = t["vocab_size"].as_u64().unwrap_or(3072) as usize;
        let talker_text_vocab_size = t["text_vocab_size"].as_u64().unwrap_or(151936) as usize;
        let talker_text_hidden_size = t["text_hidden_size"].as_u64().unwrap_or(2048) as usize;
        let talker_rms_norm_eps = t["rms_norm_eps"].as_f64().unwrap_or(1e-6);
        let talker_rope_theta = t["rope_theta"].as_f64().unwrap_or(1000000.0);
        let talker_max_position_embeddings =
            t["max_position_embeddings"].as_u64().unwrap_or(32768) as usize;

        // Parse MRoPE section from rope_scaling
        let mrope_section = t["rope_scaling"]["mrope_section"]
            .as_array()
            .and_then(|arr| {
                if arr.len() == 3 {
                    Some([
                        arr[0].as_u64()? as usize,
                        arr[1].as_u64()? as usize,
                        arr[2].as_u64()? as usize,
                    ])
                } else {
                    None
                }
            });

        let cp_hidden_size = cp["hidden_size"].as_u64().unwrap_or(1024) as usize;
        let cp_intermediate_size = cp["intermediate_size"].as_u64().unwrap_or(3072) as usize;
        let cp_num_hidden_layers = cp["num_hidden_layers"].as_u64().unwrap_or(5) as usize;
        let cp_num_attention_heads = cp["num_attention_heads"].as_u64().unwrap_or(16) as usize;
        let cp_num_key_value_heads = cp["num_key_value_heads"].as_u64().unwrap_or(8) as usize;
        let cp_head_dim = cp["head_dim"].as_u64().unwrap_or(128) as usize;
        let cp_vocab_size = cp["vocab_size"].as_u64().unwrap_or(2048) as usize;
        let cp_num_code_groups = cp["num_code_groups"].as_u64().unwrap_or(16) as usize;
        let cp_rms_norm_eps = cp["rms_norm_eps"].as_f64().unwrap_or(1e-6);
        let cp_rope_theta = cp["rope_theta"].as_f64().unwrap_or(1000000.0);

        let speaker_encoder_config = if v["speaker_encoder_config"].is_object() {
            let se = &v["speaker_encoder_config"];
            Some(SpeakerEncoderConfig {
                enc_dim: se["enc_dim"].as_u64().unwrap_or(1024) as usize,
                sample_rate: se["sample_rate"].as_u64().unwrap_or(24000) as u32,
                ..Default::default()
            })
        } else {
            None
        };

        Ok(Self {
            model_type,
            model_size,
            talker_hidden_size,
            talker_intermediate_size,
            talker_num_hidden_layers,
            talker_num_attention_heads,
            talker_num_key_value_heads,
            talker_head_dim,
            talker_vocab_size,
            talker_text_vocab_size,
            talker_text_hidden_size,
            talker_rms_norm_eps,
            talker_rope_theta,
            talker_max_position_embeddings,
            mrope_section,
            cp_hidden_size,
            cp_intermediate_size,
            cp_num_hidden_layers,
            cp_num_attention_heads,
            cp_num_key_value_heads,
            cp_head_dim,
            cp_vocab_size,
            cp_num_code_groups,
            cp_rms_norm_eps,
            cp_rope_theta,
            speaker_encoder_config,
        })
    }

    /// Human-readable label, e.g. "1.7B CustomVoice"
    pub fn label(&self) -> String {
        let size = match self.model_size.as_str() {
            "0b6" => "0.6B",
            "1b7" => "1.7B",
            other => other,
        };
        let variant = match self.model_type {
            ModelType::Base => "Base",
            ModelType::CustomVoice => "CustomVoice",
            ModelType::VoiceDesign => "VoiceDesign",
        };
        format!("{} {}", size, variant)
    }
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
            head_dim_override: None,
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

        // Try downloading from HuggingFace Hub
        #[cfg(feature = "hub")]
        {
            tracing::info!("Downloading config from HuggingFace Hub: {}", model_id);
            let api =
                hf_hub::api::sync::Api::new().context("Failed to create HuggingFace API client")?;
            let repo = api.model(model_id.to_string());
            let config_file = repo
                .get("config.json")
                .context("Failed to download config.json from HuggingFace Hub")?;
            Self::from_file(&config_file)
        }

        #[cfg(not(feature = "hub"))]
        anyhow::bail!(
            "Remote model loading requires the `hub` feature. \
             Either enable it or download the model locally first: {}",
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

    /// Head dimension (uses override if set, otherwise computes from hidden_size)
    pub fn head_dim(&self) -> usize {
        self.head_dim_override
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen3_tts_config_default() {
        let config = Qwen3TTSConfig::default();
        assert_eq!(config.model_type, "qwen3_tts");
        assert_eq!(config.vocab_size, 151936);
        assert_eq!(config.hidden_size, 896);
        assert_eq!(config.intermediate_size, 4864);
        assert_eq!(config.num_hidden_layers, 24);
        assert_eq!(config.num_attention_heads, 14);
        assert!(config.num_key_value_heads.is_none());
        assert_eq!(config.max_position_embeddings, 32768);
        assert!((config.rope_theta - 1000000.0).abs() < 1e-6);
        assert!((config.rms_norm_eps - 1e-6).abs() < 1e-12);
        assert!(config.sliding_window.is_none());
        assert_eq!(config.num_codebook_groups, 16);
        assert_eq!(config.codebook_size, 2048);
        assert_eq!(config.speaker_embed_dim, 1024);
        assert!(!config.use_flash_attention);
    }

    #[test]
    fn test_num_kv_heads_default() {
        let config = Qwen3TTSConfig::default();
        // When num_key_value_heads is None, should return num_attention_heads
        assert_eq!(config.num_kv_heads(), 14);
    }

    #[test]
    fn test_num_kv_heads_explicit() {
        let config = Qwen3TTSConfig {
            num_key_value_heads: Some(7),
            ..Default::default()
        };
        assert_eq!(config.num_kv_heads(), 7);
    }

    #[test]
    fn test_head_dim() {
        let config = Qwen3TTSConfig::default();
        // hidden_size=896, num_attention_heads=14 => head_dim=64
        assert_eq!(config.head_dim(), 64);
    }

    #[test]
    fn test_head_dim_custom() {
        let config = Qwen3TTSConfig {
            hidden_size: 1024,
            num_attention_heads: 16,
            ..Default::default()
        };
        assert_eq!(config.head_dim(), 64);
    }

    #[test]
    fn test_config_serialization() {
        let config = Qwen3TTSConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: Qwen3TTSConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.model_type, config.model_type);
        assert_eq!(parsed.vocab_size, config.vocab_size);
        assert_eq!(parsed.hidden_size, config.hidden_size);
    }

    #[test]
    fn test_config_deserialization_with_defaults() {
        // Deserialize minimal JSON - should use defaults
        let json = r#"{"model_type": "test_model"}"#;
        let config: Qwen3TTSConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.model_type, "test_model");
        // Other fields should be defaults
        assert_eq!(config.vocab_size, 151936);
        assert_eq!(config.hidden_size, 896);
    }

    #[test]
    fn test_config_deserialization_custom_values() {
        let json = r#"{
            "model_type": "custom",
            "vocab_size": 50000,
            "hidden_size": 512,
            "num_attention_heads": 8,
            "num_key_value_heads": 4
        }"#;
        let config: Qwen3TTSConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.model_type, "custom");
        assert_eq!(config.vocab_size, 50000);
        assert_eq!(config.hidden_size, 512);
        assert_eq!(config.num_attention_heads, 8);
        assert_eq!(config.num_key_value_heads, Some(4));
    }

    #[test]
    fn test_talker_config() {
        let talker = TalkerConfig {
            hidden_size: 512,
            intermediate_size: 2048,
            num_hidden_layers: 12,
            num_attention_heads: 8,
            num_key_value_heads: Some(4),
            vocab_size: 30000,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            sliding_window: Some(4096),
        };
        let json = serde_json::to_string(&talker).unwrap();
        let parsed: TalkerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.hidden_size, 512);
        assert_eq!(parsed.num_hidden_layers, 12);
        assert_eq!(parsed.sliding_window, Some(4096));
    }

    #[test]
    fn test_from_pretrained_nonexistent() {
        let result = Qwen3TTSConfig::from_pretrained("/nonexistent/path");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_file_nonexistent() {
        let result = Qwen3TTSConfig::from_file("/nonexistent/config.json");
        assert!(result.is_err());
    }

    #[test]
    fn test_config_clone() {
        let config = Qwen3TTSConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.model_type, config.model_type);
        assert_eq!(cloned.vocab_size, config.vocab_size);
    }

    #[test]
    fn test_config_debug() {
        let config = Qwen3TTSConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("qwen3_tts"));
        assert!(debug_str.contains("896"));
    }

    #[test]
    fn test_parsed_model_config_06b_base() {
        let path = Path::new("test_data/models/0.6b-base/config.json");
        if !path.exists() {
            return; // skip if test data not present
        }
        let cfg = ParsedModelConfig::from_file(path).unwrap();
        assert_eq!(cfg.model_type, ModelType::Base);
        assert_eq!(cfg.model_size, "0b6");
        assert_eq!(cfg.talker_hidden_size, 1024);
        assert_eq!(cfg.talker_intermediate_size, 3072);
        assert_eq!(cfg.cp_hidden_size, 1024);
        assert_eq!(cfg.mrope_section, Some([24, 20, 20]));
        assert!(cfg.speaker_encoder_config.is_some());
        assert_eq!(cfg.speaker_encoder_config.as_ref().unwrap().enc_dim, 1024);
        assert_eq!(cfg.label(), "0.6B Base");
    }

    #[test]
    fn test_parsed_model_config_17b_base() {
        let path = Path::new("test_data/models/1.7b-base/config.json");
        if !path.exists() {
            return;
        }
        let cfg = ParsedModelConfig::from_file(path).unwrap();
        assert_eq!(cfg.model_type, ModelType::Base);
        assert_eq!(cfg.model_size, "1b7");
        assert_eq!(cfg.talker_hidden_size, 2048);
        assert_eq!(cfg.talker_intermediate_size, 6144);
        assert_eq!(cfg.cp_hidden_size, 1024);
        assert_eq!(cfg.mrope_section, Some([24, 20, 20]));
        assert!(cfg.speaker_encoder_config.is_some());
        assert_eq!(cfg.speaker_encoder_config.as_ref().unwrap().enc_dim, 2048);
        assert_eq!(cfg.label(), "1.7B Base");
    }

    #[test]
    fn test_parsed_model_config_06b_customvoice() {
        let path = Path::new("test_data/models/0.6b-customvoice/config.json");
        if !path.exists() {
            return;
        }
        let cfg = ParsedModelConfig::from_file(path).unwrap();
        assert_eq!(cfg.model_type, ModelType::CustomVoice);
        assert_eq!(cfg.model_size, "0b6");
        assert_eq!(cfg.talker_hidden_size, 1024);
        assert!(cfg.speaker_encoder_config.is_none());
        assert_eq!(cfg.label(), "0.6B CustomVoice");
    }

    #[test]
    fn test_parsed_model_config_17b_voicedesign() {
        let path = Path::new("test_data/models/1.7b-voicedesign/config.json");
        if !path.exists() {
            return;
        }
        let cfg = ParsedModelConfig::from_file(path).unwrap();
        assert_eq!(cfg.model_type, ModelType::VoiceDesign);
        assert_eq!(cfg.model_size, "1b7");
        assert_eq!(cfg.talker_hidden_size, 2048);
        assert!(cfg.speaker_encoder_config.is_none());
        assert_eq!(cfg.label(), "1.7B VoiceDesign");
    }

    #[test]
    fn test_model_type_display() {
        assert_eq!(format!("{}", ModelType::Base), "base");
        assert_eq!(format!("{}", ModelType::CustomVoice), "custom_voice");
        assert_eq!(format!("{}", ModelType::VoiceDesign), "voice_design");
    }

    /// Verify that from_parsed produces correct TalkerConfig for 0.6B (hidden=1024)
    #[test]
    fn test_from_parsed_talker_06b() {
        let path = Path::new("test_data/models/0.6b-base/config.json");
        if !path.exists() {
            return;
        }
        let parsed = ParsedModelConfig::from_file(path).unwrap();
        let tc = super::super::talker::TalkerConfig::from_parsed(&parsed);
        assert_eq!(tc.hidden_size, 1024);
        assert_eq!(tc.intermediate_size, 3072);
        assert_eq!(tc.text_embed_dim, 2048);
        assert_eq!(tc.num_hidden_layers, 28);
        assert_eq!(tc.num_attention_heads, 16);
        assert_eq!(tc.num_key_value_heads, 8);
        assert_eq!(tc.head_dim, 128);
        assert_eq!(tc.codec_vocab_size, 3072);
        assert_eq!(tc.mrope_section, Some([24, 20, 20]));
    }

    /// Verify that from_parsed produces correct TalkerConfig for 1.7B (hidden=2048)
    #[test]
    fn test_from_parsed_talker_17b() {
        let path = Path::new("test_data/models/1.7b-base/config.json");
        if !path.exists() {
            return;
        }
        let parsed = ParsedModelConfig::from_file(path).unwrap();
        let tc = super::super::talker::TalkerConfig::from_parsed(&parsed);
        assert_eq!(tc.hidden_size, 2048);
        assert_eq!(tc.intermediate_size, 6144);
        assert_eq!(tc.text_embed_dim, 2048);
        assert_eq!(tc.mrope_section, Some([24, 20, 20]));
    }

    /// Verify CodePredictorConfig: 0.6B has no projection, 1.7B needs codec_embed_dim=2048
    #[test]
    fn test_from_parsed_code_predictor() {
        let path_06b = Path::new("test_data/models/0.6b-base/config.json");
        let path_17b = Path::new("test_data/models/1.7b-base/config.json");

        if path_06b.exists() {
            let parsed = ParsedModelConfig::from_file(path_06b).unwrap();
            let cp = super::super::code_predictor::CodePredictorConfig::from_parsed(&parsed);
            assert_eq!(cp.hidden_size, 1024);
            assert_eq!(cp.intermediate_size, 3072);
            assert_eq!(cp.num_hidden_layers, 5);
            assert_eq!(cp.vocab_size, 2048);
            assert!(cp.codec_embed_dim.is_none()); // 0.6B: talker=1024, cp=1024, no projection
        }

        if path_17b.exists() {
            let parsed = ParsedModelConfig::from_file(path_17b).unwrap();
            let cp = super::super::code_predictor::CodePredictorConfig::from_parsed(&parsed);
            assert_eq!(cp.hidden_size, 1024);
            assert_eq!(cp.codec_embed_dim, Some(2048)); // 1.7B: needs 2048→1024 projection
        }
    }
}
