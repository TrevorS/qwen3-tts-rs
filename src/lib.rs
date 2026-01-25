//! # Qwen3-TTS
//!
//! Pure Rust inference for Qwen3-TTS text-to-speech model.
//!
//! ## Features
//!
//! - **CPU**: Default, with optional MKL/Accelerate for faster BLAS
//! - **CUDA**: NVIDIA GPU acceleration
//! - **Metal**: Apple Silicon GPU acceleration
//!
//! ## Architecture
//!
//! The TTS pipeline consists of:
//! 1. **TalkerModel**: Generates semantic tokens from text autoregressively
//! 2. **CodePredictor**: Generates acoustic tokens (15) for each semantic token
//! 3. **Decoder12Hz**: Converts codec tokens to audio waveform
//!
//! ## Example
//!
//! ```rust,ignore
//! use qwen3_tts::{Qwen3TTS, Config};
//!
//! let model = Qwen3TTS::from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base")?;
//! let audio = model.synthesize("Hello, world!", None)?;
//! audio.save("output.wav")?;
//! ```

pub mod audio;
pub mod generation;
pub mod models;
pub mod tokenizer;

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use std::collections::HashMap;
use std::path::Path;

use models::codec::Decoder12Hz;
use models::talker::TalkerModel;
use models::KVCache;

/// Re-exports for convenience
pub use audio::AudioBuffer;
pub use models::config::Qwen3TTSConfig;
pub use models::Qwen3TTSModel;
pub use models::{CodePredictor, CodePredictorConfig, TalkerConfig, TalkerModel as Talker};
pub use models::talker::{Language, Speaker, codec_tokens, special_tokens, tts_tokens};

/// Main TTS interface using proper autoregressive pipeline
pub struct Qwen3TTS {
    /// Talker model for semantic token generation
    talker: TalkerModel,
    /// Code predictor for acoustic token generation
    code_predictor: CodePredictor,
    /// 12Hz decoder for audio synthesis
    decoder: Decoder12Hz,
    /// Text tokenizer
    text_tokenizer: tokenizer::TextTokenizer,
    /// Device to run inference on
    device: Device,
}

impl Qwen3TTS {
    /// Load a model from a HuggingFace model ID or local path
    pub fn from_pretrained(model_id: &str, device: Device) -> Result<Self> {
        tracing::info!("Loading Qwen3-TTS from: {}", model_id);

        // Load text tokenizer
        let text_tokenizer = tokenizer::TextTokenizer::from_pretrained(model_id)?;

        // Load model weights
        let model_path = Path::new(model_id).join("model.safetensors");
        if !model_path.exists() {
            anyhow::bail!(
                "Model weights not found at {}. Please download the model first.",
                model_path.display()
            );
        }

        let weights = Self::load_weights(&model_path, &device)?;

        // Create TalkerModel
        let talker = TalkerModel::from_weights(&weights, &device)?;

        // Create CodePredictor
        let cp_config = CodePredictorConfig::default();
        let cp_weights = Self::filter_weights(&weights, "talker.code_predictor.");
        let cp_vb = candle_nn::VarBuilder::from_tensors(cp_weights, DType::F32, &device);
        let code_predictor = CodePredictor::new(cp_config, cp_vb)?;

        // Load speech tokenizer for decoder
        let st_path = Path::new(model_id).join("speech_tokenizer/model.safetensors");
        let st_weights = if st_path.exists() {
            Self::load_weights(&st_path, &device)?
        } else {
            // Fall back to looking in parent dir
            let alt_path = Path::new(model_id)
                .parent()
                .map(|p| p.join("speech_tokenizer/model.safetensors"));
            if let Some(p) = alt_path {
                if p.exists() {
                    Self::load_weights(&p, &device)?
                } else {
                    anyhow::bail!("Speech tokenizer weights not found");
                }
            } else {
                anyhow::bail!("Speech tokenizer weights not found");
            }
        };

        // Create Decoder12Hz
        let decoder = Decoder12Hz::from_weights(&st_weights, Default::default())?;

        Ok(Self {
            talker,
            code_predictor,
            decoder,
            text_tokenizer,
            device,
        })
    }

    /// Load from pre-loaded weight tensors
    pub fn from_weights(
        model_weights: &HashMap<String, Tensor>,
        decoder_weights: &HashMap<String, Tensor>,
        text_tokenizer: tokenizer::TextTokenizer,
        device: &Device,
    ) -> Result<Self> {
        // Create TalkerModel
        let talker = TalkerModel::from_weights(model_weights, device)?;

        // Create CodePredictor
        let cp_config = CodePredictorConfig::default();
        let cp_weights = Self::filter_weights(model_weights, "talker.code_predictor.");
        let cp_vb = candle_nn::VarBuilder::from_tensors(cp_weights, DType::F32, device);
        let code_predictor = CodePredictor::new(cp_config, cp_vb)?;

        // Create Decoder12Hz
        let decoder = Decoder12Hz::from_weights(decoder_weights, Default::default())?;

        Ok(Self {
            talker,
            code_predictor,
            decoder,
            text_tokenizer,
            device: device.clone(),
        })
    }

    /// Synthesize speech from text using proper autoregressive pipeline
    ///
    /// Pipeline:
    /// 1. Tokenize text
    /// 2. Prefill TalkerModel with text tokens
    /// 3. Autoregressively generate semantic tokens
    /// 4. For each semantic token, generate 15 acoustic tokens via CodePredictor
    /// 5. Decode accumulated codes to audio via Decoder12Hz
    pub fn synthesize(&self, text: &str, options: Option<SynthesisOptions>) -> Result<AudioBuffer> {
        let options = options.unwrap_or_default();

        // Tokenize text
        let input_ids = self.text_tokenizer.encode(text)?;
        let input_tensor = Tensor::new(input_ids.as_slice(), &self.device)?.unsqueeze(0)?;

        // Generate codes using proper pipeline
        let codes = self.generate_codes(
            &input_tensor,
            &generation::GenerationConfig {
                max_new_tokens: options.max_length,
                temperature: options.temperature,
                top_k: options.top_k,
                top_p: options.top_p,
                repetition_penalty: options.repetition_penalty,
                eos_token_id: options.eos_token_id,
            },
        )?;

        // Decode to audio
        let waveform = self.decoder.decode(&codes)?;

        AudioBuffer::from_tensor(waveform, 24000)
    }

    /// Generate codec tokens using proper autoregressive pipeline
    ///
    /// Returns tensor of shape [batch, 16, num_frames]
    pub fn generate_codes(
        &self,
        input_ids: &Tensor,
        config: &generation::GenerationConfig,
    ) -> Result<Tensor> {
        let mut talker_kv_caches: Vec<KVCache> = self.talker.new_kv_caches();

        // Prefill talker with text
        let (hidden, logits) = self.talker.prefill(input_ids, &mut talker_kv_caches)?;
        let mut offset = input_ids.dim(1)?;

        // Get hidden state for last position (input to code predictor)
        let seq_len = hidden.dim(1)?;
        let mut last_hidden = hidden.i((.., seq_len - 1..seq_len, ..))?;

        // Sample first semantic token
        let first_token = generation::sample(&logits.squeeze(1)?, config)?;
        let first_token_id: u32 = first_token.flatten_all()?.to_vec1::<u32>()?[0];

        // Check for EOS
        if let Some(eos_id) = config.eos_token_id {
            if first_token_id == eos_id {
                // Return empty codes
                return Ok(Tensor::zeros((1, 16, 0), DType::I64, &self.device)?);
            }
        }

        // Collect all frames: Vec<[semantic, acoustic_0..14]>
        let mut all_codes: Vec<Vec<u32>> = Vec::new();

        // Generate first frame's acoustic tokens
        let semantic_embed = self.talker.get_codec_embedding(first_token_id)?;
        let acoustic_codes = self.code_predictor.generate_acoustic_codes(&last_hidden, &semantic_embed)?;
        let mut frame_codes = vec![first_token_id];
        frame_codes.extend(acoustic_codes);
        all_codes.push(frame_codes);

        // Generate remaining frames
        for _ in 1..config.max_new_tokens {
            // Generate next semantic token
            let prev_token = all_codes.last().unwrap()[0];
            let (hidden, logits) = self.talker.generate_step(prev_token, &mut talker_kv_caches, offset)?;
            offset += 1;
            last_hidden = hidden;

            // Sample semantic token
            let next_token = generation::sample(&logits.squeeze(1)?, config)?;
            let next_token_id: u32 = next_token.flatten_all()?.to_vec1::<u32>()?[0];

            // Check for EOS
            if let Some(eos_id) = config.eos_token_id {
                if next_token_id == eos_id {
                    break;
                }
            }

            // Generate acoustic tokens for this frame
            let semantic_embed = self.talker.get_codec_embedding(next_token_id)?;
            let acoustic_codes = self.code_predictor.generate_acoustic_codes(&last_hidden, &semantic_embed)?;
            let mut frame_codes = vec![next_token_id];
            frame_codes.extend(acoustic_codes);
            all_codes.push(frame_codes);
        }

        // Convert to tensor [1, 16, num_frames]
        self.codes_to_tensor(&all_codes)
    }

    /// Convert list of frame codes to tensor [batch, 16, num_frames]
    fn codes_to_tensor(&self, codes: &[Vec<u32>]) -> Result<Tensor> {
        let num_frames = codes.len();
        if num_frames == 0 {
            return Ok(Tensor::zeros((1, 16, 0), DType::I64, &self.device)?);
        }

        let mut data = vec![0i64; 16 * num_frames];
        for (frame, frame_codes) in codes.iter().enumerate() {
            for (q, &code) in frame_codes.iter().enumerate() {
                // Layout: [q0_f0, q0_f1, ...], [q1_f0, q1_f1, ...], ...
                data[q * num_frames + frame] = code as i64;
            }
        }

        Ok(Tensor::from_vec(data, (1, 16, num_frames), &self.device)?)
    }

    /// Get the device this model is running on
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Load weights from safetensors file
    fn load_weights(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
        let tensors: HashMap<String, Tensor> = candle_core::safetensors::load(path, device)?;
        // Convert BF16 to F32
        let tensors: HashMap<String, Tensor> = tensors
            .into_iter()
            .map(|(name, tensor)| {
                let converted = if tensor.dtype() == DType::BF16 {
                    tensor.to_dtype(DType::F32).unwrap()
                } else {
                    tensor
                };
                (name, converted)
            })
            .collect();
        Ok(tensors)
    }

    /// Filter weights by prefix, removing the prefix from keys
    fn filter_weights(weights: &HashMap<String, Tensor>, prefix: &str) -> HashMap<String, Tensor> {
        weights
            .iter()
            .filter_map(|(k, v)| {
                if k.starts_with(prefix) {
                    Some((k.strip_prefix(prefix).unwrap().to_string(), v.clone()))
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Audio end-of-sequence token ID for Qwen3-TTS
pub const AUDIO_EOS_TOKEN_ID: u32 = 151670;

/// Options for speech synthesis
#[derive(Debug, Clone)]
pub struct SynthesisOptions {
    /// Maximum number of tokens to generate
    pub max_length: usize,
    /// Sampling temperature (higher = more random)
    pub temperature: f64,
    /// Top-k sampling
    pub top_k: usize,
    /// Top-p (nucleus) sampling
    pub top_p: f64,
    /// Repetition penalty
    pub repetition_penalty: f64,
    /// Optional speaker embedding for voice cloning
    pub speaker_embedding: Option<Tensor>,
    /// Language code (e.g., "en", "zh")
    pub language: Option<String>,
    /// End-of-sequence token ID (defaults to audio_end token 151670)
    pub eos_token_id: Option<u32>,
}

impl Default for SynthesisOptions {
    fn default() -> Self {
        Self {
            max_length: 2048,
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.0,
            speaker_embedding: None,
            language: None,
            eos_token_id: Some(AUDIO_EOS_TOKEN_ID),
        }
    }
}

/// Select the best available device
pub fn auto_device() -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        if let Ok(device) = Device::cuda_if_available(0) {
            if device.is_cuda() {
                tracing::info!("Using CUDA device");
                return Ok(device);
            }
        }
    }

    #[cfg(feature = "metal")]
    {
        if let Ok(device) = Device::new_metal(0) {
            tracing::info!("Using Metal device");
            return Ok(device);
        }
    }

    tracing::info!("Using CPU device");
    Ok(Device::Cpu)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthesis_options_default() {
        let options = SynthesisOptions::default();
        assert_eq!(options.max_length, 2048);
        assert!((options.temperature - 0.7).abs() < 1e-6);
        assert_eq!(options.top_k, 50);
        assert!((options.top_p - 0.9).abs() < 1e-6);
        assert!((options.repetition_penalty - 1.0).abs() < 1e-6);
        assert!(options.speaker_embedding.is_none());
        assert!(options.language.is_none());
        assert_eq!(options.eos_token_id, Some(AUDIO_EOS_TOKEN_ID));
    }

    #[test]
    fn test_synthesis_options_custom() {
        let options = SynthesisOptions {
            max_length: 512,
            temperature: 0.5,
            top_k: 10,
            top_p: 0.8,
            repetition_penalty: 1.2,
            speaker_embedding: None,
            language: Some("en".to_string()),
            eos_token_id: Some(AUDIO_EOS_TOKEN_ID),
        };
        assert_eq!(options.max_length, 512);
        assert!((options.temperature - 0.5).abs() < 1e-6);
        assert_eq!(options.language, Some("en".to_string()));
        assert_eq!(options.eos_token_id, Some(151670));
    }

    #[test]
    fn test_synthesis_options_clone() {
        let options = SynthesisOptions::default();
        let cloned = options.clone();
        assert_eq!(cloned.max_length, options.max_length);
        assert_eq!(cloned.top_k, options.top_k);
    }

    #[test]
    fn test_synthesis_options_debug() {
        let options = SynthesisOptions::default();
        let debug_str = format!("{:?}", options);
        assert!(debug_str.contains("max_length"));
        assert!(debug_str.contains("2048"));
    }

    #[test]
    fn test_auto_device() {
        // Should always succeed on CPU
        let device = auto_device().unwrap();
        // Just verify it returns a valid device
        assert!(
            matches!(device, Device::Cpu)
                || matches!(device, Device::Cuda(_))
                || matches!(device, Device::Metal(_))
        );
    }

    #[test]
    fn test_audio_buffer_reexport() {
        // Verify re-exports work
        let buffer = AudioBuffer::new(vec![0.0f32; 100], 24000);
        assert_eq!(buffer.sample_rate, 24000);
    }

    #[test]
    fn test_config_reexport() {
        // Verify config re-export works
        let config = Qwen3TTSConfig::default();
        assert_eq!(config.model_type, "qwen3_tts");
    }
}
