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
use candle_core::{Device, Tensor};

/// Re-exports for convenience
pub use audio::AudioBuffer;
pub use models::config::Qwen3TTSConfig;
pub use models::Qwen3TTSModel;

/// Main TTS interface
pub struct Qwen3TTS {
    /// The underlying model
    model: Qwen3TTSModel,
    /// Text tokenizer
    text_tokenizer: tokenizer::TextTokenizer,
    /// Audio codec for decoding generated tokens to waveform
    audio_codec: models::codec::AudioCodec,
    /// Device to run inference on
    device: Device,
}

impl Qwen3TTS {
    /// Load a model from a HuggingFace model ID or local path
    pub fn from_pretrained(model_id: &str, device: Device) -> Result<Self> {
        tracing::info!("Loading Qwen3-TTS from: {}", model_id);

        // Load config
        let config = Qwen3TTSConfig::from_pretrained(model_id)?;

        // Load text tokenizer
        let text_tokenizer = tokenizer::TextTokenizer::from_pretrained(model_id)?;

        // Load main TTS model
        let model = Qwen3TTSModel::from_pretrained(model_id, &config, &device)?;

        // Load audio codec (tokenizer)
        let codec_id = config.tokenizer_model_id.as_deref().unwrap_or(model_id);
        let audio_codec = models::codec::AudioCodec::from_pretrained(codec_id, &device)?;

        Ok(Self {
            model,
            text_tokenizer,
            audio_codec,
            device,
        })
    }

    /// Synthesize speech from text
    pub fn synthesize(
        &self,
        text: &str,
        options: Option<SynthesisOptions>,
    ) -> Result<AudioBuffer> {
        let options = options.unwrap_or_default();

        // Tokenize text
        let input_ids = self.text_tokenizer.encode(text)?;
        let input_tensor = Tensor::new(input_ids.as_slice(), &self.device)?
            .unsqueeze(0)?; // Add batch dimension

        // Generate codec tokens
        let codec_tokens = self.model.generate(
            &input_tensor,
            options.speaker_embedding.as_ref(),
            &generation::GenerationConfig {
                max_new_tokens: options.max_length,
                temperature: options.temperature,
                top_k: options.top_k,
                top_p: options.top_p,
                repetition_penalty: options.repetition_penalty,
            },
        )?;

        // Decode codec tokens to audio waveform
        let waveform = self.audio_codec.decode(&codec_tokens)?;

        Ok(AudioBuffer::from_tensor(waveform, 24000)?)
    }

    /// Get the device this model is running on
    pub fn device(&self) -> &Device {
        &self.device
    }
}

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
