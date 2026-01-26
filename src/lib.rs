//! # Qwen3-TTS
//!
//! Pure Rust inference for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS),
//! a high-quality text-to-speech model from Alibaba.
//!
//! ## Features
//!
//! - **CPU inference** with optional MKL/Accelerate for faster BLAS operations
//! - **CUDA** support for NVIDIA GPU acceleration
//! - **Metal** support for Apple Silicon
//! - **Streaming-friendly** architecture with incremental token generation
//! - **Voice cloning** support via speaker encoder (ECAPA-TDNN)
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use qwen3_tts::{Qwen3TTS, SynthesisOptions, auto_device};
//!
//! // Load model (downloads from HuggingFace on first use)
//! let device = auto_device()?;
//! let model = Qwen3TTS::from_pretrained("path/to/model", device)?;
//!
//! // Synthesize speech with default settings
//! let audio = model.synthesize("Hello, world!", None)?;
//! audio.save("output.wav")?;
//!
//! // Or with custom options
//! let options = SynthesisOptions {
//!     temperature: 0.8,
//!     top_k: 30,
//!     ..Default::default()
//! };
//! let audio = model.synthesize("Custom settings!", Some(options))?;
//! ```
//!
//! ## Architecture
//!
//! The TTS pipeline consists of three stages:
//!
//! 1. **TalkerModel**: Transformer that generates semantic tokens from text
//!    autoregressively. Based on Qwen2 architecture with dual embeddings
//!    (text + codec) and RoPE position encoding.
//!
//! 2. **CodePredictor**: For each semantic token, generates 15 acoustic
//!    tokens using a smaller autoregressive decoder. Uses residual VQ
//!    pattern where embeddings are summed across codebooks.
//!
//! 3. **Decoder12Hz**: Converts the 16-codebook codec tokens to audio
//!    waveform at 24kHz. Uses ConvNeXt blocks and transposed convolutions
//!    for upsampling.
//!
//! ## Model Variants
//!
//! - **Base**: General English/Chinese TTS
//! - **CustomVoice**: Voice cloning with speaker encoder
//!
//! ## Sample Rate
//!
//! Output audio is always 24kHz mono. Use [`audio::resample()`] if you need
//! a different sample rate.

pub mod audio;
pub mod generation;
#[cfg(feature = "hub")]
pub mod hub;
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
#[cfg(feature = "hub")]
pub use hub::ModelPaths;
pub use models::config::Qwen3TTSConfig;
// StreamingSession is defined in this module, exported as top-level type
pub use models::talker::{codec_tokens, special_tokens, tts_tokens, Language, Speaker};
pub use models::{CodePredictor, CodePredictorConfig, TalkerConfig, TalkerModel as Talker};

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

    /// Load from downloaded model paths.
    ///
    /// Use with [`ModelPaths::download`] for automatic model downloading.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use qwen3_tts::{Qwen3TTS, ModelPaths, auto_device};
    ///
    /// let paths = ModelPaths::download(None)?;
    /// let device = auto_device()?;
    /// let model = Qwen3TTS::from_paths(&paths, device)?;
    /// ```
    #[cfg(feature = "hub")]
    pub fn from_paths(paths: &hub::ModelPaths, device: Device) -> Result<Self> {
        tracing::info!("Loading Qwen3-TTS from downloaded paths...");

        // Load text tokenizer
        let text_tokenizer = tokenizer::TextTokenizer::from_file(&paths.tokenizer)?;

        // Load model weights
        let weights = Self::load_weights(&paths.model_weights, &device)?;

        // Create TalkerModel
        let talker = TalkerModel::from_weights(&weights, &device)?;

        // Create CodePredictor
        let cp_config = CodePredictorConfig::default();
        let cp_weights = Self::filter_weights(&weights, "talker.code_predictor.");
        let cp_vb = candle_nn::VarBuilder::from_tensors(cp_weights, DType::F32, &device);
        let code_predictor = CodePredictor::new(cp_config, cp_vb)?;

        // Load decoder weights
        let st_weights = Self::load_weights(&paths.decoder_weights, &device)?;
        let decoder = Decoder12Hz::from_weights(&st_weights, Default::default())?;

        Ok(Self {
            talker,
            code_predictor,
            decoder,
            text_tokenizer,
            device,
        })
    }

    /// Synthesize speech from text with default voice (Ryan, English).
    ///
    /// Convenience wrapper around [`synthesize_with_voice`](Self::synthesize_with_voice).
    pub fn synthesize(&self, text: &str, options: Option<SynthesisOptions>) -> Result<AudioBuffer> {
        self.synthesize_with_voice(text, Speaker::Ryan, Language::English, options)
    }

    /// Synthesize speech with a specific voice and language.
    ///
    /// Uses the correct generation loop: CustomVoice prefill, autoregressive
    /// semantic tokens, per-frame acoustic code prediction via CodePredictor,
    /// residual VQ summation, and trailing text fusion.
    ///
    /// # Arguments
    ///
    /// * `text` - Text to synthesize
    /// * `speaker` - Predefined speaker voice
    /// * `language` - Target language
    /// * `options` - Synthesis options (temperature, top_k, etc.)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use qwen3_tts::{Qwen3TTS, Speaker, Language, SynthesisOptions};
    ///
    /// let audio = model.synthesize_with_voice(
    ///     "Hello, world!",
    ///     Speaker::Ryan,
    ///     Language::English,
    ///     None,
    /// )?;
    /// audio.save("output.wav")?;
    /// ```
    pub fn synthesize_with_voice(
        &self,
        text: &str,
        speaker: Speaker,
        language: Language,
        options: Option<SynthesisOptions>,
    ) -> Result<AudioBuffer> {
        let options = options.unwrap_or_default();
        let input_ids = self.text_tokenizer.encode(text)?;

        let gen_config = generation::GenerationConfig {
            max_new_tokens: options.max_length,
            temperature: options.temperature,
            top_k: options.top_k,
            top_p: options.top_p,
            repetition_penalty: options.repetition_penalty,
            eos_token_id: options.eos_token_id,
        };

        // Build trailing text: remaining text tokens projected + tts_eos.
        // After trailing text is exhausted, tts_pad is used for each subsequent step.
        let trailing_text_hidden = if input_ids.len() > 1 {
            let remaining_proj = self.talker.get_projected_text_embeddings(&input_ids[1..])?;
            let tts_eos_embed = self.talker.get_tts_eos_embed()?;
            Tensor::cat(&[&remaining_proj, &tts_eos_embed], 1)?
        } else {
            self.talker.get_tts_eos_embed()?
        };
        let trailing_text_len = trailing_text_hidden.dim(1)?;
        let tts_pad_embed = self.talker.get_tts_pad_embed()?;

        // Prefill with CustomVoice format
        let mut kv_caches = self.talker.new_kv_caches();
        let (hidden, logits) =
            self.talker
                .prefill_custom_voice(&input_ids, speaker, language, &mut kv_caches)?;
        let prefill_len = hidden.dim(1)?;
        let mut offset = prefill_len;
        let mut last_hidden = hidden.i((.., prefill_len - 1..prefill_len, ..))?;

        // Sample first semantic token (with token suppression)
        let logits_suppressed =
            generation::apply_token_suppression(&logits.squeeze(1)?, 3072, 151670)?;
        let first_token = generation::sample(&logits_suppressed, &gen_config)?;
        let mut semantic_token: u32 = first_token.flatten_all()?.to_vec1::<u32>()?[0];

        let mut all_codes: Vec<Vec<u32>> = Vec::new();

        // Generation loop: semantic token → acoustic codes → residual VQ sum → trailing text → next step
        for frame_idx in 0..gen_config.max_new_tokens {
            // Check EOS
            if let Some(eos_id) = gen_config.eos_token_id {
                if semantic_token == eos_id {
                    break;
                }
            }

            let semantic_embed = self.talker.get_codec_embedding(semantic_token)?;

            // Generate 15 acoustic codes autoregressively
            let acoustic_codes = self
                .code_predictor
                .generate_acoustic_codes(&last_hidden, &semantic_embed)?;

            // Save frame [semantic, acoustic_0..14]
            let mut frame_codes = vec![semantic_token];
            frame_codes.extend(&acoustic_codes);
            all_codes.push(frame_codes);

            // Build input for next talker step:
            // sum all 16 code embeddings (residual VQ pattern)
            let acoustic_embed_sum = self
                .code_predictor
                .get_acoustic_embeddings_sum(&acoustic_codes, &self.device)?;
            let summed = semantic_embed.add(&acoustic_embed_sum)?;

            // Add trailing text embedding (or tts_pad if exhausted)
            let text_addition = if frame_idx < trailing_text_len {
                trailing_text_hidden.i((.., frame_idx..frame_idx + 1, ..))?
            } else {
                tts_pad_embed.clone()
            };
            let step_input = summed.add(&text_addition)?;

            // Run through talker to get next hidden state and logits
            let (h, new_logits) =
                self.talker
                    .generate_step_with_embed(&step_input, &mut kv_caches, offset)?;
            offset += 1;
            last_hidden = h;

            // Sample next semantic token
            let logits_suppressed =
                generation::apply_token_suppression(&new_logits.squeeze(1)?, 3072, 151670)?;
            let next_token = generation::sample(&logits_suppressed, &gen_config)?;
            semantic_token = next_token.flatten_all()?.to_vec1::<u32>()?[0];
        }

        // Decode to audio
        let codes = self.codes_to_tensor(&all_codes)?;
        let waveform = self.decoder.decode(&codes)?;
        AudioBuffer::from_tensor(waveform, 24000)
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

    /// Create a streaming synthesis session with a specific voice and language.
    ///
    /// Returns an iterator that yields audio chunks as they are generated.
    /// Each chunk contains approximately `chunk_frames` frames worth of audio
    /// (default: 10 frames = ~800ms at 12.5 Hz frame rate).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use qwen3_tts::{Qwen3TTS, Speaker, Language, SynthesisOptions};
    ///
    /// let options = SynthesisOptions::default();
    /// for chunk in model.synthesize_streaming("Hello!", Speaker::Ryan, Language::English, options)? {
    ///     let audio = chunk?;
    ///     // Play or process audio chunk (each ~800ms)
    /// }
    /// ```
    pub fn synthesize_streaming(
        &self,
        text: &str,
        speaker: Speaker,
        language: Language,
        options: SynthesisOptions,
    ) -> Result<StreamingSession<'_>> {
        let input_ids = self.text_tokenizer.encode(text)?;
        StreamingSession::new(self, &input_ids, speaker, language, options)
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

/// Samples per frame at 24kHz with 12.5 Hz frame rate
pub const SAMPLES_PER_FRAME: usize = 1920;

/// Streaming synthesis session.
///
/// Yields audio chunks as they are generated. Use with
/// [`Qwen3TTS::synthesize_streaming`].
pub struct StreamingSession<'a> {
    model: &'a Qwen3TTS,
    config: generation::GenerationConfig,
    kv_caches: Vec<KVCache>,
    offset: usize,
    last_hidden: Tensor,
    current_token: Option<u32>,
    frames_generated: usize,
    frame_buffer: Vec<Vec<u32>>,
    chunk_frames: usize,
    done: bool,
    // Trailing text state for residual VQ + text fusion
    trailing_text_hidden: Tensor,
    trailing_text_len: usize,
    tts_pad_embed: Tensor,
}

impl<'a> StreamingSession<'a> {
    fn new(
        model: &'a Qwen3TTS,
        input_ids: &[u32],
        speaker: Speaker,
        language: Language,
        options: SynthesisOptions,
    ) -> Result<Self> {
        let config = generation::GenerationConfig {
            max_new_tokens: options.max_length,
            temperature: options.temperature,
            top_k: options.top_k,
            top_p: options.top_p,
            repetition_penalty: options.repetition_penalty,
            eos_token_id: options.eos_token_id,
        };

        // Build trailing text embeddings
        let trailing_text_hidden = if input_ids.len() > 1 {
            let remaining_proj = model
                .talker
                .get_projected_text_embeddings(&input_ids[1..])?;
            let tts_eos_embed = model.talker.get_tts_eos_embed()?;
            Tensor::cat(&[&remaining_proj, &tts_eos_embed], 1)?
        } else {
            model.talker.get_tts_eos_embed()?
        };
        let trailing_text_len = trailing_text_hidden.dim(1)?;
        let tts_pad_embed = model.talker.get_tts_pad_embed()?;

        // Prefill with CustomVoice format
        let mut kv_caches = model.talker.new_kv_caches();
        let (hidden, logits) =
            model
                .talker
                .prefill_custom_voice(input_ids, speaker, language, &mut kv_caches)?;
        let prefill_len = hidden.dim(1)?;
        let last_hidden = hidden.i((.., prefill_len - 1..prefill_len, ..))?;

        // Sample first token with token suppression
        let logits_suppressed =
            generation::apply_token_suppression(&logits.squeeze(1)?, 3072, 151670)?;
        let first_token = generation::sample(&logits_suppressed, &config)?;
        let first_token_id: u32 = first_token.flatten_all()?.to_vec1::<u32>()?[0];

        let done = config.eos_token_id == Some(first_token_id);

        Ok(Self {
            model,
            config,
            kv_caches,
            offset: prefill_len,
            last_hidden,
            current_token: if done { None } else { Some(first_token_id) },
            frames_generated: 0,
            frame_buffer: Vec::new(),
            chunk_frames: options.chunk_frames,
            done,
            trailing_text_hidden,
            trailing_text_len,
            tts_pad_embed,
        })
    }

    /// Generate the next chunk of audio.
    ///
    /// Returns `Some(AudioBuffer)` for each chunk, or `None` when generation is complete.
    pub fn next_chunk(&mut self) -> Result<Option<AudioBuffer>> {
        if self.done {
            // Flush remaining buffer
            if !self.frame_buffer.is_empty() {
                let codes = self.model.codes_to_tensor(&self.frame_buffer)?;
                self.frame_buffer.clear();
                let audio = self.model.decoder.decode(&codes)?;
                return Ok(Some(AudioBuffer::from_tensor(audio, 24000)?));
            }
            return Ok(None);
        }

        // Generate frames until we have enough for a chunk
        while self.frame_buffer.len() < self.chunk_frames
            && self.frames_generated < self.config.max_new_tokens
        {
            let token_id = match self.current_token {
                Some(id) => id,
                None => {
                    self.done = true;
                    break;
                }
            };

            let semantic_embed = self.model.talker.get_codec_embedding(token_id)?;

            // Generate 15 acoustic codes
            let acoustic_codes = self
                .model
                .code_predictor
                .generate_acoustic_codes(&self.last_hidden, &semantic_embed)?;

            let mut frame_codes = vec![token_id];
            frame_codes.extend(&acoustic_codes);
            self.frame_buffer.push(frame_codes);

            let frame_idx = self.frames_generated;
            self.frames_generated += 1;

            // Build residual VQ sum + trailing text for next step
            let acoustic_embed_sum = self
                .model
                .code_predictor
                .get_acoustic_embeddings_sum(&acoustic_codes, &self.model.device)?;
            let summed = semantic_embed.add(&acoustic_embed_sum)?;

            let text_addition = if frame_idx < self.trailing_text_len {
                self.trailing_text_hidden
                    .i((.., frame_idx..frame_idx + 1, ..))?
            } else {
                self.tts_pad_embed.clone()
            };
            let step_input = summed.add(&text_addition)?;

            // Run talker step with fused embedding
            let (h, new_logits) = self.model.talker.generate_step_with_embed(
                &step_input,
                &mut self.kv_caches,
                self.offset,
            )?;
            self.offset += 1;
            self.last_hidden = h;

            // Sample next semantic token
            let logits_suppressed =
                generation::apply_token_suppression(&new_logits.squeeze(1)?, 3072, 151670)?;
            let next_token = generation::sample(&logits_suppressed, &self.config)?;
            let next_token_id: u32 = next_token.flatten_all()?.to_vec1::<u32>()?[0];

            if self.config.eos_token_id == Some(next_token_id) {
                self.current_token = None;
                self.done = true;
            } else {
                self.current_token = Some(next_token_id);
            }
        }

        // Decode the buffered frames
        if self.frame_buffer.is_empty() {
            return Ok(None);
        }

        let codes = self.model.codes_to_tensor(&self.frame_buffer)?;
        self.frame_buffer.clear();
        let audio = self.model.decoder.decode(&codes)?;
        Ok(Some(AudioBuffer::from_tensor(audio, 24000)?))
    }

    /// Returns the total number of frames generated so far.
    pub fn frames_generated(&self) -> usize {
        self.frames_generated
    }

    /// Returns true if generation is complete.
    pub fn is_done(&self) -> bool {
        self.done && self.frame_buffer.is_empty()
    }
}

impl<'a> Iterator for StreamingSession<'a> {
    type Item = Result<AudioBuffer>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_chunk() {
            Ok(Some(audio)) => Some(Ok(audio)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

/// Options for speech synthesis
#[derive(Debug, Clone)]
pub struct SynthesisOptions {
    /// Maximum number of frames to generate
    pub max_length: usize,
    /// Sampling temperature (higher = more random)
    pub temperature: f64,
    /// Top-k sampling
    pub top_k: usize,
    /// Top-p (nucleus) sampling
    pub top_p: f64,
    /// Repetition penalty
    pub repetition_penalty: f64,
    /// End-of-sequence token ID (defaults to audio_end token 151670)
    pub eos_token_id: Option<u32>,
    /// Frames per streaming chunk (default: 10 = ~800ms)
    pub chunk_frames: usize,
}

impl Default for SynthesisOptions {
    fn default() -> Self {
        Self {
            max_length: 2048,
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.0,
            eos_token_id: Some(AUDIO_EOS_TOKEN_ID),
            chunk_frames: 10, // ~800ms per chunk at 12.5 Hz
        }
    }
}

/// Select the best available compute device for inference.
///
/// Checks for available hardware in order: CUDA → Metal → CPU.
/// Falls back to CPU if no GPU acceleration is available.
///
/// # Feature Flags
///
/// - `cuda`: Enables NVIDIA GPU support
/// - `metal`: Enables Apple Silicon GPU support
///
/// # Example
///
/// ```rust,ignore
/// let device = qwen3_tts::auto_device()?;
/// let model = Qwen3TTS::from_pretrained("path/to/model", device)?;
/// ```
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
        assert_eq!(options.eos_token_id, Some(AUDIO_EOS_TOKEN_ID));
        assert_eq!(options.chunk_frames, 10);
    }

    #[test]
    fn test_synthesis_options_custom() {
        let options = SynthesisOptions {
            max_length: 512,
            temperature: 0.5,
            top_k: 10,
            top_p: 0.8,
            repetition_penalty: 1.2,
            eos_token_id: Some(AUDIO_EOS_TOKEN_ID),
            chunk_frames: 5,
        };
        assert_eq!(options.max_length, 512);
        assert!((options.temperature - 0.5).abs() < 1e-6);
        assert_eq!(options.eos_token_id, Some(151670));
        assert_eq!(options.chunk_frames, 5);
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
