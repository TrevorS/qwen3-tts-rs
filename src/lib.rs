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
//! - **Voice cloning** via ECAPA-TDNN speaker encoder (Base models)
//! - **Auto-detection** of model variant from `config.json`
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use qwen3_tts::{Qwen3TTS, SynthesisOptions, auto_device};
//!
//! // Load model — variant auto-detected from config.json
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
//!    autoregressively. Uses dual embeddings (text + codec) with MRoPE
//!    (multimodal rotary position encoding) across all variants.
//!
//! 2. **CodePredictor**: For each semantic token, generates 15 acoustic
//!    tokens using a 5-layer autoregressive decoder. The code predictor
//!    always has `hidden_size=1024` regardless of the talker size; 1.7B
//!    models use a `small_to_mtp_projection` layer to bridge the gap.
//!
//! 3. **Decoder12Hz**: Converts the 16-codebook codec tokens to audio
//!    waveform at 24kHz. Uses ConvNeXt blocks and transposed convolutions
//!    for upsampling. Shared across all model variants.
//!
//! ## Model Variants
//!
//! Five official variants exist in two size classes:
//!
//! | Variant | Size | Talker hidden | Speaker conditioning | HuggingFace ID |
//! |---------|------|---------------|---------------------|----------------|
//! | 0.6B Base | 1.8 GB | 1024 | Voice cloning (ECAPA-TDNN) | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` |
//! | 0.6B CustomVoice | 1.8 GB | 1024 | 9 preset speakers | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` |
//! | 1.7B Base | 3.9 GB | 2048 | Voice cloning (ECAPA-TDNN) | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` |
//! | 1.7B CustomVoice | 3.9 GB | 2048 | 9 preset speakers | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` |
//! | 1.7B VoiceDesign | 3.8 GB | 2048 | Text-described voices | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` |
//!
//! **Base**: Includes a speaker encoder for voice cloning from reference audio.
//! Supports x_vector_only (speaker embedding) and ICL (in-context learning
//! with reference audio + text) modes.
//!
//! **CustomVoice**: 9 preset speakers (Serena, Vivian, Ryan, Aiden, etc.) with
//! no speaker encoder. Uses discrete speaker token IDs for voice selection.
//!
//! **VoiceDesign**: Creates novel voices from text descriptions (e.g.,
//! "a deep male voice"). No speaker encoder or preset speakers.
//!
//! All variants share the same speech tokenizer and decoder weights. The
//! code predictor architecture is identical (1024 hidden, 5 layers, 16 heads)
//! across all variants.
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

use models::codec::{Decoder12Hz, Encoder12Hz};
use models::speaker::SpeakerEncoder;
use models::talker::TalkerModel;
use models::KVCache;

/// Re-exports for convenience
pub use audio::AudioBuffer;
#[cfg(feature = "hub")]
pub use hub::ModelPaths;
pub use models::config::Qwen3TTSConfig;
// StreamingSession is defined in this module, exported as top-level type
pub use models::talker::{codec_tokens, special_tokens, tts_tokens, Language, Speaker};
pub use models::{
    CodePredictor, CodePredictorConfig, ModelType, ParsedModelConfig, SpeakerEncoderConfig,
    TalkerConfig, TalkerModel as Talker,
};

/// Reference audio prompt for voice cloning.
///
/// Holds the speaker embedding and optional ICL (in-context learning) data.
/// Created via [`Qwen3TTS::create_voice_clone_prompt`].
pub struct VoiceClonePrompt {
    /// Speaker embedding from the ECAPA-TDNN encoder, shape `[enc_dim]` (typically 1024).
    pub speaker_embedding: Tensor,
    /// Reference audio codec codes for ICL mode, shape `[T, 16]`. `None` = x_vector_only mode.
    pub ref_codes: Option<Tensor>,
    /// Tokenized reference text for ICL mode.
    pub ref_text_ids: Option<Vec<u32>>,
}

/// Main TTS interface using proper autoregressive pipeline.
///
/// Supports all 5 Qwen3-TTS model variants. Use [`model_type()`](Self::model_type)
/// to check which variant was loaded and [`supports_voice_cloning()`](Self::supports_voice_cloning)
/// / [`supports_preset_speakers()`](Self::supports_preset_speakers) to check capabilities.
pub struct Qwen3TTS {
    /// Talker model for semantic token generation
    talker: TalkerModel,
    /// Code predictor for acoustic token generation
    code_predictor: CodePredictor,
    /// 12Hz decoder for audio synthesis
    decoder: Decoder12Hz,
    /// Text tokenizer
    text_tokenizer: tokenizer::TextTokenizer,
    /// Speaker encoder for voice cloning (loaded when weights are present)
    speaker_encoder: Option<SpeakerEncoder>,
    /// Speech tokenizer encoder for ICL voice cloning (encodes reference audio → codes)
    speech_encoder: Option<Encoder12Hz>,
    /// Detected model variant (None if loaded without config.json)
    model_type: Option<ModelType>,
    /// Device to run inference on
    device: Device,
}

impl Qwen3TTS {
    /// Load a model from a HuggingFace model ID or local path.
    ///
    /// Auto-detects the model variant (0.6B/1.7B, Base/CustomVoice/VoiceDesign)
    /// from `config.json` if present, falling back to weight inspection.
    pub fn from_pretrained(model_id: &str, device: Device) -> Result<Self> {
        tracing::info!("Loading Qwen3-TTS from: {}", model_id);

        // Try to parse config.json for auto-detection
        let config_path = Path::new(model_id).join("config.json");
        let parsed_config = if config_path.exists() {
            match ParsedModelConfig::from_file(&config_path) {
                Ok(cfg) => {
                    tracing::info!("Detected model variant: {}", cfg.label());
                    Some(cfg)
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to parse config.json, falling back to weight inspection: {}",
                        e
                    );
                    None
                }
            }
        } else {
            None
        };

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

        // Create TalkerModel — prefer config.json, fallback to weight inspection
        let talker = if let Some(ref cfg) = parsed_config {
            let talker_config = TalkerConfig::from_parsed(cfg);
            TalkerModel::from_weights_with_config(&weights, talker_config, &device)?
        } else {
            TalkerModel::from_weights(&weights, &device)?
        };

        // Create CodePredictor — prefer config.json, fallback to defaults
        let cp_config = if let Some(ref cfg) = parsed_config {
            CodePredictorConfig::from_parsed(cfg)
        } else {
            CodePredictorConfig::default()
        };
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

        // Load speaker encoder if weights are present
        let se_config = parsed_config
            .as_ref()
            .and_then(|c| c.speaker_encoder_config.clone());
        let speaker_encoder =
            Self::try_load_speaker_encoder(&weights, se_config.as_ref(), &device)?;

        // Load speech encoder (for ICL voice cloning) from speech tokenizer weights
        let speech_encoder = Self::try_load_speech_encoder(&st_weights, &device)?;

        let model_type = parsed_config.as_ref().map(|c| c.model_type);

        Ok(Self {
            talker,
            code_predictor,
            decoder,
            text_tokenizer,
            speaker_encoder,
            speech_encoder,
            model_type,
            device,
        })
    }

    /// Load from pre-loaded weight tensors.
    ///
    /// Uses weight inspection for auto-detection. For config.json-based
    /// detection, use [`from_pretrained`](Self::from_pretrained) instead.
    pub fn from_weights(
        model_weights: &HashMap<String, Tensor>,
        decoder_weights: &HashMap<String, Tensor>,
        text_tokenizer: tokenizer::TextTokenizer,
        device: &Device,
    ) -> Result<Self> {
        // Create TalkerModel (auto-detects from weight shapes)
        let talker = TalkerModel::from_weights(model_weights, device)?;

        // Create CodePredictor — infer codec_embed_dim from talker hidden_size
        let talker_hidden = talker.config().hidden_size;
        let cp_config = if talker_hidden != 1024 {
            CodePredictorConfig {
                codec_embed_dim: Some(talker_hidden),
                ..CodePredictorConfig::default()
            }
        } else {
            CodePredictorConfig::default()
        };
        let cp_weights = Self::filter_weights(model_weights, "talker.code_predictor.");
        let cp_vb = candle_nn::VarBuilder::from_tensors(cp_weights, DType::F32, device);
        let code_predictor = CodePredictor::new(cp_config, cp_vb)?;

        // Create Decoder12Hz
        let decoder = Decoder12Hz::from_weights(decoder_weights, Default::default())?;

        // Load speaker encoder if weights are present
        let speaker_encoder = Self::try_load_speaker_encoder(model_weights, None, device)?;

        // Load speech encoder from decoder weights (same safetensors file)
        let speech_encoder = Self::try_load_speech_encoder(decoder_weights, device)?;

        Ok(Self {
            talker,
            code_predictor,
            decoder,
            text_tokenizer,
            speaker_encoder,
            speech_encoder,
            model_type: None,
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

        // Create TalkerModel (auto-detects from weight shapes)
        let talker = TalkerModel::from_weights(&weights, &device)?;

        // Create CodePredictor — infer codec_embed_dim from talker hidden_size
        let talker_hidden = talker.config().hidden_size;
        let cp_config = if talker_hidden != 1024 {
            CodePredictorConfig {
                codec_embed_dim: Some(talker_hidden),
                ..CodePredictorConfig::default()
            }
        } else {
            CodePredictorConfig::default()
        };
        let cp_weights = Self::filter_weights(&weights, "talker.code_predictor.");
        let cp_vb = candle_nn::VarBuilder::from_tensors(cp_weights, DType::F32, &device);
        let code_predictor = CodePredictor::new(cp_config, cp_vb)?;

        // Load decoder weights
        let st_weights = Self::load_weights(&paths.decoder_weights, &device)?;
        let decoder = Decoder12Hz::from_weights(&st_weights, Default::default())?;

        // Load speaker encoder if weights are present
        let speaker_encoder = Self::try_load_speaker_encoder(&weights, None, &device)?;

        // Load speech encoder for ICL voice cloning
        let speech_encoder = Self::try_load_speech_encoder(&st_weights, &device)?;

        Ok(Self {
            talker,
            code_predictor,
            decoder,
            text_tokenizer,
            speaker_encoder,
            speech_encoder,
            model_type: None,
            device,
        })
    }

    /// Returns the detected model type, or `None` if loaded without config.json.
    pub fn model_type(&self) -> Option<&ModelType> {
        self.model_type.as_ref()
    }

    /// Whether this model supports voice cloning (Base models with speaker encoder).
    pub fn supports_voice_cloning(&self) -> bool {
        self.speaker_encoder.is_some()
    }

    /// Whether this model supports preset speaker selection (CustomVoice models).
    ///
    /// Returns `true` for CustomVoice, `false` for Base and VoiceDesign.
    /// When `model_type` is unknown (loaded without config.json), returns `true`
    /// as a permissive default.
    pub fn supports_preset_speakers(&self) -> bool {
        match &self.model_type {
            Some(ModelType::CustomVoice) => true,
            Some(ModelType::Base) | Some(ModelType::VoiceDesign) => false,
            None => true, // permissive when unknown
        }
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
        if let Some(ModelType::Base) = &self.model_type {
            tracing::warn!(
                "Using preset speaker {:?} on a Base model. Base models are trained for \
                 voice cloning, not preset speakers — output will have an unpredictable voice. \
                 Use synthesize_voice_clone() with reference audio instead.",
                speaker
            );
        } else if let Some(ModelType::VoiceDesign) = &self.model_type {
            tracing::warn!(
                "Using preset speaker {:?} on a VoiceDesign model. VoiceDesign models \
                 are trained for text-described voice creation, not preset speakers.",
                speaker
            );
        }

        let options = options.unwrap_or_default();
        let input_ids = self.text_tokenizer.encode(text)?;

        let gen_config = generation::GenerationConfig {
            max_new_tokens: options.max_length,
            temperature: options.temperature,
            top_k: options.top_k,
            top_p: options.top_p,
            repetition_penalty: options.repetition_penalty,
            eos_token_id: options.eos_token_id,
            min_new_tokens: options.min_new_tokens,
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

        // Track generated tokens for repetition penalty
        let mut generated_tokens: Vec<u32> = Vec::new();

        // Sample first semantic token
        let logits_2d = logits.squeeze(1)?;
        let logits_2d =
            self.apply_generation_penalties(&logits_2d, &generated_tokens, &gen_config, 0)?;
        let first_token = generation::sample(&logits_2d, &gen_config)?;
        let mut semantic_token: u32 = first_token.flatten_all()?.to_vec1::<u32>()?[0];
        generated_tokens.push(semantic_token);

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

            // Sample next semantic token with repetition penalty + token suppression + min_new_tokens
            let logits_2d = new_logits.squeeze(1)?;
            let logits_2d = self.apply_generation_penalties(
                &logits_2d,
                &generated_tokens,
                &gen_config,
                generated_tokens.len(),
            )?;
            let next_token = generation::sample(&logits_2d, &gen_config)?;
            semantic_token = next_token.flatten_all()?.to_vec1::<u32>()?[0];
            generated_tokens.push(semantic_token);
        }

        // Decode to audio
        let codes = self.codes_to_tensor(&all_codes)?;
        let waveform = self.decoder.decode(&codes)?;
        AudioBuffer::from_tensor(waveform, 24000)
    }

    /// Convert list of frame codes to tensor [batch, 16, num_frames]
    pub fn codes_to_tensor(&self, codes: &[Vec<u32>]) -> Result<Tensor> {
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

    /// Decode raw frame codes to audio.
    ///
    /// Takes a slice of frames (each frame is a `Vec<u32>` of 16 codebook values)
    /// and runs the 12Hz decoder to produce an audio waveform.
    pub fn decode_codes(&self, codes: &[Vec<u32>]) -> Result<AudioBuffer> {
        let tensor = self.codes_to_tensor(codes)?;
        let waveform = self.decoder.decode(&tensor)?;
        AudioBuffer::from_tensor(waveform, 24000)
    }

    /// Synthesize speech using a cloned voice, returning raw codes alongside audio.
    ///
    /// Identical to [`synthesize_voice_clone`](Self::synthesize_voice_clone) but also
    /// returns the raw generated codes (`Vec<Vec<u32>>`) for debugging.
    /// Each inner `Vec<u32>` is one frame: `[semantic, acoustic_0..14]` (16 values).
    pub fn synthesize_voice_clone_debug(
        &self,
        text: &str,
        prompt: &VoiceClonePrompt,
        language: Language,
        options: Option<SynthesisOptions>,
    ) -> Result<(AudioBuffer, Vec<Vec<u32>>)> {
        let options = options.unwrap_or_default();
        let input_ids = self.text_tokenizer.encode(text)?;

        // Determine if ICL mode is active (ref_codes + ref_text present)
        let is_icl = prompt.ref_codes.is_some() && prompt.ref_text_ids.is_some();

        let gen_config = generation::GenerationConfig {
            max_new_tokens: options.max_length,
            temperature: options.temperature,
            top_k: options.top_k,
            top_p: options.top_p,
            repetition_penalty: options.repetition_penalty,
            eos_token_id: options.eos_token_id,
            min_new_tokens: options.min_new_tokens,
        };

        // Voice clone prefill (9 positions for ICL, 10 for x_vector_only)
        let mut kv_caches = self.talker.new_kv_caches();
        let (hidden, logits) = self.talker.prefill_voice_clone(
            &input_ids,
            &prompt.speaker_embedding,
            language,
            is_icl,
            &mut kv_caches,
        )?;
        let prefill_len = hidden.dim(1)?;
        let mut offset = prefill_len;

        // Initialize last_hidden from prefill; updated by ICL block if active.
        let mut last_hidden = hidden.i((.., prefill_len - 1..prefill_len, ..))?;

        // ICL extension (if reference codes + text are provided)
        let (trailing_text_hidden, logits) = if let (Some(ref_codes), Some(ref_text_ids)) =
            (&prompt.ref_codes, &prompt.ref_text_ids)
        {
            let ref_codec_embeds = self.sum_ref_codec_embeddings(ref_codes)?;

            // In ICL mode, all text tokens go into the ICL prompt (Python:
            // text_id=input_id[:, 3:-5] passes ALL target text tokens).
            // In the non-ICL path the first text token is consumed by the prefill,
            // so only the remaining tokens go to trailing_text.
            let (icl_embed, icl_trailing) =
                self.talker
                    .build_icl_prompt(&input_ids, ref_text_ids, &ref_codec_embeds)?;

            let icl_len = icl_embed.dim(1)?;
            if icl_len > 0 {
                let mut mask_data = vec![0.0f32; icl_len * (offset + icl_len)];
                for i in 0..icl_len {
                    for j in (offset + i + 1)..(offset + icl_len) {
                        mask_data[i * (offset + icl_len) + j] = f32::NEG_INFINITY;
                    }
                }
                let mask =
                    Tensor::from_vec(mask_data, (1, 1, icl_len, offset + icl_len), &self.device)?;

                let mut icl_hidden = icl_embed;
                for (i, layer) in self.talker.layers_iter().enumerate() {
                    icl_hidden = layer.forward(
                        &icl_hidden,
                        self.talker.rope(),
                        Some(&mask),
                        Some(&mut kv_caches[i]),
                        offset,
                    )?;
                }
                icl_hidden = self.talker.apply_norm(&icl_hidden)?;
                offset += icl_len;

                let last_icl_hidden = icl_hidden.i((.., icl_len - 1..icl_len, ..))?;
                let new_logits = self.talker.apply_codec_head(&last_icl_hidden)?;

                // Update last_hidden so the code predictor is conditioned on
                // the ICL context, not the stale prefill hidden state.
                last_hidden = last_icl_hidden;

                (icl_trailing, new_logits)
            } else {
                let trailing = self.build_default_trailing_text(&input_ids)?;
                (trailing, logits)
            }
        } else {
            let trailing = self.build_default_trailing_text(&input_ids)?;
            (trailing, logits)
        };

        let trailing_text_len = trailing_text_hidden.dim(1)?;
        let tts_pad_embed = self.talker.get_tts_pad_embed()?;

        // Track generated tokens for repetition penalty
        let mut generated_tokens: Vec<u32> = Vec::new();

        // Sample first semantic token
        let logits_2d = logits.squeeze(1)?;
        let logits_2d =
            self.apply_generation_penalties(&logits_2d, &generated_tokens, &gen_config, 0)?;
        let first_token = generation::sample(&logits_2d, &gen_config)?;
        let mut semantic_token: u32 = first_token.flatten_all()?.to_vec1::<u32>()?[0];
        generated_tokens.push(semantic_token);

        let mut all_codes: Vec<Vec<u32>> = Vec::new();

        // Generation loop
        for frame_idx in 0..gen_config.max_new_tokens {
            if let Some(eos_id) = gen_config.eos_token_id {
                if semantic_token == eos_id {
                    break;
                }
            }

            let semantic_embed = self.talker.get_codec_embedding(semantic_token)?;

            let acoustic_codes = self
                .code_predictor
                .generate_acoustic_codes(&last_hidden, &semantic_embed)?;

            let mut frame_codes = vec![semantic_token];
            frame_codes.extend(&acoustic_codes);
            all_codes.push(frame_codes);

            let acoustic_embed_sum = self
                .code_predictor
                .get_acoustic_embeddings_sum(&acoustic_codes, &self.device)?;
            let summed = semantic_embed.add(&acoustic_embed_sum)?;

            let text_addition = if frame_idx < trailing_text_len {
                trailing_text_hidden.i((.., frame_idx..frame_idx + 1, ..))?
            } else {
                tts_pad_embed.clone()
            };
            let step_input = summed.add(&text_addition)?;

            let (h, new_logits) =
                self.talker
                    .generate_step_with_embed(&step_input, &mut kv_caches, offset)?;
            offset += 1;
            last_hidden = h;

            // Sample next semantic token with repetition penalty + token suppression + min_new_tokens
            let logits_2d = new_logits.squeeze(1)?;
            let logits_2d = self.apply_generation_penalties(
                &logits_2d,
                &generated_tokens,
                &gen_config,
                generated_tokens.len(),
            )?;
            let next_token = generation::sample(&logits_2d, &gen_config)?;
            semantic_token = next_token.flatten_all()?.to_vec1::<u32>()?[0];
            generated_tokens.push(semantic_token);
        }

        // Prepend ref_codes for ICL decoder context (same fix as synthesize_voice_clone)
        let audio = if let Some(ref_codes) = &prompt.ref_codes {
            let ref_frames = self.tensor_to_frame_codes(ref_codes)?;
            let ref_len = ref_frames.len();
            let mut combined = ref_frames;
            combined.extend(all_codes.iter().cloned());
            let total_len = combined.len();

            let codes = self.codes_to_tensor(&combined)?;
            let waveform = self.decoder.decode(&codes)?;
            let mut audio = AudioBuffer::from_tensor(waveform, 24000)?;
            let cut_samples = ref_len * audio.len() / total_len;
            audio.samples = audio.samples[cut_samples..].to_vec();
            audio
        } else {
            let codes = self.codes_to_tensor(&all_codes)?;
            let waveform = self.decoder.decode(&codes)?;
            AudioBuffer::from_tensor(waveform, 24000)?
        };
        Ok((audio, all_codes))
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

    // ── Voice cloning API ─────────────────────────────────────────────────

    /// Create a voice clone prompt from reference audio.
    ///
    /// When `ref_text` is `None`, produces an **x_vector_only** prompt (speaker
    /// embedding only). When `Some`, produces an **ICL** prompt (speaker embedding
    /// + reference audio codes + reference text) — requires a speech encoder.
    ///
    /// # Errors
    ///
    /// Returns an error if the speaker encoder is not loaded.
    pub fn create_voice_clone_prompt(
        &self,
        ref_audio: &AudioBuffer,
        ref_text: Option<&str>,
    ) -> Result<VoiceClonePrompt> {
        let encoder = self.speaker_encoder.as_ref().ok_or_else(|| {
            let hint = match &self.model_type {
                Some(ModelType::CustomVoice) => {
                    " CustomVoice models use preset speakers (synthesize_with_voice), \
                     not voice cloning. Use a Base model for voice cloning."
                }
                Some(ModelType::VoiceDesign) => {
                    " VoiceDesign models use text-described voices, not voice cloning. \
                     Use a Base model for voice cloning."
                }
                _ => {
                    " Ensure model weights contain `speaker_encoder.*` keys \
                     (only Base models include a speaker encoder)."
                }
            };
            anyhow::anyhow!("Speaker encoder not available.{}", hint)
        })?;

        let speaker_embedding = encoder.encode(ref_audio)?; // [enc_dim]

        // ICL data: encode reference audio to codes and tokenize reference text
        let (ref_codes, ref_text_ids) = if let Some(text) = ref_text {
            let speech_enc = self.speech_encoder.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "ICL voice cloning requires a speech encoder, but it was not loaded. \
                     Ensure the speech tokenizer weights contain encoder keys, or use \
                     x_vector_only mode by passing ref_text=None."
                )
            })?;

            let codes = speech_enc.encode(ref_audio)?; // [T_frames, 16]
            let text_ids = self.text_tokenizer.encode(text)?;

            (Some(codes), Some(text_ids))
        } else {
            (None, None)
        };

        Ok(VoiceClonePrompt {
            speaker_embedding,
            ref_codes,
            ref_text_ids,
        })
    }

    /// Synthesize speech using a cloned voice.
    ///
    /// Uses the same generation loop as [`synthesize_with_voice`] but runs the
    /// voice-clone prefill instead of the predefined-speaker prefill.
    ///
    /// When the prompt contains ICL data (ref_codes + ref_text_ids), the model
    /// is conditioned on reference audio/text to better reproduce the speaker's voice.
    pub fn synthesize_voice_clone(
        &self,
        text: &str,
        prompt: &VoiceClonePrompt,
        language: Language,
        options: Option<SynthesisOptions>,
    ) -> Result<AudioBuffer> {
        let options = options.unwrap_or_default();
        let input_ids = self.text_tokenizer.encode(text)?;

        // Determine if ICL mode is active (ref_codes + ref_text present)
        let is_icl = prompt.ref_codes.is_some() && prompt.ref_text_ids.is_some();

        let gen_config = generation::GenerationConfig {
            max_new_tokens: options.max_length,
            temperature: options.temperature,
            top_k: options.top_k,
            top_p: options.top_p,
            repetition_penalty: options.repetition_penalty,
            eos_token_id: options.eos_token_id,
            min_new_tokens: options.min_new_tokens,
        };

        // Voice clone prefill (9 positions for ICL, 10 for x_vector_only)
        let mut kv_caches = self.talker.new_kv_caches();
        let (hidden, logits) = self.talker.prefill_voice_clone(
            &input_ids,
            &prompt.speaker_embedding,
            language,
            is_icl,
            &mut kv_caches,
        )?;
        let prefill_len = hidden.dim(1)?;
        let mut offset = prefill_len;

        // Initialize last_hidden from prefill; updated by ICL block if active.
        // The code predictor uses this to condition acoustic code generation —
        // it must come from the position that produced the first semantic logits.
        let mut last_hidden = hidden.i((.., prefill_len - 1..prefill_len, ..))?;

        // ── ICL extension (if reference codes + text are provided) ──────────
        let (trailing_text_hidden, logits) = if let (Some(ref_codes), Some(ref_text_ids)) =
            (&prompt.ref_codes, &prompt.ref_text_ids)
        {
            // Sum reference codec embeddings across all 16 codebook groups
            let ref_codec_embeds = self.sum_ref_codec_embeddings(ref_codes)?;

            // In ICL mode, all text tokens go into the ICL prompt (Python:
            // text_id=input_id[:, 3:-5] passes ALL target text tokens).
            let (icl_embed, icl_trailing) =
                self.talker
                    .build_icl_prompt(&input_ids, ref_text_ids, &ref_codec_embeds)?;

            // Feed ICL embeddings through the model to extend the KV cache.
            // Process in a single pass (no generation, just context extension).
            let icl_len = icl_embed.dim(1)?;
            if icl_len > 0 {
                // Create causal mask for ICL tokens
                let mut mask_data = vec![0.0f32; icl_len * (offset + icl_len)];
                for i in 0..icl_len {
                    for j in (offset + i + 1)..(offset + icl_len) {
                        mask_data[i * (offset + icl_len) + j] = f32::NEG_INFINITY;
                    }
                }
                let mask =
                    Tensor::from_vec(mask_data, (1, 1, icl_len, offset + icl_len), &self.device)?;

                let mut icl_hidden = icl_embed;
                for (i, layer) in self.talker.layers_iter().enumerate() {
                    icl_hidden = layer.forward(
                        &icl_hidden,
                        self.talker.rope(),
                        Some(&mask),
                        Some(&mut kv_caches[i]),
                        offset,
                    )?;
                }
                icl_hidden = self.talker.apply_norm(&icl_hidden)?;
                offset += icl_len;

                // Get logits from last ICL position (replaces prefill logits)
                let last_icl_hidden = icl_hidden.i((.., icl_len - 1..icl_len, ..))?;
                let new_logits = self.talker.apply_codec_head(&last_icl_hidden)?;

                // Update last_hidden so the code predictor is conditioned on
                // the ICL context, not the stale prefill hidden state.
                last_hidden = last_icl_hidden;

                (icl_trailing, new_logits)
            } else {
                // No ICL content, fall back to default trailing text
                let trailing = self.build_default_trailing_text(&input_ids)?;
                (trailing, logits)
            }
        } else {
            // No ICL data — use default trailing text
            let trailing = self.build_default_trailing_text(&input_ids)?;
            (trailing, logits)
        };

        let trailing_text_len = trailing_text_hidden.dim(1)?;
        let tts_pad_embed = self.talker.get_tts_pad_embed()?;

        // Track generated tokens for repetition penalty
        let mut generated_tokens: Vec<u32> = Vec::new();

        // Sample first semantic token
        let logits_2d = logits.squeeze(1)?;
        let logits_2d =
            self.apply_generation_penalties(&logits_2d, &generated_tokens, &gen_config, 0)?;
        let first_token = generation::sample(&logits_2d, &gen_config)?;
        let mut semantic_token: u32 = first_token.flatten_all()?.to_vec1::<u32>()?[0];
        generated_tokens.push(semantic_token);

        let mut all_codes: Vec<Vec<u32>> = Vec::new();

        // Generation loop
        for frame_idx in 0..gen_config.max_new_tokens {
            if let Some(eos_id) = gen_config.eos_token_id {
                if semantic_token == eos_id {
                    break;
                }
            }

            let semantic_embed = self.talker.get_codec_embedding(semantic_token)?;

            let acoustic_codes = self
                .code_predictor
                .generate_acoustic_codes(&last_hidden, &semantic_embed)?;

            let mut frame_codes = vec![semantic_token];
            frame_codes.extend(&acoustic_codes);
            all_codes.push(frame_codes);

            let acoustic_embed_sum = self
                .code_predictor
                .get_acoustic_embeddings_sum(&acoustic_codes, &self.device)?;
            let summed = semantic_embed.add(&acoustic_embed_sum)?;

            let text_addition = if frame_idx < trailing_text_len {
                trailing_text_hidden.i((.., frame_idx..frame_idx + 1, ..))?
            } else {
                tts_pad_embed.clone()
            };
            let step_input = summed.add(&text_addition)?;

            let (h, new_logits) =
                self.talker
                    .generate_step_with_embed(&step_input, &mut kv_caches, offset)?;
            offset += 1;
            last_hidden = h;

            // Sample next semantic token with repetition penalty + token suppression + min_new_tokens
            let logits_2d = new_logits.squeeze(1)?;
            let logits_2d = self.apply_generation_penalties(
                &logits_2d,
                &generated_tokens,
                &gen_config,
                generated_tokens.len(),
            )?;
            let next_token = generation::sample(&logits_2d, &gen_config)?;
            semantic_token = next_token.flatten_all()?.to_vec1::<u32>()?[0];
            generated_tokens.push(semantic_token);
        }

        // For ICL mode, the decoder needs reference codes as causal context.
        // Prepend ref_codes to generated codes before decoding, then trim
        // the reference audio from the output (matching Python behavior).
        if let Some(ref_codes) = &prompt.ref_codes {
            let ref_frames = self.tensor_to_frame_codes(ref_codes)?;
            let ref_len = ref_frames.len();
            let mut combined = ref_frames;
            combined.extend(all_codes);
            let total_len = combined.len();

            let codes = self.codes_to_tensor(&combined)?;
            let waveform = self.decoder.decode(&codes)?;
            let mut audio = AudioBuffer::from_tensor(waveform, 24000)?;

            // Trim the reference audio portion (proportional, like Python)
            let cut_samples = ref_len * audio.len() / total_len;
            audio.samples = audio.samples[cut_samples..].to_vec();
            Ok(audio)
        } else {
            let codes = self.codes_to_tensor(&all_codes)?;
            let waveform = self.decoder.decode(&codes)?;
            AudioBuffer::from_tensor(waveform, 24000)
        }
    }

    /// Convert a ref_codes tensor `[T_frames, 16]` to `Vec<Vec<u32>>` frame format.
    fn tensor_to_frame_codes(&self, codes: &Tensor) -> Result<Vec<Vec<u32>>> {
        let (n_frames, n_codebooks) = codes.dims2()?;
        let codes_u32 = codes.to_dtype(DType::U32)?;
        let mut frames = Vec::with_capacity(n_frames);
        for f in 0..n_frames {
            let frame_tensor = codes_u32.i(f)?; // [16]
            let frame_vec: Vec<u32> = frame_tensor.to_vec1()?;
            debug_assert_eq!(frame_vec.len(), n_codebooks);
            frames.push(frame_vec);
        }
        Ok(frames)
    }

    /// Sum reference codec embeddings across all 16 codebook groups.
    ///
    /// For each frame:
    /// - Group 0 (semantic): `talker.codec_embedding(ref_codes[:, 0])`
    /// - Groups 1–15 (acoustic): `code_predictor.codec_embeddings[i-1](ref_codes[:, i])`
    /// - Sum all 16 → single embedding per frame
    ///
    /// # Arguments
    /// * `ref_codes` — shape `[T_frames, 16]` of i64 codes
    ///
    /// # Returns
    /// Tensor of shape `[1, T_frames, hidden_size]`
    fn sum_ref_codec_embeddings(&self, ref_codes: &Tensor) -> Result<Tensor> {
        // Group 0: semantic codes → talker.codec_embedding
        let semantic_codes = ref_codes.i((.., 0))?; // [T_frames]
        let semantic_codes = semantic_codes.to_dtype(candle_core::DType::U32)?;
        let summed = self.talker.get_codec_embedding_batch(&semantic_codes)?; // [1, T, hidden]

        // Groups 1-15: acoustic codes → code_predictor.embed_codes_for_group
        let mut summed = summed;
        for group in 1..16 {
            let group_codes = ref_codes.i((.., group))?; // [T_frames]
            let group_codes = group_codes.to_dtype(candle_core::DType::U32)?;
            let group_embed = self
                .code_predictor
                .embed_codes_for_group(group - 1, &group_codes)?; // [1, T, embed_dim]
            summed = summed.add(&group_embed)?;
        }

        Ok(summed)
    }

    /// Build default trailing text embeddings (for non-ICL mode).
    fn build_default_trailing_text(&self, input_ids: &[u32]) -> Result<Tensor> {
        if input_ids.len() > 1 {
            let remaining_proj = self.talker.get_projected_text_embeddings(&input_ids[1..])?;
            let tts_eos_embed = self.talker.get_tts_eos_embed()?;
            Ok(Tensor::cat(&[&remaining_proj, &tts_eos_embed], 1)?)
        } else {
            self.talker.get_tts_eos_embed()
        }
    }

    /// Apply repetition penalty, token suppression, and min_new_tokens EOS suppression.
    ///
    /// This is the standard logit processing pipeline matching Python/HuggingFace order:
    /// 1. Repetition penalty (penalize previously generated tokens)
    /// 2. Token suppression (mask reserved control tokens [2048, 3072) except EOS)
    /// 3. Min new tokens (suppress EOS if fewer than min_new_tokens have been generated)
    fn apply_generation_penalties(
        &self,
        logits: &Tensor,
        generated_tokens: &[u32],
        config: &generation::GenerationConfig,
        token_count: usize,
    ) -> Result<Tensor> {
        // 1. Repetition penalty
        let logits = if config.repetition_penalty != 1.0 && !generated_tokens.is_empty() {
            let prev = Tensor::new(generated_tokens, &self.device)?;
            generation::apply_repetition_penalty(logits, &prev, config.repetition_penalty)?
        } else {
            logits.clone()
        };

        // 2. Token suppression
        let logits = generation::apply_token_suppression(&logits, 3072, CODEC_EOS_TOKEN_ID)?;

        // 3. Min new tokens: suppress EOS if we haven't generated enough tokens yet
        if token_count < config.min_new_tokens {
            if let Some(eos_id) = config.eos_token_id {
                let mut logits_vec: Vec<f32> = logits.flatten_all()?.to_vec1()?;
                let vocab = logits.dim(1)?;
                let batch = logits.dim(0)?;
                for b in 0..batch {
                    logits_vec[b * vocab + eos_id as usize] = f32::NEG_INFINITY;
                }
                return Ok(Tensor::from_vec(
                    logits_vec,
                    logits.shape(),
                    logits.device(),
                )?);
            }
        }

        Ok(logits)
    }

    /// Returns `true` if a speaker encoder is loaded (voice cloning is available).
    pub fn has_speaker_encoder(&self) -> bool {
        self.speaker_encoder.is_some()
    }

    /// Returns `true` if a speech encoder is loaded (ICL voice cloning is available).
    pub fn has_speech_encoder(&self) -> bool {
        self.speech_encoder.is_some()
    }

    // ── Private helpers ─────────────────────────────────────────────────

    /// Attempt to load the speaker encoder from model weights.
    ///
    /// Returns `Ok(Some(encoder))` if `speaker_encoder.*` keys are found,
    /// `Ok(None)` if they are absent. When `config` is provided, uses the
    /// parsed enc_dim; otherwise falls back to defaults (enc_dim=1024).
    fn try_load_speaker_encoder(
        weights: &HashMap<String, Tensor>,
        config: Option<&SpeakerEncoderConfig>,
        device: &Device,
    ) -> Result<Option<SpeakerEncoder>> {
        let has_se_weights = weights.keys().any(|k| k.starts_with("speaker_encoder."));
        if !has_se_weights {
            return Ok(None);
        }

        let config = config.cloned().unwrap_or_default();
        tracing::info!(
            "Loading speaker encoder (ECAPA-TDNN, enc_dim={}) for voice cloning...",
            config.enc_dim
        );
        let se_weights = Self::filter_weights(weights, "speaker_encoder.");
        let se_vb = candle_nn::VarBuilder::from_tensors(se_weights, DType::F32, device);
        let encoder = SpeakerEncoder::new(config, se_vb)?;
        Ok(Some(encoder))
    }

    /// Attempt to load the speech encoder (Mimi) from speech tokenizer weights.
    ///
    /// The speech encoder encodes raw audio to 12Hz codec codes, needed for
    /// ICL voice cloning. Returns `Ok(None)` if encoder keys are absent or
    /// loading fails (non-fatal — ICL mode just won't be available).
    fn try_load_speech_encoder(
        weights: &HashMap<String, Tensor>,
        device: &Device,
    ) -> Result<Option<Encoder12Hz>> {
        // Check for encoder-related keys (either HF or candle format)
        let has_encoder_keys = weights
            .keys()
            .any(|k| k.starts_with("encoder.") || k.starts_with("encoder_transformer."));
        if !has_encoder_keys {
            return Ok(None);
        }

        tracing::debug!("Attempting to load speech encoder (Mimi) for ICL voice cloning...");
        match Encoder12Hz::from_weights(weights, device) {
            Ok(enc) => {
                tracing::info!("Loaded speech encoder — ICL voice cloning available");
                Ok(Some(enc))
            }
            Err(e) => {
                tracing::debug!(
                    "Speech encoder not available ({}). ICL voice cloning disabled.",
                    e
                );
                Ok(None)
            }
        }
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
/// Codec EOS token ID — signals end of speech generation.
///
/// This is in the codec vocabulary [0, 3072), not the text vocabulary.
/// Value comes from model config: `talker_config.codec_eos_token_id`.
pub const CODEC_EOS_TOKEN_ID: u32 = 2150;

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
    // Track generated semantic tokens for repetition penalty
    generated_tokens: Vec<u32>,
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
            min_new_tokens: options.min_new_tokens,
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

        // Sample first token with full penalty pipeline
        let mut generated_tokens: Vec<u32> = Vec::new();
        let logits_2d = logits.squeeze(1)?;
        let logits_2d =
            model.apply_generation_penalties(&logits_2d, &generated_tokens, &config, 0)?;
        let first_token = generation::sample(&logits_2d, &config)?;
        let first_token_id: u32 = first_token.flatten_all()?.to_vec1::<u32>()?[0];
        generated_tokens.push(first_token_id);

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
            generated_tokens,
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

            // Sample next semantic token with repetition penalty + token suppression + min_new_tokens
            let logits_2d = new_logits.squeeze(1)?;
            let logits_2d = self.model.apply_generation_penalties(
                &logits_2d,
                &self.generated_tokens,
                &self.config,
                self.generated_tokens.len(),
            )?;
            let next_token = generation::sample(&logits_2d, &self.config)?;
            let next_token_id: u32 = next_token.flatten_all()?.to_vec1::<u32>()?[0];
            self.generated_tokens.push(next_token_id);

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
    /// Repetition penalty (1.0 = disabled, 1.05 = Python default)
    pub repetition_penalty: f64,
    /// End-of-sequence token ID (defaults to codec EOS token 2150)
    pub eos_token_id: Option<u32>,
    /// Frames per streaming chunk (default: 10 = ~800ms)
    pub chunk_frames: usize,
    /// Minimum tokens before EOS is allowed (default: 2, matching Python)
    pub min_new_tokens: usize,
}

impl Default for SynthesisOptions {
    fn default() -> Self {
        Self {
            max_length: 2048,
            temperature: 0.9,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.05,
            eos_token_id: Some(CODEC_EOS_TOKEN_ID),
            chunk_frames: 10, // ~800ms per chunk at 12.5 Hz
            min_new_tokens: 2,
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
        assert!((options.temperature - 0.9).abs() < 1e-6);
        assert_eq!(options.top_k, 50);
        assert!((options.top_p - 0.9).abs() < 1e-6);
        assert!((options.repetition_penalty - 1.05).abs() < 1e-6);
        assert_eq!(options.eos_token_id, Some(CODEC_EOS_TOKEN_ID));
        assert_eq!(options.chunk_frames, 10);
        assert_eq!(options.min_new_tokens, 2);
    }

    #[test]
    fn test_synthesis_options_custom() {
        let options = SynthesisOptions {
            max_length: 512,
            temperature: 0.5,
            top_k: 10,
            top_p: 0.8,
            repetition_penalty: 1.2,
            eos_token_id: Some(CODEC_EOS_TOKEN_ID),
            chunk_frames: 5,
            min_new_tokens: 0,
        };
        assert_eq!(options.max_length, 512);
        assert!((options.temperature - 0.5).abs() < 1e-6);
        assert_eq!(options.eos_token_id, Some(CODEC_EOS_TOKEN_ID));
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
