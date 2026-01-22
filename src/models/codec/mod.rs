//! Audio codec for Qwen3-TTS
//!
//! This module implements the audio encoder/decoder that converts
//! between raw audio waveforms and discrete codec tokens.
//!
//! Two codec variants are supported:
//! - 12Hz (Mimi-based): Higher quality, newer architecture
//! - 25Hz (BigVGAN-based): Faster, more established

mod decoder;
mod quantizer;

pub use decoder::CodecDecoder;
pub use quantizer::{ResidualVectorQuantizer, VectorQuantizer};

use anyhow::Result;
use candle_core::{Device, Tensor};

use super::config::AudioCodecConfig;

/// Audio codec for encoding/decoding between waveforms and tokens
pub struct AudioCodec {
    config: AudioCodecConfig,
    decoder: CodecDecoder,
    device: Device,
}

impl AudioCodec {
    /// Load codec from pretrained weights
    pub fn from_pretrained(model_id: &str, device: &Device) -> Result<Self> {
        // TODO: Implement weight loading
        anyhow::bail!("Audio codec loading not yet implemented: {}", model_id)
    }

    /// Create codec with given config and weights
    pub fn new(config: AudioCodecConfig, decoder: CodecDecoder, device: Device) -> Self {
        Self {
            config,
            decoder,
            device,
        }
    }

    /// Decode codec tokens to audio waveform
    ///
    /// # Arguments
    /// * `tokens` - Tensor of shape [batch, num_quantizers, seq_len] or [batch, seq_len * num_quantizers]
    ///
    /// # Returns
    /// Audio tensor of shape [batch, samples]
    pub fn decode(&self, tokens: &Tensor) -> Result<Tensor> {
        self.decoder.decode(tokens)
    }

    /// Get the frame rate in Hz
    pub fn frame_rate(&self) -> f32 {
        self.config.frame_rate
    }

    /// Get number of quantizers
    pub fn num_quantizers(&self) -> usize {
        self.config.num_quantizers
    }

    /// Get codebook size
    pub fn codebook_size(&self) -> usize {
        self.config.codebook_size
    }

    /// Calculate audio duration from token count
    pub fn tokens_to_seconds(&self, num_tokens: usize) -> f32 {
        num_tokens as f32 / self.config.frame_rate
    }

    /// Calculate token count from audio duration
    pub fn seconds_to_tokens(&self, seconds: f32) -> usize {
        (seconds * self.config.frame_rate).ceil() as usize
    }
}

/// Predefined codec configurations
pub mod presets {
    use super::AudioCodecConfig;

    /// 12Hz codec configuration (Mimi-based)
    pub fn codec_12hz() -> AudioCodecConfig {
        AudioCodecConfig {
            codec_type: "12hz".to_string(),
            sample_rate: 24000,
            num_quantizers: 16,
            codebook_size: 2048,
            frame_rate: 12.5,
            decoder_hidden_size: 1024,
            decoder_num_layers: 8,
        }
    }

    /// 25Hz codec configuration (BigVGAN-based)
    pub fn codec_25hz() -> AudioCodecConfig {
        AudioCodecConfig {
            codec_type: "25hz".to_string(),
            sample_rate: 24000,
            num_quantizers: 16,
            codebook_size: 2048,
            frame_rate: 25.0,
            decoder_hidden_size: 512,
            decoder_num_layers: 6,
        }
    }
}
