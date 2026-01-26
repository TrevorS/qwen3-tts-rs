//! Neural network models for Qwen3-TTS
//!
//! This module contains:
//! - `transformer`: Shared building blocks (KVCache, RoPE, Attention, MLP, DecoderLayer)
//! - `talker`: TalkerModel for semantic token generation
//! - `code_predictor`: Acoustic token predictor
//! - `speaker`: Speaker encoder (ECAPA-TDNN)
//! - `codec`: Audio codec for encoding/decoding
//! - `config`: Model configuration

pub mod code_predictor;
pub mod codec;
pub mod config;
pub mod speaker;
pub mod talker;
pub mod transformer;

pub use code_predictor::{CodePredictor, CodePredictorConfig};
pub use config::Qwen3TTSConfig;
pub use speaker::SpeakerEncoder;
pub use talker::{TalkerConfig, TalkerModel};
pub use transformer::{KVCache, MRoPE, RotaryEmbedding};
