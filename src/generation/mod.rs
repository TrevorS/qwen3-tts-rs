//! Generation and sampling utilities for Qwen3-TTS
//!
//! This module provides:
//! - Sampling strategies (greedy, top-k, top-p, temperature)
//! - Generation configuration
//! - Repetition penalty
//! - Per-session RNG via [`SamplingContext`] for reproducible generation
//! - TTS-specific generation (prefill construction, token suppression)

mod sampling;
pub mod tts;

pub use sampling::{
    apply_repetition_penalty, greedy_sample, sample, GenerationConfig, SamplingContext,
};

pub use tts::apply_token_suppression;
