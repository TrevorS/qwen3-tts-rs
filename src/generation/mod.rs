//! Generation and sampling utilities for Qwen3-TTS
//!
//! This module provides:
//! - Sampling strategies (greedy, top-k, top-p, temperature)
//! - Generation configuration
//! - Repetition penalty
//! - Seeded RNG for reproducible generation
//! - TTS-specific generation (prefill construction, token suppression)

mod sampling;
pub mod tts;

pub use sampling::{
    apply_repetition_penalty, clear_seed, get_seed, greedy_sample, is_seeded, reset_rng, sample,
    set_seed, GenerationConfig,
};

pub use tts::apply_token_suppression;
