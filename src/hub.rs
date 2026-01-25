//! HuggingFace Hub integration for downloading Qwen3-TTS models.
//!
//! This module provides utilities for downloading model weights from HuggingFace Hub.
//! Enable with the `hub` feature.
//!
//! # Example
//!
//! ```rust,ignore
//! use qwen3_tts::hub::ModelPaths;
//!
//! // Download default model
//! let paths = ModelPaths::download(None)?;
//!
//! // Or download a specific variant
//! let paths = ModelPaths::download(Some("Qwen/Qwen3-TTS-12Hz-0.6B-Base"))?;
//!
//! // Use the downloaded paths
//! let model = Qwen3TTS::from_paths(&paths, device)?;
//! ```

use anyhow::{Context, Result};
use hf_hub::api::sync::Api;
use std::path::PathBuf;

/// Default HuggingFace model IDs for Qwen3-TTS components.
pub mod model_ids {
    /// Main TalkerModel (0.6B parameters)
    pub const TALKER: &str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base";

    /// Speech tokenizer (12Hz decoder)
    pub const SPEECH_TOKENIZER: &str = "Qwen/Qwen3-TTS-Tokenizer-12Hz";

    /// Text tokenizer (Qwen2 vocabulary)
    pub const TEXT_TOKENIZER: &str = "Qwen/Qwen2-0.5B";
}

/// Paths to downloaded model files.
#[derive(Debug, Clone)]
pub struct ModelPaths {
    /// Path to main model weights (model.safetensors)
    pub model_weights: PathBuf,
    /// Path to speech tokenizer weights
    pub decoder_weights: PathBuf,
    /// Path to tokenizer.json file
    pub tokenizer: PathBuf,
    /// Path to model config.json
    pub config: PathBuf,
}

impl ModelPaths {
    /// Download all model components from HuggingFace Hub.
    ///
    /// Uses default model IDs if none specified. Downloads to HuggingFace cache.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Optional custom model ID for the main talker model
    pub fn download(model_id: Option<&str>) -> Result<Self> {
        let api = Api::new().context("Failed to create HuggingFace API")?;

        let talker_id = model_id.unwrap_or(model_ids::TALKER);

        tracing::info!("Downloading Qwen3-TTS model files...");

        // Download main model
        tracing::info!("  Downloading talker model: {}", talker_id);
        let talker_repo = api.model(talker_id.to_string());
        let model_weights = talker_repo
            .get("model.safetensors")
            .context("Failed to download model.safetensors")?;
        let config = talker_repo
            .get("config.json")
            .context("Failed to download config.json")?;

        // Download speech tokenizer (decoder)
        tracing::info!(
            "  Downloading speech tokenizer: {}",
            model_ids::SPEECH_TOKENIZER
        );
        let st_repo = api.model(model_ids::SPEECH_TOKENIZER.to_string());
        let decoder_weights = st_repo
            .get("model.safetensors")
            .context("Failed to download speech tokenizer")?;

        // Download text tokenizer
        tracing::info!(
            "  Downloading text tokenizer: {}",
            model_ids::TEXT_TOKENIZER
        );
        let tok_repo = api.model(model_ids::TEXT_TOKENIZER.to_string());
        let tokenizer = tok_repo
            .get("tokenizer.json")
            .context("Failed to download tokenizer.json")?;

        tracing::info!("Download complete!");

        Ok(Self {
            model_weights,
            decoder_weights,
            tokenizer,
            config,
        })
    }

    /// Download with a specific revision/branch.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Model ID on HuggingFace Hub
    /// * `revision` - Git revision (branch, tag, or commit hash)
    pub fn download_revision(model_id: &str, revision: &str) -> Result<Self> {
        let api = Api::new().context("Failed to create HuggingFace API")?;

        tracing::info!("Downloading {} @ {}", model_id, revision);

        let talker_repo = api.repo(hf_hub::Repo::with_revision(
            model_id.to_string(),
            hf_hub::RepoType::Model,
            revision.to_string(),
        ));

        let model_weights = talker_repo
            .get("model.safetensors")
            .context("Failed to download model.safetensors")?;
        let config = talker_repo
            .get("config.json")
            .context("Failed to download config.json")?;

        // Speech tokenizer and text tokenizer use main branch
        let st_repo = api.model(model_ids::SPEECH_TOKENIZER.to_string());
        let decoder_weights = st_repo
            .get("model.safetensors")
            .context("Failed to download speech tokenizer")?;

        let tok_repo = api.model(model_ids::TEXT_TOKENIZER.to_string());
        let tokenizer = tok_repo
            .get("tokenizer.json")
            .context("Failed to download tokenizer.json")?;

        Ok(Self {
            model_weights,
            decoder_weights,
            tokenizer,
            config,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_ids() {
        assert!(model_ids::TALKER.starts_with("Qwen/"));
        assert!(model_ids::SPEECH_TOKENIZER.starts_with("Qwen/"));
        assert!(model_ids::TEXT_TOKENIZER.starts_with("Qwen/"));
    }
}
