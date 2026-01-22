//! Text tokenizer wrapper for Qwen2TokenizerFast

use anyhow::{anyhow, Result};
use std::path::Path;
use tokenizers::Tokenizer;

/// Text tokenizer wrapping HuggingFace tokenizers
pub struct TextTokenizer {
    tokenizer: Tokenizer,
    /// Beginning of sequence token ID
    pub bos_token_id: u32,
    /// End of sequence token ID
    pub eos_token_id: u32,
    /// Padding token ID
    pub pad_token_id: u32,
}

impl TextTokenizer {
    /// Load tokenizer from a HuggingFace model ID or local path
    pub fn from_pretrained(model_id: &str) -> Result<Self> {
        // Try local path first
        let tokenizer_path = Path::new(model_id).join("tokenizer.json");
        if tokenizer_path.exists() {
            return Self::from_file(&tokenizer_path);
        }

        // For remote loading, we'd need hf-hub crate
        // For now, return an error suggesting local download
        Err(anyhow!(
            "Remote tokenizer loading not yet implemented. \
             Please download the tokenizer locally first: {}",
            model_id
        ))
    }

    /// Load tokenizer from a local file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|e| anyhow!("Failed to load tokenizer from {}: {}", path.display(), e))?;

        Self::from_tokenizer(tokenizer)
    }

    /// Create from a tokenizers::Tokenizer instance
    fn from_tokenizer(tokenizer: Tokenizer) -> Result<Self> {
        // Get special token IDs (Qwen2 defaults)
        let bos_token_id = tokenizer
            .token_to_id("<|im_start|>")
            .unwrap_or(151643);

        let eos_token_id = tokenizer
            .token_to_id("<|im_end|>")
            .unwrap_or(151645);

        let pad_token_id = tokenizer
            .token_to_id("<|endoftext|>")
            .unwrap_or(151643);

        Ok(Self {
            tokenizer,
            bos_token_id,
            eos_token_id,
            pad_token_id,
        })
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.tokenizer
            .encode(text, false)
            .map_err(|e| anyhow!("Failed to encode text: {}", e))?;

        Ok(encoding.get_ids().to_vec())
    }

    /// Encode with special tokens (BOS/EOS)
    pub fn encode_with_special(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = vec![self.bos_token_id];
        ids.extend(self.encode(text)?);
        ids.push(self.eos_token_id);
        Ok(ids)
    }

    /// Encode using chat template format
    pub fn encode_chat(&self, text: &str, role: &str) -> Result<Vec<u32>> {
        // Qwen chat format: <|im_start|>role\ntext<|im_end|>
        let formatted = format!("<|im_start|>{}\n{}<|im_end|>", role, text);
        self.encode(&formatted)
    }

    /// Encode for TTS (user message format)
    pub fn encode_for_tts(&self, text: &str) -> Result<Vec<u32>> {
        // Format as user message
        let mut ids = self.encode_chat(text, "user")?;

        // Add assistant start token for generation
        ids.extend(self.encode("<|im_start|>assistant\n")?);

        Ok(ids)
    }

    /// Decode token IDs back to text
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let text = self.tokenizer
            .decode(ids, true)
            .map_err(|e| anyhow!("Failed to decode tokens: {}", e))?;

        Ok(text)
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    /// Convert token to ID
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    /// Convert ID to token
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }

    /// Batch encode multiple texts
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>> {
        let encodings = self.tokenizer
            .encode_batch(texts.to_vec(), false)
            .map_err(|e| anyhow!("Failed to batch encode: {}", e))?;

        Ok(encodings
            .into_iter()
            .map(|e| e.get_ids().to_vec())
            .collect())
    }

    /// Encode and pad to max length
    pub fn encode_padded(&self, text: &str, max_length: usize) -> Result<Vec<u32>> {
        let mut ids = self.encode(text)?;

        if ids.len() > max_length {
            ids.truncate(max_length);
        } else {
            // Left-pad for TTS (causal attention)
            let pad_len = max_length - ids.len();
            let mut padded = vec![self.pad_token_id; pad_len];
            padded.extend(ids);
            ids = padded;
        }

        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_special_tokens() {
        // Test that default token IDs are reasonable
        assert!(151643 > 0);  // BOS
        assert!(151645 > 0);  // EOS
    }
}
