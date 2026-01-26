//! TTS-specific generation logic
//!
//! Token suppression during sampling to prevent the model from generating
//! tokens in the reserved control range.

use anyhow::Result;
use candle_core::Tensor;

/// Apply token suppression to logits
///
/// Masks out tokens in range [vocab_size - 1024, vocab_size) except for EOS token
pub fn apply_token_suppression(
    logits: &Tensor,
    vocab_size: usize,
    eos_token_id: u32,
) -> Result<Tensor> {
    let suppress_start = vocab_size - 1024;
    let logits_vec: Vec<f32> = logits.flatten_all()?.to_vec1()?;
    let batch_size = logits.dim(0)?;
    let vocab = logits.dim(1)?;

    let mut suppressed = logits_vec;
    for b in 0..batch_size {
        for v in suppress_start..vocab_size {
            if v as u32 != eos_token_id {
                suppressed[b * vocab + v] = f32::NEG_INFINITY;
            }
        }
    }

    Ok(Tensor::from_vec(
        suppressed,
        logits.shape(),
        logits.device(),
    )?)
}
