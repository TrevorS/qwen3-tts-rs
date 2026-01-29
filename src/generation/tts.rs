//! TTS-specific generation logic
//!
//! Token suppression during sampling to prevent the model from generating
//! tokens in the reserved control range.

use anyhow::Result;
use candle_core::Tensor;

/// Apply token suppression to logits.
///
/// Masks out tokens in range `[vocab_size - 1024, vocab_size)` except for the
/// EOS token, which is preserved. This prevents the model from generating
/// reserved control tokens.
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_suppression_masks_control_tokens() {
        let device = Device::Cpu;
        let vocab_size = 3072;
        let eos_id = 2150u32;
        // All logits at 1.0
        let logits = Tensor::ones((1, vocab_size), candle_core::DType::F32, &device).unwrap();
        let result = apply_token_suppression(&logits, vocab_size, eos_id).unwrap();
        let vals: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();

        // Non-control tokens should be unchanged
        assert!((vals[0] - 1.0).abs() < 1e-6);
        assert!((vals[2047] - 1.0).abs() < 1e-6);

        // Control tokens (except EOS) should be -inf
        assert!(vals[2048].is_infinite() && vals[2048] < 0.0); // suppress_start = 3072 - 1024 = 2048
        assert!(vals[2149].is_infinite() && vals[2149] < 0.0); // 2149 != 2150

        // EOS should be preserved
        assert!((vals[2150] - 1.0).abs() < 1e-6);

        // Other control tokens suppressed
        assert!(vals[2151].is_infinite() && vals[2151] < 0.0);
        assert!(vals[3071].is_infinite() && vals[3071] < 0.0);
    }

    #[test]
    fn test_suppression_batch() {
        let device = Device::Cpu;
        let vocab_size = 3072;
        let eos_id = 2150u32;
        let logits = Tensor::ones((2, vocab_size), candle_core::DType::F32, &device).unwrap();
        let result = apply_token_suppression(&logits, vocab_size, eos_id).unwrap();
        let vals: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();

        // Both batches should have suppression
        assert!(vals[2048].is_infinite()); // batch 0
        assert!(vals[vocab_size + 2048].is_infinite()); // batch 1

        // EOS preserved in both
        assert!((vals[2150] - 1.0).abs() < 1e-6);
        assert!((vals[vocab_size + 2150] - 1.0).abs() < 1e-6);
    }
}
