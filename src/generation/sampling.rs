//! Token sampling strategies for autoregressive generation

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor, D};

/// Configuration for autoregressive generation
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum number of new tokens to generate
    pub max_new_tokens: usize,
    /// Sampling temperature (1.0 = no change, <1.0 = more focused, >1.0 = more random)
    pub temperature: f64,
    /// Top-k sampling (0 = disabled)
    pub top_k: usize,
    /// Top-p (nucleus) sampling threshold
    pub top_p: f64,
    /// Repetition penalty (1.0 = no penalty)
    pub repetition_penalty: f64,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 2048,
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.0,
        }
    }
}

/// Sample next token from logits
///
/// # Arguments
/// * `logits` - Logits tensor of shape [batch, vocab_size]
/// * `config` - Generation configuration
///
/// # Returns
/// Token indices of shape [batch]
pub fn sample(logits: &Tensor, config: &GenerationConfig) -> Result<Tensor> {
    let logits = logits.to_dtype(DType::F32)?;

    // Apply temperature
    let logits = if config.temperature != 1.0 && config.temperature > 0.0 {
        (logits / config.temperature)?
    } else {
        logits
    };

    // For simplicity, if temperature is very low, use greedy
    if config.temperature < 0.01 {
        return greedy_sample(&logits);
    }

    // Convert to probabilities
    let probs = candle_nn::ops::softmax_last_dim(&logits)?;

    // Sample from distribution
    multinomial_sample(&probs)
}

/// Compute cumulative sum along last dimension
fn cumulative_sum(x: &Tensor) -> Result<Tensor> {
    let (batch, len) = x.dims2()?;
    let mut results = Vec::with_capacity(batch);

    for b in 0..batch {
        let row: Vec<f32> = x.i(b)?.to_vec1()?;
        let mut cumsum = Vec::with_capacity(len);
        let mut sum = 0.0f32;
        for v in row {
            sum += v;
            cumsum.push(sum);
        }
        results.push(cumsum);
    }

    let flat: Vec<f32> = results.into_iter().flatten().collect();
    Ok(Tensor::new(flat.as_slice(), x.device())?.reshape((batch, len))?)
}

/// Sample from probability distribution using multinomial sampling
fn multinomial_sample(probs: &Tensor) -> Result<Tensor> {
    let (batch, vocab) = probs.dims2()?;

    // Use cumulative distribution for sampling
    let cumsum = cumulative_sum(probs)?;

    // Generate uniform random values
    let uniform: Vec<f32> = (0..batch)
        .map(|_| rand_f32())
        .collect();
    let uniform = Tensor::new(uniform.as_slice(), probs.device())?
        .unsqueeze(1)?;

    // Find first index where cumsum >= uniform
    let mask = cumsum.ge(&uniform.broadcast_as(cumsum.shape())?)?;

    // Convert mask to f32 for operations
    let mask_f32 = mask.to_dtype(DType::F32)?;

    // Use a trick: multiply by position and find first nonzero
    let positions: Vec<f32> = (0..vocab).map(|i| i as f32 + 1.0).collect();
    let positions = Tensor::new(positions.as_slice(), probs.device())?
        .unsqueeze(0)?
        .broadcast_as(mask_f32.shape())?;

    // Where mask is true, use position; else use large value
    let large = Tensor::new(&[vocab as f32 + 1.0], probs.device())?
        .broadcast_as(mask_f32.shape())?;
    let masked_positions = mask.where_cond(&positions, &large)?;

    // Argmin gives first True position
    Ok(masked_positions.argmin(D::Minus1)?)
}

/// Generate a random f32 in [0, 1)
fn rand_f32() -> f32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos() as u64;
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);

    // LCG with seed and counter
    let state = seed.wrapping_add(count).wrapping_mul(1103515245).wrapping_add(12345);
    (state as f32) / (u64::MAX as f32)
}

/// Apply repetition penalty to logits
pub fn apply_repetition_penalty(
    logits: &Tensor,
    input_ids: &Tensor,
    penalty: f64,
) -> Result<Tensor> {
    if (penalty - 1.0).abs() < 1e-9 {
        return Ok(logits.clone());
    }

    let (batch, vocab) = logits.dims2()?;
    let input_ids_vec: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;
    let mut logits_vec: Vec<f32> = logits.flatten_all()?.to_vec1()?;

    // Apply penalty to previously generated tokens
    for token_id in input_ids_vec {
        let idx = token_id as usize;
        if idx < vocab {
            for b in 0..batch {
                let i = b * vocab + idx;
                if logits_vec[i] > 0.0 {
                    logits_vec[i] /= penalty as f32;
                } else {
                    logits_vec[i] *= penalty as f32;
                }
            }
        }
    }

    Ok(Tensor::new(logits_vec.as_slice(), logits.device())?.reshape((batch, vocab))?)
}

/// Greedy sampling (argmax)
pub fn greedy_sample(logits: &Tensor) -> Result<Tensor> {
    Ok(logits.argmax(D::Minus1)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_new_tokens, 2048);
        assert!((config.temperature - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_cumulative_sum() {
        let device = Device::Cpu;
        let x = Tensor::new(&[[0.1f32, 0.2, 0.3, 0.4]], &device).unwrap();
        let cumsum = cumulative_sum(&x).unwrap();
        let result: Vec<f32> = cumsum.flatten_all().unwrap().to_vec1().unwrap();
        assert!((result[0] - 0.1).abs() < 1e-5);
        assert!((result[1] - 0.3).abs() < 1e-5);
        assert!((result[2] - 0.6).abs() < 1e-5);
        assert!((result[3] - 1.0).abs() < 1e-5);
    }
}
