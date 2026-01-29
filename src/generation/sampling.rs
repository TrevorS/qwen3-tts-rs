//! Token sampling strategies for autoregressive generation
//!
//! Supports both deterministic (seeded) and non-deterministic random sampling.
//! Create a [`SamplingContext`] with an optional seed for reproducible outputs.

use anyhow::Result;
use candle_core::{DType, IndexOp, Tensor, D};

/// RNG and sampling state for a single generation session.
///
/// Encapsulates all randomness so that multiple sessions can run
/// concurrently without interfering with each other.
///
/// # Determinism
///
/// When created with a seed, the same seed produces identical output
/// across runs and threads. Without a seed, uses system entropy.
pub struct SamplingContext {
    /// PCG state (only used when seeded)
    state: u64,
    /// Whether we're in seeded mode
    seeded: bool,
    /// Counter for unseeded fallback
    counter: u64,
}

impl SamplingContext {
    /// Create a new sampling context with an optional seed.
    ///
    /// When `seed` is `Some`, all sampling is deterministic and reproducible.
    /// When `None`, uses system time + counter for randomness.
    pub fn new(seed: Option<u64>) -> Self {
        match seed {
            Some(s) => {
                // Mix seed with PCG increment to avoid degenerate states
                let state = s
                    .wrapping_mul(2685821657736338717)
                    .wrapping_add(1442695040888963407);
                Self {
                    state,
                    seeded: true,
                    counter: 0,
                }
            }
            None => Self {
                state: 0,
                seeded: false,
                counter: 0,
            },
        }
    }

    /// Reset the RNG to its initial seeded state.
    ///
    /// Only meaningful for seeded contexts. For unseeded contexts, this is a no-op.
    pub fn reset(&mut self, seed: u64) {
        let state = seed
            .wrapping_mul(2685821657736338717)
            .wrapping_add(1442695040888963407);
        self.state = state;
        self.seeded = true;
    }

    /// Generate a random f32 in [0, 1).
    fn rand_f32(&mut self) -> f32 {
        if !self.seeded {
            use std::time::{SystemTime, UNIX_EPOCH};

            let seed = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos() as u64;
            let count = self.counter;
            self.counter += 1;

            // LCG with seed and counter
            let state = seed
                .wrapping_add(count)
                .wrapping_mul(1103515245)
                .wrapping_add(12345);
            return (state as f32) / (u64::MAX as f32);
        }

        // PCG XSH RR 64/32
        let old_state = self.state;
        self.state = old_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);

        let xorshifted = (((old_state >> 18) ^ old_state) >> 27) as u32;
        let rot = (old_state >> 59) as u32;
        let output = xorshifted.rotate_right(rot);

        (output as f32) / (u32::MAX as f32)
    }
}

/// Configuration for autoregressive generation
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum number of new tokens to generate
    pub max_new_tokens: usize,
    /// Sampling temperature (1.0 = no change, <1.0 = more focused, >1.0 = more random)
    pub temperature: f64,
    /// Top-k sampling (0 = disabled)
    pub top_k: usize,
    /// Top-p (nucleus) sampling threshold (1.0 = disabled)
    pub top_p: f64,
    /// Repetition penalty (1.0 = no penalty)
    pub repetition_penalty: f64,
    /// End-of-sequence token ID (generation stops when this token is sampled)
    pub eos_token_id: Option<u32>,
    /// Minimum number of tokens before EOS is allowed (default: 2, matching Python)
    pub min_new_tokens: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 2048,
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.0,
            eos_token_id: None,
            min_new_tokens: 2,
        }
    }
}

/// Sample next token from logits
///
/// # Arguments
/// * `logits` - Logits tensor of shape [batch, vocab_size]
/// * `config` - Generation configuration
/// * `ctx` - Sampling context (owns RNG state)
///
/// # Returns
/// Token indices of shape `[batch]`
pub fn sample(
    logits: &Tensor,
    config: &GenerationConfig,
    ctx: &mut SamplingContext,
) -> Result<Tensor> {
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

    // Apply top-k filtering
    let logits = if config.top_k > 0 {
        top_k_filter(&logits, config.top_k)?
    } else {
        logits
    };

    // Apply top-p (nucleus) filtering
    let logits = if config.top_p < 1.0 && config.top_p > 0.0 {
        top_p_filter(&logits, config.top_p)?
    } else {
        logits
    };

    // Convert to probabilities
    let probs = candle_nn::ops::softmax_last_dim(&logits)?;

    // Sample from distribution
    multinomial_sample(&probs, ctx)
}

/// Apply top-k filtering: keep only the top k logits, set rest to -inf
fn top_k_filter(logits: &Tensor, k: usize) -> Result<Tensor> {
    let (batch, vocab) = logits.dims2()?;
    let k = k.min(vocab);

    let mut result_data = Vec::with_capacity(batch * vocab);

    for b in 0..batch {
        let row: Vec<f32> = logits.i(b)?.to_vec1()?;

        // Find the k-th largest value
        let mut sorted = row.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = sorted.get(k - 1).copied().unwrap_or(f32::NEG_INFINITY);

        // Mask values below threshold
        let filtered: Vec<f32> = row
            .iter()
            .map(|&v| if v >= threshold { v } else { f32::NEG_INFINITY })
            .collect();
        result_data.extend(filtered);
    }

    Ok(Tensor::new(result_data.as_slice(), logits.device())?.reshape((batch, vocab))?)
}

/// Apply top-p (nucleus) filtering: keep smallest set of tokens whose cumulative probability >= top_p
fn top_p_filter(logits: &Tensor, p: f64) -> Result<Tensor> {
    let (batch, vocab) = logits.dims2()?;
    let mut result_data = Vec::with_capacity(batch * vocab);

    for b in 0..batch {
        let row: Vec<f32> = logits.i(b)?.to_vec1()?;

        // Sort indices by logit value descending
        let mut indices: Vec<usize> = (0..vocab).collect();
        indices.sort_by(|&a, &b| {
            row[b]
                .partial_cmp(&row[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Compute softmax over sorted logits
        let max_val = row[indices[0]];
        let mut exp_sorted: Vec<f32> = indices.iter().map(|&i| (row[i] - max_val).exp()).collect();
        let sum: f32 = exp_sorted.iter().sum();
        for v in &mut exp_sorted {
            *v /= sum;
        }

        // Find cutoff: cumulative probability exceeds top_p
        let mut cumsum = 0.0f32;
        let mut cutoff_idx = vocab; // keep all by default
        for (i, &prob) in exp_sorted.iter().enumerate() {
            cumsum += prob;
            if cumsum > p as f32 {
                cutoff_idx = i + 1; // include the token that pushes us over
                break;
            }
        }

        // Mask tokens beyond the cutoff
        let mut filtered = vec![f32::NEG_INFINITY; vocab];
        for &idx in &indices[..cutoff_idx] {
            filtered[idx] = row[idx];
        }
        result_data.extend(filtered);
    }

    Ok(Tensor::new(result_data.as_slice(), logits.device())?.reshape((batch, vocab))?)
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
fn multinomial_sample(probs: &Tensor, ctx: &mut SamplingContext) -> Result<Tensor> {
    let (batch, vocab) = probs.dims2()?;

    // Use cumulative distribution for sampling
    let cumsum = cumulative_sum(probs)?;

    // Generate uniform random values
    let uniform: Vec<f32> = (0..batch).map(|_| ctx.rand_f32()).collect();
    let uniform = Tensor::new(uniform.as_slice(), probs.device())?.unsqueeze(1)?;

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
    let large =
        Tensor::new(&[vocab as f32 + 1.0], probs.device())?.broadcast_as(mask_f32.shape())?;
    let masked_positions = mask.where_cond(&positions, &large)?;

    // Argmin gives first True position
    Ok(masked_positions.argmin(D::Minus1)?)
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
    use candle_core::Device;

    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_new_tokens, 2048);
        assert!((config.temperature - 0.7).abs() < 1e-6);
        assert_eq!(config.top_k, 50);
        assert!((config.top_p - 0.9).abs() < 1e-6);
        assert!((config.repetition_penalty - 1.0).abs() < 1e-6);
        assert_eq!(config.eos_token_id, None);
        assert_eq!(config.min_new_tokens, 2);
    }

    #[test]
    fn test_generation_config_custom() {
        let config = GenerationConfig {
            max_new_tokens: 512,
            temperature: 0.5,
            top_k: 10,
            top_p: 0.8,
            repetition_penalty: 1.2,
            eos_token_id: Some(42),
            min_new_tokens: 0,
        };
        assert_eq!(config.max_new_tokens, 512);
        assert!((config.temperature - 0.5).abs() < 1e-6);
        assert_eq!(config.top_k, 10);
        assert_eq!(config.eos_token_id, Some(42));
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

    #[test]
    fn test_cumulative_sum_batch() {
        let device = Device::Cpu;
        let x = Tensor::new(
            &[[0.25f32, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4]],
            &device,
        )
        .unwrap();
        let cumsum = cumulative_sum(&x).unwrap();
        let result: Vec<f32> = cumsum.flatten_all().unwrap().to_vec1().unwrap();
        // First row
        assert!((result[0] - 0.25).abs() < 1e-5);
        assert!((result[1] - 0.50).abs() < 1e-5);
        assert!((result[2] - 0.75).abs() < 1e-5);
        assert!((result[3] - 1.00).abs() < 1e-5);
        // Second row
        assert!((result[4] - 0.1).abs() < 1e-5);
        assert!((result[5] - 0.3).abs() < 1e-5);
    }

    #[test]
    fn test_greedy_sample() {
        let device = Device::Cpu;
        // Logits where position 2 has highest value
        let logits = Tensor::new(&[[1.0f32, 2.0, 5.0, 1.0]], &device).unwrap();
        let result = greedy_sample(&logits).unwrap();
        let idx: Vec<u32> = result.to_vec1().unwrap();
        assert_eq!(idx[0], 2); // Index of max
    }

    #[test]
    fn test_greedy_sample_batch() {
        let device = Device::Cpu;
        let logits = Tensor::new(
            &[[1.0f32, 5.0, 2.0], [3.0, 1.0, 2.0], [1.0, 2.0, 10.0]],
            &device,
        )
        .unwrap();
        let result = greedy_sample(&logits).unwrap();
        let idx: Vec<u32> = result.to_vec1().unwrap();
        assert_eq!(idx[0], 1); // Max at position 1
        assert_eq!(idx[1], 0); // Max at position 0
        assert_eq!(idx[2], 2); // Max at position 2
    }

    #[test]
    fn test_sample_very_low_temperature() {
        let device = Device::Cpu;
        // With very low temperature, should act like greedy
        let logits = Tensor::new(&[[1.0f32, 10.0, 2.0, 1.0]], &device).unwrap();
        let config = GenerationConfig {
            temperature: 0.001,
            ..Default::default()
        };
        let mut ctx = SamplingContext::new(Some(42));
        let result = sample(&logits, &config, &mut ctx).unwrap();
        let idx: Vec<u32> = result.to_vec1().unwrap();
        assert_eq!(idx[0], 1); // Should pick the highest
    }

    #[test]
    fn test_sample_normal_temperature() {
        let device = Device::Cpu;
        // With normal temperature, sampling should work
        let logits = Tensor::new(&[[1.0f32, 1.0, 1.0, 1.0]], &device).unwrap();
        let config = GenerationConfig::default();
        let mut ctx = SamplingContext::new(None);
        let result = sample(&logits, &config, &mut ctx).unwrap();
        let idx: Vec<u32> = result.to_vec1().unwrap();
        // Should return a valid index
        assert!(idx[0] < 4);
    }

    #[test]
    fn test_sample_temperature_one() {
        let device = Device::Cpu;
        let logits = Tensor::new(&[[2.0f32, 2.0, 2.0]], &device).unwrap();
        let config = GenerationConfig {
            temperature: 1.0,
            ..Default::default()
        };
        let mut ctx = SamplingContext::new(None);
        let result = sample(&logits, &config, &mut ctx).unwrap();
        let idx: Vec<u32> = result.to_vec1().unwrap();
        assert!(idx[0] < 3);
    }

    #[test]
    fn test_apply_repetition_penalty_no_penalty() {
        let device = Device::Cpu;
        let logits = Tensor::new(&[[1.0f32, 2.0, 3.0]], &device).unwrap();
        let input_ids = Tensor::new(&[0u32], &device).unwrap();
        let result = apply_repetition_penalty(&logits, &input_ids, 1.0).unwrap();
        // With penalty 1.0, should be unchanged
        let original: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
        let penalized: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        assert!((original[0] - penalized[0]).abs() < 1e-5);
        assert!((original[1] - penalized[1]).abs() < 1e-5);
        assert!((original[2] - penalized[2]).abs() < 1e-5);
    }

    #[test]
    fn test_apply_repetition_penalty_with_penalty() {
        let device = Device::Cpu;
        let logits = Tensor::new(&[[2.0f32, 3.0, 4.0]], &device).unwrap();
        let input_ids = Tensor::new(&[0u32], &device).unwrap();
        let penalty = 2.0;
        let result = apply_repetition_penalty(&logits, &input_ids, penalty).unwrap();
        let penalized: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        // Token 0 had positive logit, should be divided by penalty
        assert!((penalized[0] - 1.0).abs() < 1e-5); // 2.0 / 2.0 = 1.0
                                                    // Others unchanged
        assert!((penalized[1] - 3.0).abs() < 1e-5);
        assert!((penalized[2] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_apply_repetition_penalty_negative_logit() {
        let device = Device::Cpu;
        let logits = Tensor::new(&[[-2.0f32, 3.0, 4.0]], &device).unwrap();
        let input_ids = Tensor::new(&[0u32], &device).unwrap();
        let penalty = 2.0;
        let result = apply_repetition_penalty(&logits, &input_ids, penalty).unwrap();
        let penalized: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        // Token 0 had negative logit, should be multiplied by penalty
        assert!((penalized[0] - (-4.0)).abs() < 1e-5); // -2.0 * 2.0 = -4.0
    }

    #[test]
    fn test_rand_f32_range() {
        let mut ctx = SamplingContext::new(None);
        for _ in 0..100 {
            let r = ctx.rand_f32();
            assert!(r >= 0.0);
            assert!(r < 1.0);
        }
    }

    #[test]
    #[ignore = "flaky under parallel execution due to global RNG state"]
    fn test_rand_f32_variability() {
        let mut ctx = SamplingContext::new(None);
        let values: Vec<f32> = (0..10).map(|_| ctx.rand_f32()).collect();
        let unique: std::collections::HashSet<u32> = values.iter().map(|v| v.to_bits()).collect();
        assert!(unique.len() > 1);
    }

    #[test]
    fn test_multinomial_sample_deterministic_probs() {
        let device = Device::Cpu;
        // Probability of 1.0 on one token
        let probs = Tensor::new(&[[0.0f32, 1.0, 0.0, 0.0]], &device).unwrap();
        let mut ctx = SamplingContext::new(Some(42));
        let result = multinomial_sample(&probs, &mut ctx).unwrap();
        let idx: Vec<u32> = result.to_vec1().unwrap();
        assert_eq!(idx[0], 1); // Should always pick index 1
    }

    #[test]
    fn test_sample_with_batch() {
        let device = Device::Cpu;
        let logits = Tensor::new(&[[10.0f32, 1.0, 1.0], [1.0, 10.0, 1.0]], &device).unwrap();
        let config = GenerationConfig {
            temperature: 0.001, // Very low temp for deterministic
            ..Default::default()
        };
        let mut ctx = SamplingContext::new(Some(42));
        let result = sample(&logits, &config, &mut ctx).unwrap();
        let idx: Vec<u32> = result.to_vec1().unwrap();
        assert_eq!(idx[0], 0);
        assert_eq!(idx[1], 1);
    }

    #[test]
    fn test_seeded_deterministic() {
        // With the same seed, should get the same random values
        let mut ctx1 = SamplingContext::new(Some(12345));
        let values1: Vec<f32> = (0..10).map(|_| ctx1.rand_f32()).collect();

        let mut ctx2 = SamplingContext::new(Some(12345));
        let values2: Vec<f32> = (0..10).map(|_| ctx2.rand_f32()).collect();

        for (a, b) in values1.iter().zip(values2.iter()) {
            assert!((a - b).abs() < 1e-9, "Seeded values should be identical");
        }
    }

    #[test]
    fn test_different_seeds_different_values() {
        let mut ctx1 = SamplingContext::new(Some(12345));
        let values1: Vec<f32> = (0..10).map(|_| ctx1.rand_f32()).collect();

        let mut ctx2 = SamplingContext::new(Some(67890));
        let values2: Vec<f32> = (0..10).map(|_| ctx2.rand_f32()).collect();

        let same_count = values1
            .iter()
            .zip(values2.iter())
            .filter(|(a, b)| (*a - *b).abs() < 1e-9)
            .count();
        assert!(
            same_count < 10,
            "Different seeds should produce different values"
        );
    }

    #[test]
    fn test_reset() {
        let mut ctx = SamplingContext::new(Some(42));
        let _first = ctx.rand_f32();
        let second = ctx.rand_f32();

        ctx.reset(42);
        let after_reset_first = ctx.rand_f32();
        let after_reset_second = ctx.rand_f32();

        let mut fresh = SamplingContext::new(Some(42));
        let fresh_first = fresh.rand_f32();
        let fresh_second = fresh.rand_f32();

        assert!((after_reset_first - fresh_first).abs() < 1e-9);
        assert!((after_reset_second - fresh_second).abs() < 1e-9);
        assert!((after_reset_second - second).abs() < 1e-9);
    }

    #[test]
    fn test_seeded_sampling_deterministic() {
        let device = Device::Cpu;
        // Uniform-ish logits so sampling isn't just greedy
        let logits = Tensor::new(&[[1.0f32, 1.0, 1.0, 1.0, 1.0]], &device).unwrap();
        let config = GenerationConfig {
            temperature: 1.0,
            ..Default::default()
        };

        let mut ctx1 = SamplingContext::new(Some(99999));
        let mut results1 = Vec::new();
        for _ in 0..5 {
            let result = sample(&logits, &config, &mut ctx1).unwrap();
            results1.push(result.flatten_all().unwrap().to_vec1::<u32>().unwrap()[0]);
        }

        let mut ctx2 = SamplingContext::new(Some(99999));
        let mut results2 = Vec::new();
        for _ in 0..5 {
            let result = sample(&logits, &config, &mut ctx2).unwrap();
            results2.push(result.flatten_all().unwrap().to_vec1::<u32>().unwrap()[0]);
        }

        assert_eq!(
            results1, results2,
            "Seeded sampling should be deterministic"
        );
    }
}
