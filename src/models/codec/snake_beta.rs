//! SnakeBeta Activation Function
//!
//! A modified Snake activation with separate trainable parameters for
//! frequency (alpha) and magnitude (beta).
//!
//! Formula: x + (1/β) * sin²(α * x)
//! where α = exp(alpha_param) and β = exp(beta_param)

use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::VarBuilder;

/// SnakeBeta activation function
///
/// Reference: "Neural Networks Fail to Learn Periodic Functions and How to Fix It"
/// (<https://arxiv.org/abs/2006.08195>)
pub struct SnakeBeta {
    /// Learned frequency parameter (exponentiated before use)
    alpha: Tensor,
    /// Learned magnitude parameter (exponentiated before use)
    beta: Tensor,
    /// Small constant to prevent division by zero
    epsilon: f64,
}

impl SnakeBeta {
    /// Create a new SnakeBeta activation.
    ///
    /// # Arguments
    /// * `channels` - Number of channels (alpha and beta have shape `[channels]`)
    /// * `vb` - Variable builder for loading weights
    pub fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let alpha = vb.get((channels,), "alpha")?;
        let beta = vb.get((channels,), "beta")?;

        Ok(Self {
            alpha,
            beta,
            epsilon: 1e-9,
        })
    }

    /// Create from raw weight tensors.
    ///
    /// Alpha and beta should have shape `[channels]`.
    pub fn from_weights(alpha: Tensor, beta: Tensor) -> Result<Self> {
        Ok(Self {
            alpha,
            beta,
            epsilon: 1e-9,
        })
    }

    /// Forward pass.
    ///
    /// Input shape: [batch, channels, seq_len]
    /// Output shape: [batch, channels, seq_len]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Reshape alpha and beta for broadcasting: [channels] -> [1, channels, 1]
        let alpha = self.alpha.unsqueeze(0)?.unsqueeze(2)?;
        let beta = self.beta.unsqueeze(0)?.unsqueeze(2)?;

        // Exponentiate parameters
        let alpha = alpha.exp()?;
        let beta = beta.exp()?;

        // Compute sin²(alpha * x)
        let scaled_x = x.broadcast_mul(&alpha)?;
        let sin_term = scaled_x.sin()?.sqr()?;

        // Compute 1/(beta + epsilon)
        let inv_beta = (beta + self.epsilon)?.recip()?;

        // x + (1/beta) * sin²(alpha * x)
        let scaled_sin = sin_term.broadcast_mul(&inv_beta)?;
        Ok((x + scaled_sin)?)
    }
}

impl Module for SnakeBeta {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        SnakeBeta::forward(self, x).map_err(|e| candle_core::Error::Msg(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_snake_beta_shape() {
        let device = Device::Cpu;

        // Create alpha and beta with zeros (will exp to 1.0)
        let alpha = Tensor::zeros((64,), DType::F32, &device).unwrap();
        let beta = Tensor::zeros((64,), DType::F32, &device).unwrap();

        let snake = SnakeBeta::from_weights(alpha, beta).unwrap();

        let input = Tensor::randn(0.0f32, 1.0, (2, 64, 100), &device).unwrap();
        let output = snake.forward(&input).unwrap();

        assert_eq!(output.dims(), &[2, 64, 100]);
    }

    #[test]
    fn test_snake_beta_output_range() {
        let device = Device::Cpu;

        // With alpha=beta=0, we get exp(0)=1, so:
        // output = x + sin²(x), which is bounded by x + 1
        let alpha = Tensor::zeros((4,), DType::F32, &device).unwrap();
        let beta = Tensor::zeros((4,), DType::F32, &device).unwrap();

        let snake = SnakeBeta::from_weights(alpha, beta).unwrap();

        // Input in range [-1, 1]
        let input = Tensor::new(&[[[0.0f32, 0.5, -0.5, 1.0]]], &device)
            .unwrap()
            .broadcast_as((1, 4, 4))
            .unwrap();

        let output = snake.forward(&input).unwrap();

        // Output should be >= input (since we're adding a non-negative term)
        let input_sum: f32 = input.sum_all().unwrap().to_scalar().unwrap();
        let output_sum: f32 = output.sum_all().unwrap().to_scalar().unwrap();

        // sin²(x) is always >= 0, so output >= input
        assert!(output_sum >= input_sum - 0.01);
    }

    #[test]
    fn test_snake_beta_trainable_params() {
        let device = Device::Cpu;

        // Test that different alpha/beta give different outputs
        let alpha1 = Tensor::zeros((4,), DType::F32, &device).unwrap();
        let beta1 = Tensor::zeros((4,), DType::F32, &device).unwrap();

        let alpha2 = Tensor::ones((4,), DType::F32, &device).unwrap();
        let beta2 = Tensor::ones((4,), DType::F32, &device).unwrap();

        let snake1 = SnakeBeta::from_weights(alpha1, beta1).unwrap();
        let snake2 = SnakeBeta::from_weights(alpha2, beta2).unwrap();

        let input = Tensor::randn(0.0f32, 1.0, (1, 4, 10), &device).unwrap();

        let out1 = snake1.forward(&input).unwrap();
        let out2 = snake2.forward(&input).unwrap();

        // Outputs should differ
        let diff: f32 = (&out1 - &out2)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap();

        assert!(diff > 0.0);
    }
}
