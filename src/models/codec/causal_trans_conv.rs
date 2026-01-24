//! Causal Transposed 1D Convolution
//!
//! A ConvTranspose1d that maintains causality by trimming output.
//! Used for upsampling in the audio decoder.
//!
//! Matches the official Qwen3-TTS implementation which trims ceil(pad) from
//! both left and right sides, where pad = kernel_size - stride.

use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{conv_transpose1d, ConvTranspose1d, ConvTranspose1dConfig, VarBuilder};

/// Causal Transposed 1D Convolution
///
/// Applies ConvTranspose1d and trims the output to match the official model.
/// The trim amount is `ceil(kernel_size - stride)` from both left and right.
///
/// Note: This produces output shorter than `input * stride` when kernel > stride.
/// This matches the official Qwen3-TTS tokenizer behavior.
pub struct CausalTransConv1d {
    conv: ConvTranspose1d,
    /// Number of samples to trim from the left of output
    left_trim: usize,
    /// Number of samples to trim from the right of output
    right_trim: usize,
}

impl CausalTransConv1d {
    /// Create a new causal transposed conv1d layer.
    ///
    /// # Arguments
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - Size of the convolving kernel
    /// * `stride` - Stride of the convolution (upsampling factor)
    /// * `vb` - Variable builder for loading weights
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let config = ConvTranspose1dConfig {
            padding: 0,
            output_padding: 0,
            stride,
            dilation: 1,
            groups: 1,
        };

        let conv = conv_transpose1d(in_channels, out_channels, kernel_size, config, vb)?;

        // Match official Qwen3-TTS implementation:
        // pad = kernel_size - stride
        // left_pad = ceil(pad)
        // right_pad = left_pad (same as left)
        let pad = kernel_size.saturating_sub(stride);
        let left_trim = pad; // ceil(pad) where pad is already integer
        let right_trim = left_trim;

        Ok(Self {
            conv,
            left_trim,
            right_trim,
        })
    }

    /// Create from raw weight and bias tensors.
    ///
    /// Weight should have shape [in_channels, out_channels, kernel_size].
    pub fn from_weights(weight: Tensor, bias: Option<Tensor>, stride: usize) -> Result<Self> {
        let kernel_size = weight.dim(2)?;

        let config = ConvTranspose1dConfig {
            padding: 0,
            output_padding: 0,
            stride,
            dilation: 1,
            groups: 1,
        };

        let conv = ConvTranspose1d::new(weight, bias, config);

        // Match official Qwen3-TTS implementation:
        // pad = kernel_size - stride
        // left_pad = ceil(pad)
        // right_pad = left_pad (same as left)
        let pad = kernel_size.saturating_sub(stride);
        let left_trim = pad; // ceil(pad) where pad is already integer
        let right_trim = left_trim;

        Ok(Self {
            conv,
            left_trim,
            right_trim,
        })
    }

    /// Forward pass with causal output trimming.
    ///
    /// Input shape: [batch, in_channels, seq_len]
    /// Output shape: [batch, out_channels, seq_len * stride]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Apply transposed convolution
        let out = self.conv.forward(x)?;

        // Trim output for causality
        let out_len = out.dim(2)?;
        if self.left_trim > 0 || self.right_trim > 0 {
            let end = out_len.saturating_sub(self.right_trim);
            Ok(out.narrow(2, self.left_trim, end - self.left_trim)?)
        } else {
            Ok(out)
        }
    }

    /// Get the stride (upsampling factor)
    pub fn stride(&self) -> usize {
        self.conv.config().stride
    }
}

impl Module for CausalTransConv1d {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        CausalTransConv1d::forward(self, x).map_err(|e| candle_core::Error::Msg(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_causal_trans_conv_shape() {
        let device = Device::Cpu;

        // Create random weights: [in_channels, out_channels, kernel_size]
        // Note: ConvTranspose1d weight is [in, out, kernel], not [out, in, kernel]
        // kernel=4, stride=2
        let weight = Tensor::randn(0.0f32, 0.1, (64, 32, 4), &device).unwrap();
        let bias = Tensor::randn(0.0f32, 0.1, (32,), &device).unwrap();

        let conv = CausalTransConv1d::from_weights(weight, Some(bias), 2).unwrap();

        // Verify trimming matches official: trim = kernel - stride = 2 from each side
        assert_eq!(conv.left_trim, 2);
        assert_eq!(conv.right_trim, 2);

        // Input: [batch=1, channels=64, seq=10]
        let input = Tensor::randn(0.0f32, 1.0, (1, 64, 10), &device).unwrap();
        let output = conv.forward(&input).unwrap();

        // Official behavior: raw = (10-1)*2 + 4 = 22, trim 4 total -> 18
        // This is (input - 1) * stride when kernel = 2 * stride
        assert_eq!(output.dims(), &[1, 32, 18]);
    }

    #[test]
    fn test_causal_trans_conv_stride_equals_kernel() {
        // When stride == kernel_size, no trimming needed
        let device = Device::Cpu;

        let weight = Tensor::randn(0.0f32, 0.1, (32, 32, 2), &device).unwrap();
        let bias = Tensor::zeros((32,), DType::F32, &device).unwrap();

        let conv = CausalTransConv1d::from_weights(weight, Some(bias), 2).unwrap();
        // No trimming needed when kernel == stride
        assert_eq!(conv.left_trim, 0);
        assert_eq!(conv.right_trim, 0);

        let input = Tensor::randn(0.0f32, 1.0, (1, 32, 5), &device).unwrap();
        let output = conv.forward(&input).unwrap();

        // 5 * 2 = 10 (exact upsampling when kernel == stride)
        assert_eq!(output.dims(), &[1, 32, 10]);
    }

    #[test]
    fn test_causal_trans_conv_various_strides() {
        // Test with kernel = 2 * stride (typical decoder block configuration)
        let device = Device::Cpu;

        // (kernel, stride) -> expected output for input_len=4
        // Official formula: output = (input - 1) * stride when kernel = 2 * stride
        let test_cases = [
            (16, 8, 24),  // (4-1)*8 = 24
            (10, 5, 15),  // (4-1)*5 = 15
            (8, 4, 12),   // (4-1)*4 = 12
            (6, 3, 9),    // (4-1)*3 = 9
        ];

        for (kernel_size, stride, expected_len) in test_cases {
            let weight = Tensor::randn(0.0f32, 0.1, (16, 8, kernel_size), &device).unwrap();
            let bias = Tensor::zeros((8,), DType::F32, &device).unwrap();

            let conv = CausalTransConv1d::from_weights(weight, Some(bias), stride).unwrap();

            // Verify trimming: pad = kernel - stride, trim from both sides
            let expected_trim = kernel_size - stride;
            assert_eq!(conv.left_trim, expected_trim);
            assert_eq!(conv.right_trim, expected_trim);

            let input = Tensor::randn(0.0f32, 1.0, (1, 16, 4), &device).unwrap();
            let output = conv.forward(&input).unwrap();

            assert_eq!(
                output.dims(),
                &[1, 8, expected_len],
                "Failed for kernel={}, stride={}",
                kernel_size,
                stride
            );
        }
    }
}
