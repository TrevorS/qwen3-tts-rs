//! Causal Transposed 1D Convolution
//!
//! A ConvTranspose1d that maintains causality by trimming output.
//! Used for upsampling in the audio decoder.

use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{conv_transpose1d, ConvTranspose1d, ConvTranspose1dConfig, VarBuilder};

/// Causal Transposed 1D Convolution
///
/// Applies ConvTranspose1d and trims the output to maintain causality.
/// The trim amount is calculated as `(kernel_size - stride) / 2` from each side.
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

        // Calculate trim amounts to maintain causality
        // Total trim = kernel_size - stride, split between left and right
        let pad = kernel_size.saturating_sub(stride);
        let left_trim = pad.div_ceil(2);
        let right_trim = pad - left_trim;

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

        // Calculate trim amounts
        let pad = kernel_size.saturating_sub(stride);
        let left_trim = pad.div_ceil(2);
        let right_trim = pad - left_trim;

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
        let weight = Tensor::randn(0.0f32, 0.1, (64, 32, 4), &device).unwrap();
        let bias = Tensor::randn(0.0f32, 0.1, (32,), &device).unwrap();

        let conv = CausalTransConv1d::from_weights(weight, Some(bias), 2).unwrap();

        // Input: [batch=1, channels=64, seq=10]
        let input = Tensor::randn(0.0f32, 1.0, (1, 64, 10), &device).unwrap();
        let output = conv.forward(&input).unwrap();

        // Output should be upsampled by stride=2: 10 * 2 = 20
        assert_eq!(output.dims(), &[1, 32, 20]);
    }

    #[test]
    fn test_causal_trans_conv_stride_equals_kernel() {
        // When stride == kernel_size, no trimming needed
        let device = Device::Cpu;

        let weight = Tensor::randn(0.0f32, 0.1, (32, 32, 2), &device).unwrap();
        let bias = Tensor::zeros((32,), DType::F32, &device).unwrap();

        let conv = CausalTransConv1d::from_weights(weight, Some(bias), 2).unwrap();
        assert_eq!(conv.left_trim, 0);
        assert_eq!(conv.right_trim, 0);

        let input = Tensor::randn(0.0f32, 1.0, (1, 32, 5), &device).unwrap();
        let output = conv.forward(&input).unwrap();

        // 5 * 2 = 10
        assert_eq!(output.dims(), &[1, 32, 10]);
    }

    #[test]
    fn test_causal_trans_conv_various_strides() {
        let device = Device::Cpu;

        for (kernel_size, stride) in [(16, 8), (10, 5), (8, 4), (6, 3)] {
            let weight = Tensor::randn(0.0f32, 0.1, (16, 8, kernel_size), &device).unwrap();
            let bias = Tensor::zeros((8,), DType::F32, &device).unwrap();

            let conv = CausalTransConv1d::from_weights(weight, Some(bias), stride).unwrap();

            let input = Tensor::randn(0.0f32, 1.0, (1, 16, 4), &device).unwrap();
            let output = conv.forward(&input).unwrap();

            // Expected output length: input_len * stride
            let expected_len = 4 * stride;
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
