//! ConvNeXt Block
//!
//! A ConvNeXt-style block used in the audio decoder upsampling stages.
//! Consists of depthwise conv, LayerNorm, pointwise convs with GELU, and residual.

use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{layer_norm, linear, LayerNorm, LayerNormConfig, Linear, VarBuilder};

use super::CausalConv1d;

/// ConvNeXt Block
///
/// Architecture:
/// 1. Depthwise causal conv (groups=dim, kernel=7)
/// 2. LayerNorm
/// 3. Pointwise conv 1 (dim -> 4*dim)
/// 4. GELU activation
/// 5. Pointwise conv 2 (4*dim -> dim)
/// 6. Gamma scaling
/// 7. Residual connection
pub struct ConvNeXtBlock {
    /// Depthwise causal convolution
    dwconv: CausalConv1d,
    /// Layer normalization
    norm: LayerNorm,
    /// Pointwise conv 1 (expansion)
    pwconv1: Linear,
    /// Pointwise conv 2 (projection)
    pwconv2: Linear,
    /// Learnable scale parameter
    gamma: Tensor,
}

impl ConvNeXtBlock {
    /// Create a new ConvNeXt block.
    ///
    /// # Arguments
    /// * `dim` - Number of channels
    /// * `vb` - Variable builder for loading weights
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        // Depthwise causal conv: [dim, 1, 7] with groups=dim
        let dwconv = CausalConv1d::new(dim, dim, 7, 1, vb.pp("dwconv.conv"))?;

        // LayerNorm
        let norm_config = LayerNormConfig {
            eps: 1e-6,
            ..Default::default()
        };
        let norm = layer_norm(dim, norm_config, vb.pp("norm"))?;

        // Pointwise convs (implemented as Linear)
        let pwconv1 = linear(dim, 4 * dim, vb.pp("pwconv1"))?;
        let pwconv2 = linear(4 * dim, dim, vb.pp("pwconv2"))?;

        // Gamma scale
        let gamma = vb.get((dim,), "gamma")?;

        Ok(Self {
            dwconv,
            norm,
            pwconv1,
            pwconv2,
            gamma,
        })
    }

    /// Create from raw weight tensors.
    #[allow(clippy::too_many_arguments)]
    pub fn from_weights(
        dwconv_weight: Tensor,
        dwconv_bias: Option<Tensor>,
        norm_weight: Tensor,
        norm_bias: Tensor,
        pwconv1_weight: Tensor,
        pwconv1_bias: Tensor,
        pwconv2_weight: Tensor,
        pwconv2_bias: Tensor,
        gamma: Tensor,
    ) -> Result<Self> {
        // Depthwise conv with groups=dim (each channel has its own kernel)
        let dim = dwconv_weight.dim(0)?;
        let dwconv = CausalConv1d::from_weights_grouped(
            dwconv_weight,
            dwconv_bias,
            1,   // dilation
            dim, // groups = dim for depthwise
        )?;

        // LayerNorm
        let norm = LayerNorm::new(norm_weight, norm_bias, 1e-6);

        // Pointwise convs
        let pwconv1 = Linear::new(pwconv1_weight, Some(pwconv1_bias));
        let pwconv2 = Linear::new(pwconv2_weight, Some(pwconv2_bias));

        Ok(Self {
            dwconv,
            norm,
            pwconv1,
            pwconv2,
            gamma,
        })
    }

    /// Forward pass.
    ///
    /// Input shape: [batch, channels, seq_len]
    /// Output shape: [batch, channels, seq_len]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();

        // Depthwise causal conv
        let hidden = self.dwconv.forward(x)?;

        // Transpose: [B, C, T] -> [B, T, C]
        let hidden = hidden.transpose(1, 2)?;

        // LayerNorm
        let hidden = self.norm.forward(&hidden)?;

        // Pointwise conv 1 (expansion)
        let hidden = self.pwconv1.forward(&hidden)?;

        // GELU activation (exact erf-based, matches PyTorch nn.GELU())
        let hidden = hidden.gelu_erf()?;

        // Pointwise conv 2 (projection)
        let hidden = self.pwconv2.forward(&hidden)?;

        // Gamma scaling
        let hidden = hidden.broadcast_mul(&self.gamma)?;

        // Transpose back: [B, T, C] -> [B, C, T]
        let hidden = hidden.transpose(1, 2)?;

        // Residual connection
        Ok((residual + hidden)?)
    }
}

impl Module for ConvNeXtBlock {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        ConvNeXtBlock::forward(self, x).map_err(|e| candle_core::Error::Msg(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_convnext_block_construction() {
        // VarBuilder from VarMap creates zero-initialized tensors on demand
        // so we test with from_weights instead of new()
        let device = Device::Cpu;
        let dim = 16;

        let dwconv_w = Tensor::randn(0.0f32, 0.1, (dim, 1, 7), &device).unwrap();
        let dwconv_b = Tensor::zeros((dim,), DType::F32, &device).unwrap();
        let norm_w = Tensor::ones((dim,), DType::F32, &device).unwrap();
        let norm_b = Tensor::zeros((dim,), DType::F32, &device).unwrap();
        let pwconv1_w = Tensor::randn(0.0f32, 0.1, (4 * dim, dim), &device).unwrap();
        let pwconv1_b = Tensor::zeros((4 * dim,), DType::F32, &device).unwrap();
        let pwconv2_w = Tensor::randn(0.0f32, 0.1, (dim, 4 * dim), &device).unwrap();
        let pwconv2_b = Tensor::zeros((dim,), DType::F32, &device).unwrap();
        let gamma = Tensor::ones((dim,), DType::F32, &device).unwrap();

        let block = ConvNeXtBlock::from_weights(
            dwconv_w,
            Some(dwconv_b),
            norm_w,
            norm_b,
            pwconv1_w,
            pwconv1_b,
            pwconv2_w,
            pwconv2_b,
            gamma,
        );
        assert!(block.is_ok());
    }

    #[test]
    fn test_convnext_block_shape() {
        let device = Device::Cpu;
        let dim = 32;

        // Create weights manually
        // dwconv: [dim, 1, 7] for depthwise conv with groups=dim
        let dwconv_w = Tensor::randn(0.0f32, 0.1, (dim, 1, 7), &device).unwrap();
        let dwconv_b = Tensor::zeros((dim,), DType::F32, &device).unwrap();

        let norm_w = Tensor::ones((dim,), DType::F32, &device).unwrap();
        let norm_b = Tensor::zeros((dim,), DType::F32, &device).unwrap();

        let pwconv1_w = Tensor::randn(0.0f32, 0.1, (4 * dim, dim), &device).unwrap();
        let pwconv1_b = Tensor::zeros((4 * dim,), DType::F32, &device).unwrap();

        let pwconv2_w = Tensor::randn(0.0f32, 0.1, (dim, 4 * dim), &device).unwrap();
        let pwconv2_b = Tensor::zeros((dim,), DType::F32, &device).unwrap();

        let gamma = Tensor::ones((dim,), DType::F32, &device).unwrap();

        let block = ConvNeXtBlock::from_weights(
            dwconv_w,
            Some(dwconv_b),
            norm_w,
            norm_b,
            pwconv1_w,
            pwconv1_b,
            pwconv2_w,
            pwconv2_b,
            gamma,
        )
        .unwrap();

        let input = Tensor::randn(0.0f32, 1.0, (2, dim, 10), &device).unwrap();
        let output = block.forward(&input).unwrap();

        // Output should have same shape as input
        assert_eq!(output.dims(), &[2, dim, 10]);
    }
}
