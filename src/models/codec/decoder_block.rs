//! Decoder block components for BigVGAN-style audio synthesis
//!
//! These components are used in the final decoder stages for upsampling
//! and audio waveform generation.

use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::VarBuilder;

use super::{CausalConv1d, CausalTransConv1d, SnakeBeta};

/// Residual unit used in decoder blocks
///
/// Architecture:
/// 1. SnakeBeta activation
/// 2. Dilated causal conv (kernel=7)
/// 3. SnakeBeta activation
/// 4. 1x1 causal conv
/// 5. Residual connection
pub struct ResidualUnit {
    /// First activation
    act1: SnakeBeta,
    /// Dilated causal conv
    conv1: CausalConv1d,
    /// Second activation
    act2: SnakeBeta,
    /// 1x1 conv for output
    conv2: CausalConv1d,
}

impl ResidualUnit {
    /// Create a new residual unit.
    ///
    /// # Arguments
    /// * `dim` - Number of channels
    /// * `dilation` - Dilation for conv1 (1, 3, or 9)
    /// * `vb` - Variable builder for loading weights
    pub fn new(dim: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        let act1 = SnakeBeta::new(dim, vb.pp("act1"))?;
        let conv1 = CausalConv1d::new(dim, dim, 7, dilation, vb.pp("conv1.conv"))?;
        let act2 = SnakeBeta::new(dim, vb.pp("act2"))?;
        let conv2 = CausalConv1d::new(dim, dim, 1, 1, vb.pp("conv2.conv"))?;

        Ok(Self {
            act1,
            conv1,
            act2,
            conv2,
        })
    }

    /// Create from raw weight tensors.
    #[allow(clippy::too_many_arguments)]
    pub fn from_weights(
        act1_alpha: Tensor,
        act1_beta: Tensor,
        conv1_weight: Tensor,
        conv1_bias: Tensor,
        act2_alpha: Tensor,
        act2_beta: Tensor,
        conv2_weight: Tensor,
        conv2_bias: Tensor,
        dilation: usize,
    ) -> Result<Self> {
        let act1 = SnakeBeta::from_weights(act1_alpha, act1_beta)?;
        let conv1 = CausalConv1d::from_weights(conv1_weight, Some(conv1_bias), dilation)?;
        let act2 = SnakeBeta::from_weights(act2_alpha, act2_beta)?;
        let conv2 = CausalConv1d::from_weights(conv2_weight, Some(conv2_bias), 1)?;

        Ok(Self {
            act1,
            conv1,
            act2,
            conv2,
        })
    }

    /// Forward pass with residual connection.
    ///
    /// Input/Output shape: [batch, channels, seq_len]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();

        // SnakeBeta -> dilated conv -> SnakeBeta -> 1x1 conv
        let hidden = self.act1.forward(x)?;
        let hidden = self.conv1.forward(&hidden)?;
        let hidden = self.act2.forward(&hidden)?;
        let hidden = self.conv2.forward(&hidden)?;

        // Residual connection
        Ok((hidden + residual)?)
    }
}

impl Module for ResidualUnit {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        ResidualUnit::forward(self, x).map_err(|e| candle_core::Error::Msg(e.to_string()))
    }
}

/// BigVGAN-style decoder block with upsampling
///
/// Architecture:
/// 1. SnakeBeta activation
/// 2. CausalTransConv for upsampling (kernel=2*rate, stride=rate)
/// 3. ResidualUnit with dilation=1
/// 4. ResidualUnit with dilation=3
/// 5. ResidualUnit with dilation=9
pub struct DecoderBlock {
    /// Input activation
    snake: SnakeBeta,
    /// Upsampling transposed conv
    upsample: CausalTransConv1d,
    /// First residual unit (dilation=1)
    res1: ResidualUnit,
    /// Second residual unit (dilation=3)
    res2: ResidualUnit,
    /// Third residual unit (dilation=9)
    res3: ResidualUnit,
}

impl DecoderBlock {
    /// Create a new decoder block.
    ///
    /// # Arguments
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels (typically in_channels / 2)
    /// * `upsample_rate` - Upsampling factor (8, 5, 4, or 3)
    /// * `vb` - Variable builder for loading weights
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        upsample_rate: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let snake = SnakeBeta::new(in_channels, vb.pp("block.0"))?;
        let upsample = CausalTransConv1d::new(
            in_channels,
            out_channels,
            upsample_rate * 2,
            upsample_rate,
            vb.pp("block.1.conv"),
        )?;
        let res1 = ResidualUnit::new(out_channels, 1, vb.pp("block.2"))?;
        let res2 = ResidualUnit::new(out_channels, 3, vb.pp("block.3"))?;
        let res3 = ResidualUnit::new(out_channels, 9, vb.pp("block.4"))?;

        Ok(Self {
            snake,
            upsample,
            res1,
            res2,
            res3,
        })
    }

    /// Create from raw weight tensors.
    #[allow(clippy::too_many_arguments)]
    pub fn from_weights(
        snake_alpha: Tensor,
        snake_beta: Tensor,
        upsample_weight: Tensor,
        upsample_bias: Tensor,
        res1_act1_alpha: Tensor,
        res1_act1_beta: Tensor,
        res1_conv1_weight: Tensor,
        res1_conv1_bias: Tensor,
        res1_act2_alpha: Tensor,
        res1_act2_beta: Tensor,
        res1_conv2_weight: Tensor,
        res1_conv2_bias: Tensor,
        res2_act1_alpha: Tensor,
        res2_act1_beta: Tensor,
        res2_conv1_weight: Tensor,
        res2_conv1_bias: Tensor,
        res2_act2_alpha: Tensor,
        res2_act2_beta: Tensor,
        res2_conv2_weight: Tensor,
        res2_conv2_bias: Tensor,
        res3_act1_alpha: Tensor,
        res3_act1_beta: Tensor,
        res3_conv1_weight: Tensor,
        res3_conv1_bias: Tensor,
        res3_act2_alpha: Tensor,
        res3_act2_beta: Tensor,
        res3_conv2_weight: Tensor,
        res3_conv2_bias: Tensor,
        upsample_rate: usize,
    ) -> Result<Self> {
        let snake = SnakeBeta::from_weights(snake_alpha, snake_beta)?;
        let upsample =
            CausalTransConv1d::from_weights(upsample_weight, Some(upsample_bias), upsample_rate)?;
        let res1 = ResidualUnit::from_weights(
            res1_act1_alpha,
            res1_act1_beta,
            res1_conv1_weight,
            res1_conv1_bias,
            res1_act2_alpha,
            res1_act2_beta,
            res1_conv2_weight,
            res1_conv2_bias,
            1,
        )?;
        let res2 = ResidualUnit::from_weights(
            res2_act1_alpha,
            res2_act1_beta,
            res2_conv1_weight,
            res2_conv1_bias,
            res2_act2_alpha,
            res2_act2_beta,
            res2_conv2_weight,
            res2_conv2_bias,
            3,
        )?;
        let res3 = ResidualUnit::from_weights(
            res3_act1_alpha,
            res3_act1_beta,
            res3_conv1_weight,
            res3_conv1_bias,
            res3_act2_alpha,
            res3_act2_beta,
            res3_conv2_weight,
            res3_conv2_bias,
            9,
        )?;

        Ok(Self {
            snake,
            upsample,
            res1,
            res2,
            res3,
        })
    }

    /// Forward pass with upsampling.
    ///
    /// Input shape: [batch, in_channels, seq_len]
    /// Output shape: [batch, out_channels, seq_len * upsample_rate]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = self.snake.forward(x)?;
        let hidden = self.upsample.forward(&hidden)?;
        let hidden = self.res1.forward(&hidden)?;
        let hidden = self.res2.forward(&hidden)?;
        let hidden = self.res3.forward(&hidden)?;
        Ok(hidden)
    }
}

impl Module for DecoderBlock {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        DecoderBlock::forward(self, x).map_err(|e| candle_core::Error::Msg(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_residual_unit_shape() {
        let device = Device::Cpu;
        let dim = 64;
        let dilation = 3;

        // Create weights manually
        let act1_alpha = Tensor::zeros((dim,), DType::F32, &device).unwrap();
        let act1_beta = Tensor::zeros((dim,), DType::F32, &device).unwrap();
        let conv1_weight = Tensor::randn(0.0f32, 0.1, (dim, dim, 7), &device).unwrap();
        let conv1_bias = Tensor::zeros((dim,), DType::F32, &device).unwrap();
        let act2_alpha = Tensor::zeros((dim,), DType::F32, &device).unwrap();
        let act2_beta = Tensor::zeros((dim,), DType::F32, &device).unwrap();
        let conv2_weight = Tensor::randn(0.0f32, 0.1, (dim, dim, 1), &device).unwrap();
        let conv2_bias = Tensor::zeros((dim,), DType::F32, &device).unwrap();

        let unit = ResidualUnit::from_weights(
            act1_alpha,
            act1_beta,
            conv1_weight,
            conv1_bias,
            act2_alpha,
            act2_beta,
            conv2_weight,
            conv2_bias,
            dilation,
        )
        .unwrap();

        // Input: [batch=1, channels=64, seq=16]
        let input = Tensor::randn(0.0f32, 1.0, (1, dim, 16), &device).unwrap();
        let output = unit.forward(&input).unwrap();

        // Output should have same shape due to residual
        assert_eq!(output.dims(), input.dims());
    }

    #[test]
    fn test_decoder_block_shape() {
        let device = Device::Cpu;
        let in_dim = 128;
        let out_dim = 64;
        let rate = 4;

        // This is a simplified test without real weights
        // Just verify the structure compiles and basic operations work

        // Create minimal weights for decoder block
        let snake_alpha = Tensor::zeros((in_dim,), DType::F32, &device).unwrap();
        let snake_beta = Tensor::zeros((in_dim,), DType::F32, &device).unwrap();
        let upsample_weight =
            Tensor::randn(0.0f32, 0.1, (in_dim, out_dim, rate * 2), &device).unwrap();
        let upsample_bias = Tensor::zeros((out_dim,), DType::F32, &device).unwrap();

        // Create weights for 3 residual units
        let create_res_weights = |dim: usize| {
            (
                Tensor::zeros((dim,), DType::F32, &device).unwrap(), // act1_alpha
                Tensor::zeros((dim,), DType::F32, &device).unwrap(), // act1_beta
                Tensor::randn(0.0f32, 0.1, (dim, dim, 7), &device).unwrap(), // conv1_weight
                Tensor::zeros((dim,), DType::F32, &device).unwrap(), // conv1_bias
                Tensor::zeros((dim,), DType::F32, &device).unwrap(), // act2_alpha
                Tensor::zeros((dim,), DType::F32, &device).unwrap(), // act2_beta
                Tensor::randn(0.0f32, 0.1, (dim, dim, 1), &device).unwrap(), // conv2_weight
                Tensor::zeros((dim,), DType::F32, &device).unwrap(), // conv2_bias
            )
        };

        let (r1_a1a, r1_a1b, r1_c1w, r1_c1b, r1_a2a, r1_a2b, r1_c2w, r1_c2b) =
            create_res_weights(out_dim);
        let (r2_a1a, r2_a1b, r2_c1w, r2_c1b, r2_a2a, r2_a2b, r2_c2w, r2_c2b) =
            create_res_weights(out_dim);
        let (r3_a1a, r3_a1b, r3_c1w, r3_c1b, r3_a2a, r3_a2b, r3_c2w, r3_c2b) =
            create_res_weights(out_dim);

        let block = DecoderBlock::from_weights(
            snake_alpha,
            snake_beta,
            upsample_weight,
            upsample_bias,
            r1_a1a,
            r1_a1b,
            r1_c1w,
            r1_c1b,
            r1_a2a,
            r1_a2b,
            r1_c2w,
            r1_c2b,
            r2_a1a,
            r2_a1b,
            r2_c1w,
            r2_c1b,
            r2_a2a,
            r2_a2b,
            r2_c2w,
            r2_c2b,
            r3_a1a,
            r3_a1b,
            r3_c1w,
            r3_c1b,
            r3_a2a,
            r3_a2b,
            r3_c2w,
            r3_c2b,
            rate,
        )
        .unwrap();

        // Input: [batch=1, in_channels=128, seq=4]
        let input = Tensor::randn(0.0f32, 1.0, (1, in_dim, 4), &device).unwrap();
        let output = block.forward(&input).unwrap();

        // Output is upsampled: input*rate = 4*4 = 16
        // CausalTransConv trims only from right side for exact upsampling
        assert_eq!(output.dims(), &[1, out_dim, 16]);
    }
}
