//! Audio codec decoder
//!
//! Converts discrete codec tokens back to audio waveforms.

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{conv1d, conv_transpose1d, linear, rms_norm, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Linear, RmsNorm, VarBuilder};

use super::quantizer::ResidualVectorQuantizer;

/// Codec decoder configuration
#[derive(Debug, Clone)]
pub struct DecoderConfig {
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Upsampling ratios
    pub upsample_ratios: Vec<usize>,
    /// Number of quantizers
    pub num_quantizers: usize,
    /// Codebook dimension
    pub codebook_dim: usize,
    /// Codebook size
    pub codebook_size: usize,
    /// Final output channels (1 for mono audio)
    pub out_channels: usize,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            num_layers: 8,
            num_heads: 16,
            upsample_ratios: vec![4, 5, 8, 3], // Total: 480x upsampling
            num_quantizers: 16,
            codebook_dim: 256,
            codebook_size: 2048,
            out_channels: 1,
        }
    }
}

/// Upsampling block with transposed convolution
pub struct UpsampleBlock {
    conv: ConvTranspose1d,
    activation: bool,
}

impl UpsampleBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let padding = (kernel_size - stride) / 2;
        let config = ConvTranspose1dConfig {
            stride,
            padding,
            ..Default::default()
        };

        Ok(Self {
            conv: conv_transpose1d(in_channels, out_channels, kernel_size, config, vb.pp("conv"))?,
            activation: true,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        if self.activation {
            // LeakyReLU with negative_slope=0.1
            let mask = x.ge(&Tensor::zeros(x.shape(), x.dtype(), x.device())?)?;
            let positive = x.clone();
            let negative = (x * 0.1)?;
            Ok(mask.where_cond(&positive, &negative)?)
        } else {
            Ok(x)
        }
    }
}

/// Residual block for decoder
pub struct ResidualBlock {
    conv1: Conv1d,
    conv2: Conv1d,
    norm1: RmsNorm,
    norm2: RmsNorm,
}

impl ResidualBlock {
    pub fn new(channels: usize, kernel_size: usize, vb: VarBuilder) -> Result<Self> {
        let conv_config = Conv1dConfig {
            padding: kernel_size / 2,
            ..Default::default()
        };

        Ok(Self {
            conv1: conv1d(channels, channels, kernel_size, conv_config, vb.pp("conv1"))?,
            conv2: conv1d(channels, channels, kernel_size, conv_config, vb.pp("conv2"))?,
            norm1: rms_norm(channels, 1e-6, vb.pp("norm1"))?,
            norm2: rms_norm(channels, 1e-6, vb.pp("norm2"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;

        // Transpose for norm (norm expects [batch, seq, channels])
        let x = x.transpose(1, 2)?;
        let x = self.norm1.forward(&x)?.transpose(1, 2)?;
        let x = candle_nn::ops::silu(&self.conv1.forward(&x)?)?;

        let x = x.transpose(1, 2)?;
        let x = self.norm2.forward(&x)?.transpose(1, 2)?;
        let x = self.conv2.forward(&x)?;

        Ok((x + residual)?)
    }
}

/// Simple self-attention for decoder transformer
pub struct DecoderAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl DecoderAttention {
    pub fn new(hidden_size: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = hidden_size / num_heads;

        Ok(Self {
            q_proj: linear(hidden_size, hidden_size, vb.pp("q_proj"))?,
            k_proj: linear(hidden_size, hidden_size, vb.pp("k_proj"))?,
            v_proj: linear(hidden_size, hidden_size, vb.pp("v_proj"))?,
            o_proj: linear(hidden_size, hidden_size, vb.pp("o_proj"))?,
            num_heads,
            head_dim,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, hidden) = x.dims3()?;

        let q = self.q_proj.forward(x)?
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self.k_proj.forward(x)?
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self.v_proj.forward(x)?
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;

        let out = out.transpose(1, 2)?
            .reshape((batch, seq_len, hidden))?;

        Ok(self.o_proj.forward(&out)?)
    }
}

/// Decoder transformer layer
pub struct DecoderTransformerLayer {
    self_attn: DecoderAttention,
    mlp_fc1: Linear,
    mlp_fc2: Linear,
    norm1: RmsNorm,
    norm2: RmsNorm,
}

impl DecoderTransformerLayer {
    pub fn new(hidden_size: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let intermediate_size = hidden_size * 4;

        Ok(Self {
            self_attn: DecoderAttention::new(hidden_size, num_heads, vb.pp("self_attn"))?,
            mlp_fc1: linear(hidden_size, intermediate_size, vb.pp("mlp.fc1"))?,
            mlp_fc2: linear(intermediate_size, hidden_size, vb.pp("mlp.fc2"))?,
            norm1: rms_norm(hidden_size, 1e-6, vb.pp("norm1"))?,
            norm2: rms_norm(hidden_size, 1e-6, vb.pp("norm2"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Self-attention with residual
        let residual = x;
        let x = self.norm1.forward(x)?;
        let x = self.self_attn.forward(&x)?;
        let x = (x + residual)?;

        // MLP with residual
        let residual = &x;
        let x = self.norm2.forward(&x)?;
        let x = candle_nn::ops::silu(&self.mlp_fc1.forward(&x)?)?;
        let x = self.mlp_fc2.forward(&x)?;
        let x = (x + residual)?;

        Ok(x)
    }
}

/// Main codec decoder
pub struct CodecDecoder {
    config: DecoderConfig,
    /// Vector quantizer for codebook lookups
    quantizer: ResidualVectorQuantizer,
    /// Input projection from codebook embeddings
    input_proj: Linear,
    /// Pre-transformer for processing quantized embeddings
    pre_transformer: Vec<DecoderTransformerLayer>,
    /// Pre-transformer output norm
    pre_norm: RmsNorm,
    /// Upsampling blocks
    upsample_blocks: Vec<UpsampleBlock>,
    /// Residual blocks between upsampling
    residual_blocks: Vec<Vec<ResidualBlock>>,
    /// Final convolution to audio
    final_conv: Conv1d,
    device: Device,
}

impl CodecDecoder {
    /// Create new codec decoder
    pub fn new(config: DecoderConfig, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();

        // Create quantizer
        let quantizer = ResidualVectorQuantizer::new(
            config.num_quantizers,
            config.codebook_size,
            config.codebook_dim,
            vb.pp("quantizer"),
        )?;

        // Input projection
        let input_proj = linear(
            config.codebook_dim * config.num_quantizers,
            config.hidden_size,
            vb.pp("input_proj"),
        )?;

        // Pre-transformer layers
        let mut pre_transformer = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            pre_transformer.push(DecoderTransformerLayer::new(
                config.hidden_size,
                config.num_heads,
                vb.pp(format!("pre_transformer.{}", i)),
            )?);
        }
        let pre_norm = rms_norm(config.hidden_size, 1e-6, vb.pp("pre_norm"))?;

        // Upsampling blocks
        let mut upsample_blocks = Vec::new();
        let mut residual_blocks = Vec::new();
        let mut channels = config.hidden_size;

        for (i, &ratio) in config.upsample_ratios.iter().enumerate() {
            let out_channels = channels / 2;
            upsample_blocks.push(UpsampleBlock::new(
                channels,
                out_channels,
                ratio * 2,
                ratio,
                vb.pp(format!("upsample.{}", i)),
            )?);

            // Residual blocks after each upsample
            let mut res_blocks = Vec::new();
            for j in 0..3 {
                res_blocks.push(ResidualBlock::new(
                    out_channels,
                    7,
                    vb.pp(format!("residual.{}.{}", i, j)),
                )?);
            }
            residual_blocks.push(res_blocks);
            channels = out_channels;
        }

        // Final convolution
        let final_conv = conv1d(
            channels,
            config.out_channels,
            7,
            Conv1dConfig { padding: 3, ..Default::default() },
            vb.pp("final_conv"),
        )?;

        Ok(Self {
            config,
            quantizer,
            input_proj,
            pre_transformer,
            pre_norm,
            upsample_blocks,
            residual_blocks,
            final_conv,
            device,
        })
    }

    /// Decode codec tokens to audio
    ///
    /// # Arguments
    /// * `tokens` - Token indices of shape [batch, num_quantizers, seq_len]
    pub fn decode(&self, tokens: &Tensor) -> Result<Tensor> {
        // Look up codebook embeddings
        let embeddings = self.quantizer.decode(tokens)?;

        // Flatten quantizer dimension: [batch, seq, num_q * dim] -> [batch, seq, hidden]
        let (batch, seq_len, num_q, dim) = embeddings.dims4()?;
        let embeddings = embeddings.reshape((batch, seq_len, num_q * dim))?;

        // Project to hidden size
        let mut x = self.input_proj.forward(&embeddings)?;

        // Pre-transformer
        for layer in &self.pre_transformer {
            x = layer.forward(&x)?;
        }
        x = self.pre_norm.forward(&x)?;

        // Transpose for conv: [batch, seq, hidden] -> [batch, hidden, seq]
        let mut x = x.transpose(1, 2)?;

        // Upsampling with residual blocks
        for (i, upsample) in self.upsample_blocks.iter().enumerate() {
            x = upsample.forward(&x)?;
            for res_block in &self.residual_blocks[i] {
                x = res_block.forward(&x)?;
            }
        }

        // Final convolution to audio
        let audio = self.final_conv.forward(&x)?;

        // Squeeze channel dimension for mono: [batch, 1, samples] -> [batch, samples]
        Ok(audio.squeeze(1)?)
    }
}
