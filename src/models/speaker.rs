//! Speaker encoder (ECAPA-TDNN) for voice cloning
//!
//! Full ECAPA-TDNN architecture matching the Python `Qwen3TTSSpeakerEncoder`:
//!
//! - **TimeDelayNetBlock**: Conv1d (reflect-padded "same") + ReLU
//! - **Res2NetBlock**: Scale-8 split with cascaded TDNNs
//! - **SqueezeExcitationBlock**: Channel attention via Conv1d
//! - **SqueezeExcitationRes2NetBlock**: TDNN1 → Res2Net → TDNN2 → SE → residual
//! - **AttentiveStatisticsPooling**: Attention-weighted mean/std pooling
//! - **Multi-layer Feature Aggregation** (MFA): cat SE-Res2Net outputs → TDNN
//!
//! Weight prefix: `speaker_encoder.*`

use anyhow::Result;
use candle_core::{Device, Module, Tensor, D};
use candle_nn::{conv1d, Conv1d, Conv1dConfig, VarBuilder};

use crate::audio::{AudioBuffer, MelSpectrogram};
use crate::models::config::SpeakerEncoderConfig;

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Apply 1D reflect padding to a `[B, C, T]` tensor along the time dimension.
///
/// Mirrors signal values at boundaries, matching PyTorch `padding_mode="reflect"`.
fn reflect_pad_1d(x: &Tensor, pad_left: usize, pad_right: usize) -> Result<Tensor> {
    if pad_left == 0 && pad_right == 0 {
        return Ok(x.clone());
    }
    // index_select requires contiguous tensors
    let x = &x.contiguous()?;
    let (_b, _c, t) = x.dims3()?;

    let mut indices = Vec::with_capacity(pad_left + t + pad_right);

    // Left reflection: mirror from position 1 outward
    for i in (1..=pad_left).rev() {
        indices.push(i as i64);
    }
    // Original signal
    for i in 0..t {
        indices.push(i as i64);
    }
    // Right reflection: mirror from position t-2 inward
    for i in 0..pad_right {
        indices.push((t - 2 - i) as i64);
    }

    let idx = Tensor::from_vec(indices, (pad_left + t + pad_right,), x.device())?;
    Ok(x.index_select(&idx, 2)?)
}

fn relu(x: &Tensor) -> Result<Tensor> {
    Ok(x.maximum(&x.zeros_like()?)?)
}

fn sigmoid(x: &Tensor) -> Result<Tensor> {
    let neg_x = x.neg()?;
    let exp_neg_x = neg_x.exp()?;
    Ok((exp_neg_x + 1.0)?.recip()?)
}

// ── Conv1d with reflect padding ─────────────────────────────────────────────

/// Conv1d with "same" output length via reflect padding.
///
/// Matches PyTorch's `Conv1d(padding="same", padding_mode="reflect")`.
struct ReflectPadConv1d {
    conv: Conv1d,
    pad_left: usize,
    pad_right: usize,
}

impl ReflectPadConv1d {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let total_pad = dilation * (kernel_size - 1);
        let pad_left = total_pad / 2;
        let pad_right = total_pad - pad_left;

        let config = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation,
            groups: 1,
            ..Default::default()
        };

        let conv = conv1d(in_channels, out_channels, kernel_size, config, vb)?;

        Ok(Self {
            conv,
            pad_left,
            pad_right,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let padded = reflect_pad_1d(x, self.pad_left, self.pad_right)?;
        Ok(self.conv.forward(&padded)?)
    }
}

// ── Building blocks ─────────────────────────────────────────────────────────

/// Time-delay neural network block: Conv1d (reflect-padded) + ReLU.
///
/// Weight keys: `conv.weight`, `conv.bias`
struct TimeDelayNetBlock {
    conv: ReflectPadConv1d,
}

impl TimeDelayNetBlock {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            conv: ReflectPadConv1d::new(
                in_channels,
                out_channels,
                kernel_size,
                dilation,
                vb.pp("conv"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        relu(&self.conv.forward(x)?)
    }
}

/// Res2Net block with cascaded TDNNs.
///
/// Splits input channels into `scale` groups. The first group passes through
/// unchanged; subsequent groups are processed by a TDNN whose input is the
/// current chunk added to the previous group's output.
///
/// Weight keys: `blocks.{i}.conv.weight/bias`
struct Res2NetBlock {
    blocks: Vec<TimeDelayNetBlock>,
    scale: usize,
    chunk_size: usize,
}

impl Res2NetBlock {
    fn new(
        channels: usize,
        kernel_size: usize,
        dilation: usize,
        scale: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let chunk_size = channels / scale;
        let mut blocks = Vec::with_capacity(scale - 1);
        for i in 0..(scale - 1) {
            blocks.push(TimeDelayNetBlock::new(
                chunk_size,
                chunk_size,
                kernel_size,
                dilation,
                vb.pp(format!("blocks.{}", i)),
            )?);
        }
        Ok(Self {
            blocks,
            scale,
            chunk_size,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut outputs = Vec::with_capacity(self.scale);

        // First chunk passes through unchanged
        outputs.push(x.narrow(1, 0, self.chunk_size)?);

        for (i, block) in self.blocks.iter().enumerate() {
            let chunk = x.narrow(1, (i + 1) * self.chunk_size, self.chunk_size)?;
            let input = if i == 0 {
                chunk
            } else {
                (chunk + outputs.last().unwrap())?
            };
            outputs.push(block.forward(&input)?);
        }

        Ok(Tensor::cat(&outputs, 1)?)
    }
}

/// Squeeze-and-excitation block for channel attention.
///
/// Global average pool → Conv1d + ReLU → Conv1d + Sigmoid → scale.
///
/// Weight keys: `conv1.weight/bias`, `conv2.weight/bias`
struct SqueezeExcitationBlock {
    conv1: Conv1d,
    conv2: Conv1d,
}

impl SqueezeExcitationBlock {
    fn new(channels: usize, se_channels: usize, vb: VarBuilder) -> Result<Self> {
        let config = Conv1dConfig::default();
        Ok(Self {
            conv1: conv1d(channels, se_channels, 1, config, vb.pp("conv1"))?,
            conv2: conv1d(se_channels, channels, 1, config, vb.pp("conv2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Global average pool: [B, C, T] → [B, C, 1]
        let s = x.mean(D::Minus1)?.unsqueeze(D::Minus1)?;
        let s = relu(&self.conv1.forward(&s)?)?;
        let s = sigmoid(&self.conv2.forward(&s)?)?;
        Ok(x.broadcast_mul(&s)?)
    }
}

/// SE-Res2Net block: TDNN1 → Res2Net → TDNN2 → SE → residual add.
///
/// Weight keys: `tdnn1.*`, `res2net_block.*`, `tdnn2.*`, `se_block.*`
struct SqueezeExcitationRes2NetBlock {
    tdnn1: TimeDelayNetBlock,
    res2net_block: Res2NetBlock,
    tdnn2: TimeDelayNetBlock,
    se_block: SqueezeExcitationBlock,
}

impl SqueezeExcitationRes2NetBlock {
    fn new(
        channels: usize,
        kernel_size: usize,
        dilation: usize,
        scale: usize,
        se_channels: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            tdnn1: TimeDelayNetBlock::new(channels, channels, 1, 1, vb.pp("tdnn1"))?,
            res2net_block: Res2NetBlock::new(
                channels,
                kernel_size,
                dilation,
                scale,
                vb.pp("res2net_block"),
            )?,
            tdnn2: TimeDelayNetBlock::new(channels, channels, 1, 1, vb.pp("tdnn2"))?,
            se_block: SqueezeExcitationBlock::new(channels, se_channels, vb.pp("se_block"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let out = self.tdnn1.forward(x)?;
        let out = self.res2net_block.forward(&out)?;
        let out = self.tdnn2.forward(&out)?;
        let out = self.se_block.forward(&out)?;
        Ok((out + residual)?)
    }
}

/// Attentive statistics pooling.
///
/// Computes attention-weighted mean and standard deviation over time.
///
/// Weight keys: `tdnn.conv.weight/bias`, `conv.weight/bias`
struct AttentiveStatisticsPooling {
    tdnn: TimeDelayNetBlock,
    conv: Conv1d,
}

impl AttentiveStatisticsPooling {
    fn new(channels: usize, attention_channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            tdnn: TimeDelayNetBlock::new(channels * 3, attention_channels, 1, 1, vb.pp("tdnn"))?,
            conv: conv1d(
                attention_channels,
                channels,
                1,
                Conv1dConfig::default(),
                vb.pp("conv"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, C, T]
        let (b, c, t) = x.dims3()?;

        // Global statistics expanded to sequence length
        let mean = x.mean(D::Minus1)?.unsqueeze(D::Minus1)?; // [B, C, 1]
        let diff = x.broadcast_sub(&mean)?;
        let var = diff.sqr()?.mean(D::Minus1)?.unsqueeze(D::Minus1)?;
        let std = (var + 1e-5)?.sqrt()?; // [B, C, 1]

        let mean_exp = mean.broadcast_as((b, c, t))?;
        let std_exp = std.broadcast_as((b, c, t))?;

        // Concatenate [x, mean, std] along channel dim → [B, 3C, T]
        let attn_in = Tensor::cat(&[x, &mean_exp, &std_exp], 1)?;

        // Attention: TDNN(3C→attn_ch, includes ReLU) → Tanh → Conv(attn_ch→C) → Softmax
        let attn = self.tdnn.forward(&attn_in)?;
        let attn = attn.tanh()?;
        let attn = self.conv.forward(&attn)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?; // softmax over T

        // Weighted mean
        let w_mean = x
            .broadcast_mul(&attn)?
            .sum(D::Minus1)?
            .unsqueeze(D::Minus1)?; // [B, C, 1]

        // Weighted std
        let w_diff = x.broadcast_sub(&w_mean)?;
        let w_var = w_diff
            .sqr()?
            .broadcast_mul(&attn)?
            .sum(D::Minus1)?
            .unsqueeze(D::Minus1)?;
        let w_std = (w_var + 1e-5)?.sqrt()?; // [B, C, 1]

        // Output: cat([mean, std]) → [B, 2C, 1]
        Ok(Tensor::cat(&[&w_mean, &w_std], 1)?)
    }
}

// ── Main encoder ────────────────────────────────────────────────────────────

/// Full ECAPA-TDNN speaker encoder.
///
/// Architecture:
/// ```text
/// blocks[0]:   TDNN(mel_dim → ch[0], k=ks[0], d=dl[0])
/// blocks[1-3]: SE-Res2Net(ch[i], k=ks[i], d=dl[i])
/// MFA:         cat(SE-Res2Net outputs) → mfa TDNN(→ ch[4])
/// ASP:         Attentive statistics pooling
/// FC:          Conv1d(2 × ch[4] → enc_dim, k=1)
/// ```
///
/// Weight prefix: `speaker_encoder.*`
pub struct SpeakerEncoder {
    mel_extractor: MelSpectrogram,
    initial_tdnn: TimeDelayNetBlock,
    se_res2net_blocks: Vec<SqueezeExcitationRes2NetBlock>,
    mfa_tdnn: TimeDelayNetBlock,
    asp: AttentiveStatisticsPooling,
    fc: Conv1d,
    device: Device,
}

impl SpeakerEncoder {
    /// Create from VarBuilder.
    ///
    /// The `vb` should already be scoped to `speaker_encoder` (i.e., the caller
    /// passes `vb.pp("speaker_encoder")`).
    pub fn new(config: SpeakerEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();

        let mel_config = MelSpectrogram::speaker_encoder();
        let mel_extractor = MelSpectrogram::new(mel_config);

        // blocks[0]: initial TDNN (mel_dim → enc_channels[0])
        let initial_tdnn = TimeDelayNetBlock::new(
            config.mel_dim,
            config.enc_channels[0],
            config.enc_kernel_sizes[0],
            config.enc_dilations[0],
            vb.pp("blocks.0"),
        )?;

        // blocks[1-3]: SE-Res2Net blocks
        let mut se_res2net_blocks = Vec::with_capacity(3);
        for i in 1..4 {
            se_res2net_blocks.push(SqueezeExcitationRes2NetBlock::new(
                config.enc_channels[i],
                config.enc_kernel_sizes[i],
                config.enc_dilations[i],
                config.enc_res2net_scale,
                config.enc_se_channels,
                vb.pp(format!("blocks.{}", i)),
            )?);
        }

        // MFA: concatenate SE-Res2Net outputs → TDNN projection
        // Weight keys: `mfa.conv.weight`, `mfa.conv.bias`
        let mfa_in_channels: usize = config.enc_channels[1..4].iter().sum();
        let mfa_tdnn = TimeDelayNetBlock::new(
            mfa_in_channels,
            config.enc_channels[4],
            config.enc_kernel_sizes[4],
            config.enc_dilations[4],
            vb.pp("mfa"),
        )?;

        // ASP: attentive statistics pooling
        let asp = AttentiveStatisticsPooling::new(
            config.enc_channels[4],
            config.enc_attention_channels,
            vb.pp("asp"),
        )?;

        // FC: Conv1d(2 * enc_channels[4] → enc_dim, k=1)
        let fc = conv1d(
            config.enc_channels[4] * 2,
            config.enc_dim,
            1,
            Conv1dConfig::default(),
            vb.pp("fc"),
        )?;

        Ok(Self {
            mel_extractor,
            initial_tdnn,
            se_res2net_blocks,
            mfa_tdnn,
            asp,
            fc,
            device,
        })
    }

    /// Extract a speaker embedding from reference audio.
    ///
    /// Returns an embedding of shape `[enc_dim]` (default 1024).
    pub fn encode(&self, audio: &AudioBuffer) -> Result<Tensor> {
        let mel = self
            .mel_extractor
            .compute_for_speaker_encoder(&audio.samples, &self.device)?;
        let mel = mel.unsqueeze(0)?; // [1, n_mels, T]
        let embed = self.forward(&mel)?; // [1, enc_dim]
        Ok(embed.squeeze(0)?) // [enc_dim]
    }

    /// Forward pass on a batched mel spectrogram `[B, n_mels, T]`.
    ///
    /// Returns embeddings of shape `[B, enc_dim]` (unnormalized, norm ≈ 10).
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        // blocks[0]: initial TDNN
        let x = self.initial_tdnn.forward(mel)?;

        // blocks[1-3]: SE-Res2Net, collecting outputs for MFA
        let mut se_outputs = Vec::with_capacity(3);
        let mut h = x;
        for block in &self.se_res2net_blocks {
            h = block.forward(&h)?;
            se_outputs.push(h.clone());
        }

        // MFA: concatenate SE-Res2Net outputs along channel dim
        let mfa_input = Tensor::cat(&se_outputs, 1)?;

        // MFA: project concatenated SE-Res2Net outputs
        let h = self.mfa_tdnn.forward(&mfa_input)?;

        // ASP: attentive statistics pooling → [B, 2C, 1]
        let pooled = self.asp.forward(&h)?;

        // FC: project to embedding dimension → [B, enc_dim, 1]
        let embed = self.fc.forward(&pooled)?;
        let embed = embed.squeeze(D::Minus1)?; // [B, enc_dim]

        // Return raw embedding (no L2 normalization — the model was trained
        // with unnormalized speaker embeddings, norm ≈ 10)
        Ok(embed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use candle_nn::VarMap;

    fn create_mock_vb(device: &Device) -> VarBuilder<'static> {
        let varmap = VarMap::new();
        VarBuilder::from_varmap(&varmap, DType::F32, device)
    }

    #[test]
    fn test_reflect_pad_1d_no_pad() {
        let device = Device::Cpu;
        let x = Tensor::arange(0.0f32, 5.0, &device)
            .unwrap()
            .reshape((1, 1, 5))
            .unwrap();
        let padded = reflect_pad_1d(&x, 0, 0).unwrap();
        assert_eq!(padded.dims(), &[1, 1, 5]);
    }

    #[test]
    fn test_reflect_pad_1d_left() {
        let device = Device::Cpu;
        let x = Tensor::new(&[[[0.0f32, 1.0, 2.0, 3.0, 4.0]]], &device).unwrap();
        let padded = reflect_pad_1d(&x, 2, 0).unwrap();
        let vals: Vec<f32> = padded.flatten_all().unwrap().to_vec1().unwrap();
        // Expected: [2, 1, 0, 1, 2, 3, 4]
        assert_eq!(vals, vec![2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_reflect_pad_1d_right() {
        let device = Device::Cpu;
        let x = Tensor::new(&[[[0.0f32, 1.0, 2.0, 3.0, 4.0]]], &device).unwrap();
        let padded = reflect_pad_1d(&x, 0, 2).unwrap();
        let vals: Vec<f32> = padded.flatten_all().unwrap().to_vec1().unwrap();
        // Expected: [0, 1, 2, 3, 4, 3, 2]
        assert_eq!(vals, vec![0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);
    }

    #[test]
    fn test_reflect_pad_1d_both() {
        let device = Device::Cpu;
        let x = Tensor::new(&[[[0.0f32, 1.0, 2.0, 3.0, 4.0]]], &device).unwrap();
        let padded = reflect_pad_1d(&x, 2, 2).unwrap();
        let vals: Vec<f32> = padded.flatten_all().unwrap().to_vec1().unwrap();
        // Expected: [2, 1, 0, 1, 2, 3, 4, 3, 2]
        assert_eq!(vals, vec![2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);
    }

    #[test]
    fn test_relu() {
        let device = Device::Cpu;
        let x = Tensor::new(&[-1.0f32, 0.0, 1.0, 2.0], &device).unwrap();
        let result = relu(&x).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();
        assert_eq!(vals, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sigmoid() {
        let device = Device::Cpu;
        let x = Tensor::new(&[0.0f32], &device).unwrap();
        let result = sigmoid(&x).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();
        assert!((vals[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_speaker_encoder_construction() {
        let device = Device::Cpu;
        let vb = create_mock_vb(&device);
        let config = SpeakerEncoderConfig::default();
        let encoder = SpeakerEncoder::new(config, vb);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_speaker_encoder_forward_shape() {
        let device = Device::Cpu;
        let vb = create_mock_vb(&device);
        let config = SpeakerEncoderConfig::default();
        let encoder = SpeakerEncoder::new(config, vb).unwrap();

        // Input: [1, 128, 100] (batch=1, 128 mel bands, 100 time frames)
        let mel = Tensor::randn(0.0f32, 1.0, (1, 128, 100), &device).unwrap();
        let embed = encoder.forward(&mel).unwrap();

        assert_eq!(embed.dims(), &[1, 1024]);
    }
}
