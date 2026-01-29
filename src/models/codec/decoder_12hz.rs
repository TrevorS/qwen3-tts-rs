//! 12Hz Audio Decoder for Qwen3-TTS
//!
//! Full decoder that converts discrete codec tokens to audio waveforms.
//! Uses the Mimi-based architecture with BigVGAN-style upsampling.

use anyhow::Result;
use candle_core::{DType, IndexOp, Tensor, D};
use std::collections::HashMap;

use super::{CausalConv1d, CausalTransConv1d, ConvNeXtBlock, DecoderBlock, SnakeBeta};

/// Configuration for the 12Hz decoder
#[derive(Debug, Clone)]
pub struct Decoder12HzConfig {
    /// Codebook dimension (512)
    pub codebook_dim: usize,
    /// Latent dimension for transformer (1024)
    pub latent_dim: usize,
    /// Hidden size for transformer (512)
    pub hidden_size: usize,
    /// Number of transformer layers (8)
    pub num_layers: usize,
    /// Number of attention heads (16)
    pub num_heads: usize,
    /// Head dimension (64)
    pub head_dim: usize,
    /// Intermediate size for MLP (1024)
    pub intermediate_size: usize,
    /// Number of quantizers (16)
    pub num_quantizers: usize,
    /// Codebook size (2048)
    pub codebook_size: usize,
    /// Upsampling ratios for pre-upsampling (2, 2)
    pub upsampling_ratios: Vec<usize>,
    /// Decoder dimension (1536)
    pub decoder_dim: usize,
    /// Upsample rates for decoder blocks (8, 5, 4, 3)
    pub upsample_rates: Vec<usize>,
    /// RMS norm epsilon
    pub rms_norm_eps: f64,
    /// RoPE theta
    pub rope_theta: f64,
    /// Layer scale initial value
    pub layer_scale: f64,
}

impl Default for Decoder12HzConfig {
    fn default() -> Self {
        Self {
            codebook_dim: 512,
            latent_dim: 1024,
            hidden_size: 512,
            num_layers: 8,
            num_heads: 16,
            head_dim: 64,
            intermediate_size: 1024,
            num_quantizers: 16,
            codebook_size: 2048,
            upsampling_ratios: vec![2, 2],
            decoder_dim: 1536,
            upsample_rates: vec![8, 5, 4, 3],
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            layer_scale: 0.01,
        }
    }
}

/// Upsample stage with CausalTransConv + ConvNeXtBlock
pub struct UpsampleStage {
    /// Transposed convolution for upsampling
    pub trans_conv: CausalTransConv1d,
    convnext: ConvNeXtBlock,
}

impl UpsampleStage {
    /// Create from weight tensors
    #[allow(clippy::too_many_arguments)]
    pub fn from_weights(
        trans_conv_weight: Tensor,
        trans_conv_bias: Tensor,
        convnext_dwconv_weight: Tensor,
        convnext_dwconv_bias: Tensor,
        convnext_norm_weight: Tensor,
        convnext_norm_bias: Tensor,
        convnext_pwconv1_weight: Tensor,
        convnext_pwconv1_bias: Tensor,
        convnext_pwconv2_weight: Tensor,
        convnext_pwconv2_bias: Tensor,
        convnext_gamma: Tensor,
        stride: usize,
    ) -> Result<Self> {
        let trans_conv =
            CausalTransConv1d::from_weights(trans_conv_weight, Some(trans_conv_bias), stride)?;
        let convnext = ConvNeXtBlock::from_weights(
            convnext_dwconv_weight,
            Some(convnext_dwconv_bias),
            convnext_norm_weight,
            convnext_norm_bias,
            convnext_pwconv1_weight,
            convnext_pwconv1_bias,
            convnext_pwconv2_weight,
            convnext_pwconv2_bias,
            convnext_gamma,
        )?;

        Ok(Self {
            trans_conv,
            convnext,
        })
    }

    /// Forward pass
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = self.trans_conv.forward(x)?;
        self.convnext.forward(&hidden)
    }
}

/// Full 12Hz decoder
pub struct Decoder12Hz {
    config: Decoder12HzConfig,
    /// Codebook embeddings for first quantizer [codebook_size, 256]
    first_codebook: Tensor,
    /// Codebook embeddings for rest quantizers (15 codebooks) [15, codebook_size, 256]
    rest_codebooks: Vec<Tensor>,
    /// Output projection for first quantizer (semantic) [512, 256, 1]
    first_output_proj: Tensor,
    /// Output projection for rest quantizers (acoustic) [512, 256, 1]
    rest_output_proj: Tensor,
    /// Pre-conv: [codebook_dim, latent_dim, 3]
    pre_conv: CausalConv1d,
    /// Input projection for transformer
    input_proj_weight: Tensor,
    input_proj_bias: Tensor,
    /// Transformer layers weights
    transformer_weights: TransformerWeights,
    /// Final norm for transformer
    final_norm_weight: Tensor,
    /// Output projection for transformer
    output_proj_weight: Tensor,
    output_proj_bias: Tensor,
    /// Upsample stages (2 stages)
    upsample_stages: Vec<UpsampleStage>,
    /// Initial decoder conv: [latent_dim, decoder_dim, 7]
    decoder_init_conv: CausalConv1d,
    /// Decoder blocks (4 blocks with rates 8, 5, 4, 3)
    decoder_blocks: Vec<DecoderBlock>,
    /// Final SnakeBeta activation
    final_snake: SnakeBeta,
    /// Final conv: [output_dim, 1, 7]
    final_conv: CausalConv1d,
}

/// Transformer layer weights
struct TransformerLayerWeights {
    input_ln_weight: Tensor,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    attn_layer_scale: Tensor,
    post_ln_weight: Tensor,
    gate_proj_weight: Tensor,
    up_proj_weight: Tensor,
    down_proj_weight: Tensor,
    mlp_layer_scale: Tensor,
}

/// All transformer weights
struct TransformerWeights {
    layers: Vec<TransformerLayerWeights>,
}

/// Fetch a weight tensor by key, returning a helpful error if missing.
fn get_weight(weights: &HashMap<String, Tensor>, key: &str) -> Result<Tensor> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Missing weight: {}", key))
}

impl Decoder12Hz {
    /// Load decoder from safetensors weights
    pub fn from_weights(
        weights: &HashMap<String, Tensor>,
        config: Decoder12HzConfig,
    ) -> Result<Self> {
        // Load codebooks - normalize by cluster_usage as per official implementation
        // embedding = embedding_sum / cluster_usage
        let first_embedding_sum = get_weight(
            weights,
            "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum",
        )?;
        let first_cluster_usage = get_weight(
            weights,
            "decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage",
        )?;
        // Normalize: embedding = embedding_sum / cluster_usage.clamp(min=epsilon).unsqueeze(-1)
        // Per official implementation, clamp cluster_usage to avoid divide-by-zero
        let epsilon = 1e-7f32;
        let first_cluster_usage_clamped = first_cluster_usage.clamp(epsilon, f32::MAX)?;
        let first_codebook =
            first_embedding_sum.broadcast_div(&first_cluster_usage_clamped.unsqueeze(1)?)?;

        let mut rest_codebooks = Vec::with_capacity(15);
        for i in 0..15 {
            let embedding_sum = get_weight(
                weights,
                &format!(
                    "decoder.quantizer.rvq_rest.vq.layers.{}._codebook.embedding_sum",
                    i
                ),
            )?;
            let cluster_usage = get_weight(
                weights,
                &format!(
                    "decoder.quantizer.rvq_rest.vq.layers.{}._codebook.cluster_usage",
                    i
                ),
            )?;
            let cluster_usage_clamped = cluster_usage.clamp(epsilon, f32::MAX)?;
            let cb = embedding_sum.broadcast_div(&cluster_usage_clamped.unsqueeze(1)?)?;
            rest_codebooks.push(cb);
        }

        // Load output projections
        let first_output_proj =
            get_weight(weights, "decoder.quantizer.rvq_first.output_proj.weight")?;
        let rest_output_proj =
            get_weight(weights, "decoder.quantizer.rvq_rest.output_proj.weight")?;

        // Load pre_conv
        let pre_conv = CausalConv1d::from_weights(
            get_weight(weights, "decoder.pre_conv.conv.weight")?,
            Some(get_weight(weights, "decoder.pre_conv.conv.bias")?),
            1,
        )?;

        // Load transformer projections
        let input_proj_weight = get_weight(weights, "decoder.pre_transformer.input_proj.weight")?;
        let input_proj_bias = get_weight(weights, "decoder.pre_transformer.input_proj.bias")?;
        let output_proj_weight = get_weight(weights, "decoder.pre_transformer.output_proj.weight")?;
        let output_proj_bias = get_weight(weights, "decoder.pre_transformer.output_proj.bias")?;

        // Load transformer layer weights
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let p = format!("decoder.pre_transformer.layers.{}", i);
            let layer = TransformerLayerWeights {
                input_ln_weight: get_weight(weights, &format!("{p}.input_layernorm.weight"))?,
                q_proj_weight: get_weight(weights, &format!("{p}.self_attn.q_proj.weight"))?,
                k_proj_weight: get_weight(weights, &format!("{p}.self_attn.k_proj.weight"))?,
                v_proj_weight: get_weight(weights, &format!("{p}.self_attn.v_proj.weight"))?,
                o_proj_weight: get_weight(weights, &format!("{p}.self_attn.o_proj.weight"))?,
                attn_layer_scale: get_weight(weights, &format!("{p}.self_attn_layer_scale.scale"))?,
                post_ln_weight: get_weight(
                    weights,
                    &format!("{p}.post_attention_layernorm.weight"),
                )?,
                gate_proj_weight: get_weight(weights, &format!("{p}.mlp.gate_proj.weight"))?,
                up_proj_weight: get_weight(weights, &format!("{p}.mlp.up_proj.weight"))?,
                down_proj_weight: get_weight(weights, &format!("{p}.mlp.down_proj.weight"))?,
                mlp_layer_scale: get_weight(weights, &format!("{p}.mlp_layer_scale.scale"))?,
            };
            layers.push(layer);
        }
        let transformer_weights = TransformerWeights { layers };

        // Load final norm weight
        let final_norm_weight = get_weight(weights, "decoder.pre_transformer.norm.weight")?;

        // Load upsample stages
        let mut upsample_stages = Vec::with_capacity(config.upsampling_ratios.len());
        for (i, &ratio) in config.upsampling_ratios.iter().enumerate() {
            let p = format!("decoder.upsample.{}", i);
            let stage = UpsampleStage::from_weights(
                get_weight(weights, &format!("{p}.0.conv.weight"))?,
                get_weight(weights, &format!("{p}.0.conv.bias"))?,
                get_weight(weights, &format!("{p}.1.dwconv.conv.weight"))?,
                get_weight(weights, &format!("{p}.1.dwconv.conv.bias"))?,
                get_weight(weights, &format!("{p}.1.norm.weight"))?,
                get_weight(weights, &format!("{p}.1.norm.bias"))?,
                get_weight(weights, &format!("{p}.1.pwconv1.weight"))?,
                get_weight(weights, &format!("{p}.1.pwconv1.bias"))?,
                get_weight(weights, &format!("{p}.1.pwconv2.weight"))?,
                get_weight(weights, &format!("{p}.1.pwconv2.bias"))?,
                get_weight(weights, &format!("{p}.1.gamma"))?,
                ratio,
            )?;
            upsample_stages.push(stage);
        }

        // Load decoder.0 (initial conv)
        let decoder_init_conv = CausalConv1d::from_weights(
            get_weight(weights, "decoder.decoder.0.conv.weight")?,
            Some(get_weight(weights, "decoder.decoder.0.conv.bias")?),
            1,
        )?;

        // Load decoder blocks (1-4)
        let mut decoder_blocks = Vec::with_capacity(config.upsample_rates.len());
        for (i, &rate) in config.upsample_rates.iter().enumerate() {
            let block_idx = i + 1;
            let bp = format!("decoder.decoder.{}.block", block_idx);

            // Helper to load residual unit weights
            #[allow(clippy::type_complexity)]
            let load_res = |unit_idx: usize| -> Result<(
                Tensor,
                Tensor,
                Tensor,
                Tensor,
                Tensor,
                Tensor,
                Tensor,
                Tensor,
            )> {
                let u = format!("{bp}.{unit_idx}");
                Ok((
                    get_weight(weights, &format!("{u}.act1.alpha"))?,
                    get_weight(weights, &format!("{u}.act1.beta"))?,
                    get_weight(weights, &format!("{u}.conv1.conv.weight"))?,
                    get_weight(weights, &format!("{u}.conv1.conv.bias"))?,
                    get_weight(weights, &format!("{u}.act2.alpha"))?,
                    get_weight(weights, &format!("{u}.act2.beta"))?,
                    get_weight(weights, &format!("{u}.conv2.conv.weight"))?,
                    get_weight(weights, &format!("{u}.conv2.conv.bias"))?,
                ))
            };

            let (r1_a1a, r1_a1b, r1_c1w, r1_c1b, r1_a2a, r1_a2b, r1_c2w, r1_c2b) = load_res(2)?;
            let (r2_a1a, r2_a1b, r2_c1w, r2_c1b, r2_a2a, r2_a2b, r2_c2w, r2_c2b) = load_res(3)?;
            let (r3_a1a, r3_a1b, r3_c1w, r3_c1b, r3_a2a, r3_a2b, r3_c2w, r3_c2b) = load_res(4)?;

            let block = DecoderBlock::from_weights(
                get_weight(weights, &format!("{bp}.0.alpha"))?,
                get_weight(weights, &format!("{bp}.0.beta"))?,
                get_weight(weights, &format!("{bp}.1.conv.weight"))?,
                get_weight(weights, &format!("{bp}.1.conv.bias"))?,
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
            )?;
            decoder_blocks.push(block);
        }

        // Load final SnakeBeta (decoder.5)
        let final_snake = SnakeBeta::from_weights(
            get_weight(weights, "decoder.decoder.5.alpha")?,
            get_weight(weights, "decoder.decoder.5.beta")?,
        )?;

        // Load final conv (decoder.6)
        let final_conv = CausalConv1d::from_weights(
            get_weight(weights, "decoder.decoder.6.conv.weight")?,
            Some(get_weight(weights, "decoder.decoder.6.conv.bias")?),
            1,
        )?;

        Ok(Self {
            config,
            first_codebook,
            rest_codebooks,
            first_output_proj,
            rest_output_proj,
            pre_conv,
            input_proj_weight,
            input_proj_bias,
            transformer_weights,
            final_norm_weight,
            output_proj_weight,
            output_proj_bias,
            upsample_stages,
            decoder_init_conv,
            decoder_blocks,
            final_snake,
            final_conv,
        })
    }

    /// Decode codec tokens to audio waveform
    ///
    /// # Arguments
    /// * `codes` - Token indices of shape [batch, num_quantizers, seq_len]
    ///
    /// # Returns
    /// Audio tensor of shape [batch, 1, samples]
    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let device = codes.device();
        let (batch_size, _num_quantizers, seq_len) = codes.dims3()?;

        // 1. Quantizer decode - official Qwen3-TTS architecture:
        //    rvq_first.decode(): embed first codes → first_output_proj
        //    rvq_rest.decode(): sum rest embeds → rest_output_proj
        //    Then SUM the two projected outputs

        // First quantizer (semantic) - single codebook
        let first_codes = codes.i((.., 0, ..))?; // [batch, seq]
        let first_codes_flat = first_codes.flatten_all()?;
        let codebook_size = self.config.codebook_size as i64;
        // Apply modulo to map 3072 vocab → 2048 codebook
        let first_codes_vec: Vec<i64> = first_codes_flat.to_vec1()?;
        let first_codes_mod: Vec<i64> =
            first_codes_vec.iter().map(|&c| c % codebook_size).collect();
        let first_codes_tensor =
            Tensor::from_vec(first_codes_mod, first_codes_flat.dims(), device)?;
        let first_embed = self.first_codebook.index_select(&first_codes_tensor, 0)?;
        let first_embed = first_embed.reshape((batch_size, seq_len, 256))?;

        // Apply first output projection
        let first_embed_t = first_embed.transpose(1, 2)?; // [batch, 256, seq]
        let first_proj = self.conv1d_1x1(&first_embed_t, &self.first_output_proj)?; // [batch, 512, seq]

        // Rest quantizers (acoustic) - 15 codebooks, sum embeddings then project
        let mut rest_embed = Tensor::zeros((batch_size, seq_len, 256), DType::F32, device)?;
        for i in 0..15 {
            let layer_codes = codes.i((.., i + 1, ..))?;
            let layer_codes_flat = layer_codes.flatten_all()?;

            let embed = self.rest_codebooks[i].index_select(&layer_codes_flat, 0)?;
            let embed = embed.reshape((batch_size, seq_len, 256))?;
            rest_embed = (rest_embed + embed)?;
        }
        // Apply rest output projection
        let rest_embed_t = rest_embed.transpose(1, 2)?; // [batch, 256, seq]
        let rest_proj = self.conv1d_1x1(&rest_embed_t, &self.rest_output_proj)?; // [batch, 512, seq]

        // Sum the two projected outputs
        let quantized = (first_proj + rest_proj)?; // [batch, 512, seq]

        // 2. Pre-conv
        // quantized is already [batch, 512, seq] from projections
        let hidden = self.pre_conv.forward(&quantized)?; // [batch, 1024, seq]

        // 3. Pre-transformer
        let hidden = hidden.transpose(1, 2)?; // [batch, seq, 1024]
        let hidden = self.linear_3d(
            &hidden,
            &self.input_proj_weight,
            Some(&self.input_proj_bias),
        )?;

        // Run transformer layers
        let hidden = self.run_transformer(hidden, seq_len)?;

        // Final norm before output projection
        let hidden = self.rms_norm(&hidden, &self.final_norm_weight)?;

        // Output projection
        let hidden = self.linear_3d(
            &hidden,
            &self.output_proj_weight,
            Some(&self.output_proj_bias),
        )?;

        // 4. Transpose for conv: [batch, seq, 1024] -> [batch, 1024, seq]
        let mut hidden = hidden.transpose(1, 2)?;

        // 5. Upsample stages
        for stage in self.upsample_stages.iter() {
            hidden = stage.forward(&hidden)?;
        }

        // 6. Decoder.0 (initial conv)
        hidden = self.decoder_init_conv.forward(&hidden)?;

        // 7. Decoder blocks
        for block in self.decoder_blocks.iter() {
            hidden = block.forward(&hidden)?;
        }

        // 8. Final SnakeBeta
        hidden = self.final_snake.forward(&hidden)?;

        // 9. Final conv
        hidden = self.final_conv.forward(&hidden)?;

        // 10. Clamp to [-1, 1] per official implementation
        // The decoder outputs values in a large range (e.g., [-27, 27])
        // Official code uses: wav.clamp(min=-1, max=1)
        Ok(hidden.clamp(-1.0f32, 1.0f32)?)
    }

    /// 1x1 convolution (pointwise conv)
    /// Input: [batch, in_channels, seq_len]
    /// Weight: [out_channels, in_channels, 1]
    /// Output: [batch, out_channels, seq_len]
    fn conv1d_1x1(&self, x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let weight_2d = weight.squeeze(2)?; // [out_channels, in_channels]
        let (batch, in_ch, seq) = x.dims3()?;
        // Reshape to [batch * seq, in_ch] for matmul
        let x_t = x.transpose(1, 2)?; // [batch, seq, in_ch]
        let x_flat = x_t.reshape((batch * seq, in_ch))?;
        let out_flat = x_flat.matmul(&weight_2d.t()?)?; // [batch * seq, out_ch]
        let out_ch = out_flat.dim(1)?;
        let out = out_flat.reshape((batch, seq, out_ch))?;
        Ok(out.transpose(1, 2)?) // [batch, out_ch, seq]
    }

    /// Linear projection for 3D tensors
    fn linear_3d(&self, x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        let (batch, seq, _features) = x.dims3()?;
        let x_2d = x.reshape((batch * seq, x.dim(2)?))?;
        let out_2d = x_2d.matmul(&weight.t()?)?;
        let out = out_2d.reshape((batch, seq, out_2d.dim(1)?))?;
        match bias {
            Some(b) => Ok(out.broadcast_add(b)?),
            None => Ok(out),
        }
    }

    /// Run transformer layers
    fn run_transformer(&self, mut hidden: Tensor, seq_len: usize) -> Result<Tensor> {
        let device = hidden.device();
        let (batch_size, _, _) = hidden.dims3()?;

        // Build RoPE embeddings
        let positions = Tensor::arange(0u32, seq_len as u32, device)?;
        let inv_freq_vals: Vec<f32> = (0..self.config.head_dim)
            .step_by(2)
            .map(|i| {
                1.0 / (self.config.rope_theta as f32).powf(i as f32 / self.config.head_dim as f32)
            })
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq_vals, (self.config.head_dim / 2,), device)?;

        let positions_f = positions.to_dtype(DType::F32)?;
        let freqs = positions_f.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        let cos = freqs.cos()?.repeat((1, 2))?; // [seq, head_dim]
        let sin = freqs.sin()?.repeat((1, 2))?;
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, seq, head_dim]
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

        // Causal mask - create upper triangular mask with -inf
        let mut mask_data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask_data[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
        let causal_mask = Tensor::from_vec(mask_data, (seq_len, seq_len), device)?
            .unsqueeze(0)?
            .unsqueeze(0)?;

        for (layer_idx, layer) in self.transformer_weights.layers.iter().enumerate() {
            hidden = self.run_layer(
                &hidden,
                layer,
                &cos,
                &sin,
                &causal_mask,
                batch_size,
                seq_len,
                layer_idx,
            )?;
        }

        Ok(hidden)
    }

    /// Run a single transformer layer
    #[allow(clippy::too_many_arguments, unused_variables)]
    fn run_layer(
        &self,
        hidden: &Tensor,
        layer: &TransformerLayerWeights,
        cos: &Tensor,
        sin: &Tensor,
        causal_mask: &Tensor,
        batch_size: usize,
        seq_len: usize,
        layer_idx: usize,
    ) -> Result<Tensor> {
        // RMS Norm
        let normed = self.rms_norm(hidden, &layer.input_ln_weight)?;

        // Self attention
        let q = self.linear_3d(&normed, &layer.q_proj_weight, None)?;
        let k = self.linear_3d(&normed, &layer.k_proj_weight, None)?;
        let v = self.linear_3d(&normed, &layer.v_proj_weight, None)?;

        // Reshape for multi-head attention
        let q = q
            .reshape((
                batch_size,
                seq_len,
                self.config.num_heads,
                self.config.head_dim,
            ))?
            .transpose(1, 2)?; // [batch, heads, seq, head_dim]
        let k = k
            .reshape((
                batch_size,
                seq_len,
                self.config.num_heads,
                self.config.head_dim,
            ))?
            .transpose(1, 2)?;
        let v = v
            .reshape((
                batch_size,
                seq_len,
                self.config.num_heads,
                self.config.head_dim,
            ))?
            .transpose(1, 2)?;

        // Apply RoPE
        let q = self.apply_rope(&q, cos, sin)?;
        let k = self.apply_rope(&k, cos, sin)?;

        // Attention
        let scale = (self.config.head_dim as f64).powf(-0.5);
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let attn = q.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)?;
        let attn = (attn * scale)?;
        let attn = attn.broadcast_add(causal_mask)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let attn_out = attn.matmul(&v)?;

        // Reshape back
        let attn_out = attn_out.transpose(1, 2)?.reshape((
            batch_size,
            seq_len,
            self.config.num_heads * self.config.head_dim,
        ))?;

        // Output projection
        let attn_out = self.linear_3d(&attn_out, &layer.o_proj_weight, None)?;

        // Layer scale and residual
        let attn_out = attn_out.broadcast_mul(&layer.attn_layer_scale)?;
        let hidden = (hidden + attn_out)?;

        // MLP
        let normed = self.rms_norm(&hidden, &layer.post_ln_weight)?;
        let gate = self.linear_3d(&normed, &layer.gate_proj_weight, None)?;
        let up = self.linear_3d(&normed, &layer.up_proj_weight, None)?;
        let mlp_out = candle_nn::ops::silu(&gate)?.mul(&up)?;
        let mlp_out = self.linear_3d(&mlp_out, &layer.down_proj_weight, None)?;

        // Layer scale and residual
        let mlp_out = mlp_out.broadcast_mul(&layer.mlp_layer_scale)?;
        let hidden = (hidden + mlp_out)?;

        Ok(hidden)
    }

    /// RMS normalization
    fn rms_norm(&self, x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let x_normed = x.broadcast_div(&(variance + self.config.rms_norm_eps)?.sqrt()?)?;
        Ok(x_normed.broadcast_mul(weight)?)
    }

    /// Apply rotary position embedding
    fn apply_rope(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let x1 = x.narrow(D::Minus1, 0, self.config.head_dim / 2)?;
        let x2 = x.narrow(
            D::Minus1,
            self.config.head_dim / 2,
            self.config.head_dim / 2,
        )?;
        let rotated = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
        Ok((x.broadcast_mul(cos)? + rotated.broadcast_mul(sin)?)?)
    }

    /// Get total upsampling factor
    pub fn total_upsample(&self) -> usize {
        let pre_upsample: usize = self.config.upsampling_ratios.iter().product();
        let decoder_upsample: usize = self.config.upsample_rates.iter().product();
        pre_upsample * decoder_upsample
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = Decoder12HzConfig::default();
        assert_eq!(config.codebook_dim, 512);
        assert_eq!(config.num_quantizers, 16);
        assert_eq!(config.upsample_rates, vec![8, 5, 4, 3]);
    }

    #[test]
    fn test_total_upsample() {
        let config = Decoder12HzConfig::default();
        // pre: 2 * 2 = 4
        // decoder: 8 * 5 * 4 * 3 = 480
        // total: 4 * 480 = 1920
        let pre: usize = config.upsampling_ratios.iter().product();
        let dec: usize = config.upsample_rates.iter().product();
        assert_eq!(pre * dec, 1920);
    }
}
