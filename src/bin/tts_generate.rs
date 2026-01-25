//! Proper TTS generation with correct prefill construction
//!
//! This uses the dual-stream embedding fusion (text + codec) as in the official implementation.

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor, D};
use clap::Parser;
use std::collections::HashMap;
use std::path::Path;

use qwen3_tts::generation::{
    self, apply_token_suppression, build_prefill_embeddings, set_seed, TtsGenerationConfig,
    TtsTokenIds,
};
use qwen3_tts::models::codec::Decoder12Hz;
use qwen3_tts::AudioBuffer;

#[derive(Parser, Debug)]
#[command(author, version, about = "Generate TTS audio with proper prefill")]
struct Args {
    /// Text to synthesize
    #[arg(short, long, default_value = "Hello")]
    text: String,

    /// Random seed
    #[arg(short, long, default_value_t = 42)]
    seed: u64,

    /// Number of frames to generate
    #[arg(short, long, default_value_t = 50)]
    frames: usize,

    /// Sampling temperature
    #[arg(long, default_value_t = 0.7)]
    temperature: f64,

    /// Top-k sampling
    #[arg(long, default_value_t = 50)]
    top_k: usize,

    /// Model directory
    #[arg(short, long, default_value = "test_data/model")]
    model_dir: String,

    /// Output WAV file
    #[arg(short, long, default_value = "test_data/tts_output.wav")]
    output: String,
}

/// Simple text to token mapping (raw text tokens only)
fn text_to_raw_ids(text: &str) -> Vec<u32> {
    match text {
        "Hello" => vec![9707],
        "Hello world" => vec![9707, 1917],
        "Hello, this is a" => vec![9707, 11, 419, 374, 264],
        "Hello, this is a test" => vec![9707, 11, 419, 374, 264, 1273],
        _ => {
            eprintln!("Warning: Unknown text '{}', using 'Hello'", text);
            vec![9707]
        }
    }
}

/// Get raw text token IDs (build_prefill_embeddings adds ChatML structure)
fn text_to_ids(text: &str) -> Vec<u32> {
    text_to_raw_ids(text)
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("=== TTS Generation (Proper Prefill) ===");
    println!("Text: {}", args.text);
    println!("Seed: {}", args.seed);
    println!("Frames: {}", args.frames);

    set_seed(args.seed);
    generation::reset_rng();

    let device = Device::Cpu;

    // Load weights
    println!("\nLoading model weights...");
    let model_path = Path::new(&args.model_dir).join("model.safetensors");
    let weights: HashMap<String, Tensor> = candle_core::safetensors::load(&model_path, &device)?;
    let weights: HashMap<String, Tensor> = weights
        .into_iter()
        .map(|(k, v)| {
            let v = if v.dtype() == DType::BF16 {
                v.to_dtype(DType::F32).unwrap()
            } else {
                v
            };
            (k, v)
        })
        .collect();

    let decoder_path = Path::new(&args.model_dir).join("speech_tokenizer/model.safetensors");
    let decoder_weights: HashMap<String, Tensor> =
        candle_core::safetensors::load(&decoder_path, &device)?;
    let decoder_weights: HashMap<String, Tensor> = decoder_weights
        .into_iter()
        .map(|(k, v)| {
            let v = if v.dtype() == DType::BF16 {
                v.to_dtype(DType::F32).unwrap()
            } else {
                v
            };
            (k, v)
        })
        .collect();

    // Get key tensors
    let text_embedding = weights
        .get("talker.model.text_embedding.weight")
        .ok_or_else(|| anyhow::anyhow!("Missing text_embedding"))?;
    let codec_embedding = weights
        .get("talker.model.codec_embedding.weight")
        .ok_or_else(|| anyhow::anyhow!("Missing codec_embedding"))?;
    let codec_head = weights
        .get("talker.codec_head.weight")
        .ok_or_else(|| anyhow::anyhow!("Missing codec_head"))?;

    println!("  text_embedding: {:?}", text_embedding.shape());
    println!("  codec_embedding: {:?}", codec_embedding.shape());
    println!("  codec_head: {:?}", codec_head.shape());

    // Determine hidden size from codec_embedding
    let hidden_size = codec_embedding.dim(1)?;
    let vocab_size = codec_embedding.dim(0)?;
    println!("  hidden_size: {}", hidden_size);
    println!("  codec_vocab_size: {}", vocab_size);

    // Get text tokens
    let text_tokens = text_to_ids(&args.text);
    println!("\nText tokens: {:?}", text_tokens);

    // Build TTS config
    let tts_config = TtsGenerationConfig {
        max_frames: args.frames,
        temperature: args.temperature,
        top_k: args.top_k,
        language_id: TtsTokenIds::default().lang_english,
        speaker_id: None, // Base model
        use_think: true,
        token_ids: TtsTokenIds::default(),
    };

    // Build prefill embeddings
    println!("\nBuilding prefill embeddings...");
    let (prefill_embed, trailing_embed) = build_prefill_embeddings(
        &text_tokens,
        text_embedding,
        codec_embedding,
        &weights,
        &tts_config,
        &device,
    )?;

    println!("  prefill shape: {:?}", prefill_embed.shape());
    println!("  trailing shape: {:?}", trailing_embed.shape());

    // Run transformer forward pass
    println!("\nRunning talker transformer...");
    let (hidden_states, _kv_caches) =
        run_talker_forward(&prefill_embed, &weights, hidden_size, &device)?;

    println!("  hidden_states shape: {:?}", hidden_states.shape());

    // Get logits for first semantic token
    let last_hidden = hidden_states.i((.., hidden_states.dim(1)? - 1.., ..))?;
    println!("  last_hidden shape: {:?}", last_hidden.shape());
    let logits = linear_3d(&last_hidden, codec_head, None)?;
    let logits = logits.squeeze(1)?; // [batch, vocab]
    println!("  logits shape: {:?}", logits.shape());

    // Apply token suppression
    let logits = apply_token_suppression(&logits, vocab_size, tts_config.token_ids.codec_eos)?;

    // Apply temperature
    let logits = if args.temperature != 1.0 {
        (logits / args.temperature)?
    } else {
        logits
    };

    // Use greedy sampling for debugging (take argmax)
    let first_token = generation::greedy_sample(&logits)?;
    let first_token_id: u32 = first_token.flatten_all()?.to_vec1::<u32>()?[0];
    println!("\nFirst semantic token (greedy): {}", first_token_id);

    // Autoregressive generation loop
    // For each frame:
    //   1. Sample semantic token from current hidden state
    //   2. Run code predictor to get 15 acoustic codes
    //   3. Sum all 16 codec embeddings (residual pattern)
    //   4. Add next text token embedding if available
    //   5. Run through transformer to update hidden state
    println!("\nAutoregressive generation of {} frames...", args.frames);

    let mut all_codes: Vec<Vec<i64>> = Vec::new();

    // Get codec embedding table
    let codec_embedding = weights
        .get("talker.model.codec_embedding.weight")
        .ok_or_else(|| anyhow::anyhow!("Missing codec_embedding"))?;

    // Track current input sequence for transformer
    let mut current_inputs = prefill_embed.clone();
    let mut text_idx = 0; // Track which text token we're on

    for frame in 0..args.frames {
        // Get hidden states for current sequence
        let (hidden_states, _) = run_talker_forward(&current_inputs, &weights, hidden_size, &device)?;
        let last_hidden = hidden_states.i((.., hidden_states.dim(1)? - 1.., ..))?;

        // Sample semantic token
        let logits = linear_3d(&last_hidden, codec_head, None)?;
        let logits = logits.squeeze(1)?;
        let logits = apply_token_suppression(&logits, vocab_size, tts_config.token_ids.codec_eos)?;
        let logits = if args.temperature != 1.0 {
            (logits / args.temperature)?
        } else {
            logits
        };
        let semantic_token: u32 = generation::greedy_sample(&logits)?
            .flatten_all()?
            .to_vec1::<u32>()?[0];

        // Check for EOS
        if semantic_token == tts_config.token_ids.codec_eos {
            println!("  Frame {}: EOS token, stopping", frame);
            break;
        }

        // Run code predictor to get 15 acoustic codes
        let acoustic_codes = run_code_predictor(&last_hidden, semantic_token, &weights, &device)?;

        // Build full frame codes: semantic + 15 acoustic
        let mut frame_codes = vec![semantic_token as i64];
        for code in &acoustic_codes {
            frame_codes.push(*code as i64);
        }
        all_codes.push(frame_codes.clone());

        if frame < 10 {
            println!(
                "  Frame {}: semantic={}, acoustic[0..3]={:?}",
                frame,
                semantic_token,
                &frame_codes[1..4]
            );
        }

        // Build next input: sum of all 16 codec embeddings + next text token
        // Sum all 16 codec embeddings (residual VQ pattern)
        let mut all_token_ids: Vec<u32> = vec![semantic_token];
        all_token_ids.extend(&acoustic_codes);

        let mut residual_sum = Tensor::zeros((1, 1, hidden_size), DType::F32, &device)?;
        for &code_id in &all_token_ids {
            let code_idx = Tensor::new(&[code_id], &device)?;
            let code_embed = codec_embedding.index_select(&code_idx, 0)?.unsqueeze(0)?;
            residual_sum = (residual_sum + code_embed)?;
        }

        // Add next text token embedding if available
        let next_input = if text_idx < trailing_embed.dim(1)? {
            let text_pos = trailing_embed.i((.., text_idx..text_idx + 1, ..))?;
            text_idx += 1;
            (residual_sum + text_pos)?
        } else {
            residual_sum
        };

        // Append to input sequence
        current_inputs = Tensor::cat(&[&current_inputs, &next_input], 1)?;
    }

    // Convert codes to tensor [1, 16, num_frames]
    let num_frames = all_codes.len();
    let mut codes_flat = vec![0i64; 16 * num_frames];
    for (f, frame_codes) in all_codes.iter().enumerate() {
        for (q, &code) in frame_codes.iter().enumerate() {
            codes_flat[q * num_frames + f] = code;
        }
    }
    let codes_tensor = Tensor::from_vec(codes_flat, (1, 16, num_frames), &device)?;
    println!("\nCodes tensor shape: {:?}", codes_tensor.shape());

    // Save codes as binary for comparison with Python
    let codes_path = "test_data/rust_codes.bin";
    let codes_i64: Vec<i64> = codes_tensor.flatten_all()?.to_vec1()?;
    let codes_bytes: Vec<u8> = codes_i64.iter().flat_map(|&x| x.to_le_bytes()).collect();
    std::fs::write(codes_path, &codes_bytes)?;
    println!("Saved codes to: {} ({} codes, {} bytes)", codes_path, codes_i64.len(), codes_bytes.len());

    // Decode to audio
    println!("Decoding to audio...");
    let decoder = Decoder12Hz::from_weights(&decoder_weights, Default::default())?;
    let waveform = decoder.decode(&codes_tensor)?;
    let audio_samples: Vec<f32> = waveform.flatten_all()?.to_vec1()?;
    println!(
        "Audio: {} samples ({:.2}s at 24kHz)",
        audio_samples.len(),
        audio_samples.len() as f64 / 24000.0
    );

    // Save
    let audio_buffer = AudioBuffer::new(audio_samples, 24000);
    audio_buffer.save(&args.output)?;
    println!("\nSaved to: {}", args.output);

    generation::clear_seed();
    Ok(())
}

/// Run talker transformer forward pass
fn run_talker_forward(
    input_embeds: &Tensor,
    weights: &HashMap<String, Tensor>,
    _hidden_size: usize,
    device: &Device,
) -> Result<(Tensor, Vec<(Tensor, Tensor)>)> {
    let num_layers = 28;
    let num_heads = 16;
    let num_kv_heads = 8;
    let head_dim = 128;
    let eps = 1e-6f64;
    let rope_theta = 1000000.0f64;

    let (batch, seq_len, _) = input_embeds.dims3()?;

    // Build RoPE
    let positions = Tensor::arange(0u32, seq_len as u32, device)?;
    let inv_freq_vals: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|i| 1.0 / (rope_theta as f32).powf(i as f32 / head_dim as f32))
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq_vals, (head_dim / 2,), device)?;
    let freqs = positions
        .to_dtype(DType::F32)?
        .unsqueeze(1)?
        .matmul(&inv_freq.unsqueeze(0)?)?;
    let cos = freqs.cos()?.repeat((1, 2))?;
    let sin = freqs.sin()?.repeat((1, 2))?;
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, seq, head_dim]
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

    // Causal mask
    let mask = create_causal_mask(seq_len, device)?;

    let mut hidden = input_embeds.clone();
    let mut kv_caches = Vec::new();

    for layer_idx in 0..num_layers {
        let prefix = format!("talker.model.layers.{}", layer_idx);

        // Input LayerNorm
        let ln_w = weights
            .get(&format!("{}.input_layernorm.weight", prefix))
            .unwrap();
        let normed = rms_norm(&hidden, ln_w, eps)?;

        // QKV projections
        let q = linear_3d(
            &normed,
            weights.get(&format!("{}.self_attn.q_proj.weight", prefix)).unwrap(),
            None,
        )?;
        let k = linear_3d(
            &normed,
            weights.get(&format!("{}.self_attn.k_proj.weight", prefix)).unwrap(),
            None,
        )?;
        let v = linear_3d(
            &normed,
            weights.get(&format!("{}.self_attn.v_proj.weight", prefix)).unwrap(),
            None,
        )?;

        // Reshape
        let q = q.reshape((batch, seq_len, num_heads, head_dim))?;
        let k = k.reshape((batch, seq_len, num_kv_heads, head_dim))?;
        let v = v.reshape((batch, seq_len, num_kv_heads, head_dim))?;

        // QK norm
        let q = rms_norm(
            &q,
            weights.get(&format!("{}.self_attn.q_norm.weight", prefix)).unwrap(),
            eps,
        )?;
        let k = rms_norm(
            &k,
            weights.get(&format!("{}.self_attn.k_norm.weight", prefix)).unwrap(),
            eps,
        )?;

        // Transpose to [batch, heads, seq, dim]
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // RoPE
        let (q, k) = apply_rope(&q, &k, &cos, &sin)?;

        // Store KV cache
        kv_caches.push((k.clone(), v.clone()));

        // Repeat KV for GQA
        let k = repeat_kv(&k, num_heads / num_kv_heads)?;
        let v = repeat_kv(&v, num_heads / num_kv_heads)?;

        // Attention
        let scale = (head_dim as f64).powf(-0.5);
        let attn = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?.affine(scale, 0.0)?;
        let attn = attn.broadcast_add(&mask)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let attn_out = attn.matmul(&v)?;

        // Reshape back
        let attn_out = attn_out
            .transpose(1, 2)?
            .reshape((batch, seq_len, num_heads * head_dim))?;

        // O projection
        let attn_out = linear_3d(
            &attn_out,
            weights.get(&format!("{}.self_attn.o_proj.weight", prefix)).unwrap(),
            None,
        )?;

        // Residual
        hidden = hidden.add(&attn_out)?;

        // MLP
        let ln_w = weights
            .get(&format!("{}.post_attention_layernorm.weight", prefix))
            .unwrap();
        let normed = rms_norm(&hidden, ln_w, eps)?;

        let gate = linear_3d(
            &normed,
            weights.get(&format!("{}.mlp.gate_proj.weight", prefix)).unwrap(),
            None,
        )?;
        let up = linear_3d(
            &normed,
            weights.get(&format!("{}.mlp.up_proj.weight", prefix)).unwrap(),
            None,
        )?;
        let mlp_out = candle_nn::ops::silu(&gate)?.mul(&up)?;
        let mlp_out = linear_3d(
            &mlp_out,
            weights.get(&format!("{}.mlp.down_proj.weight", prefix)).unwrap(),
            None,
        )?;

        hidden = hidden.add(&mlp_out)?;
    }

    // Final norm
    let norm_w = weights.get("talker.model.norm.weight").unwrap();
    hidden = rms_norm(&hidden, norm_w, eps)?;

    Ok((hidden, kv_caches))
}

fn linear_3d(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let dims = x.dims();
    let (batch, seq, _) = (dims[0], dims[1], dims[2]);
    let x_2d = x.reshape((batch * seq, x.dim(2)?))?;
    let out_2d = x_2d.matmul(&weight.t()?)?;
    let out_3d = out_2d.reshape((batch, seq, out_2d.dim(1)?))?;
    match bias {
        Some(b) => Ok(out_3d.broadcast_add(b)?),
        None => Ok(out_3d),
    }
}

fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
    let x_norm = x.broadcast_div(&(variance + eps)?.sqrt()?)?;
    Ok(x_norm.broadcast_mul(weight)?)
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    Ok(Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?)
}

fn apply_rope(q: &Tensor, k: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<(Tensor, Tensor)> {
    let q_rot = q
        .broadcast_mul(cos)?
        .broadcast_add(&rotate_half(q)?.broadcast_mul(sin)?)?;
    let k_rot = k
        .broadcast_mul(cos)?
        .broadcast_add(&rotate_half(k)?.broadcast_mul(sin)?)?;
    Ok((q_rot, k_rot))
}

fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (batch, n_kv_heads, seq_len, head_dim) = x.dims4()?;
    let x = x.unsqueeze(2)?;
    let x = x.expand((batch, n_kv_heads, n_rep, seq_len, head_dim))?;
    Ok(x.reshape((batch, n_kv_heads * n_rep, seq_len, head_dim))?)
}

fn create_causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| {
            (0..seq_len).map(move |j| if j <= i { 0.0 } else { f32::NEG_INFINITY })
        })
        .collect();
    Ok(Tensor::from_vec(mask, (1, 1, seq_len, seq_len), device)?)
}

/// Run code predictor to generate 15 acoustic codes from talker hidden and semantic token
fn run_code_predictor(
    talker_hidden: &Tensor,
    semantic_token: u32,
    weights: &HashMap<String, Tensor>,
    device: &Device,
) -> Result<Vec<u32>> {
    // Get codec embedding for semantic token (from talker's codec_embedding)
    let codec_embedding = weights
        .get("talker.model.codec_embedding.weight")
        .ok_or_else(|| anyhow::anyhow!("Missing codec_embedding"))?;
    let semantic_idx = Tensor::new(&[semantic_token], device)?;
    let semantic_embed = codec_embedding.index_select(&semantic_idx, 0)?.unsqueeze(0)?;
    // [1, 1, hidden_size]

    // Concatenate talker_hidden + semantic_embed
    let input = Tensor::cat(&[talker_hidden, &semantic_embed], 1)?;
    // [1, 2, hidden_size]

    // Run through code predictor transformer (5 layers)
    let num_layers = 5;
    let num_heads = 16;
    let num_kv_heads = 8;
    let head_dim = 128;
    let eps = 1e-6f64;
    let rope_theta = 1000000.0f64;

    let seq_len = input.dim(1)?;
    let positions = Tensor::arange(0u32, seq_len as u32, device)?;
    let inv_freq_vals: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|i| 1.0 / (rope_theta as f32).powf(i as f32 / head_dim as f32))
        .collect();
    let inv_freq = Tensor::new(&inv_freq_vals[..], device)?;
    let freqs = positions
        .to_dtype(DType::F32)?
        .unsqueeze(1)?
        .matmul(&inv_freq.unsqueeze(0)?)?;
    let freqs = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
    let cos = freqs.cos()?.unsqueeze(0)?.unsqueeze(0)?;
    let sin = freqs.sin()?.unsqueeze(0)?.unsqueeze(0)?;

    let mask = create_causal_mask(seq_len, device)?;
    let n_rep = num_heads / num_kv_heads;

    let mut hidden = input;

    for layer_idx in 0..num_layers {
        let prefix = format!("talker.code_predictor.model.layers.{}", layer_idx);

        // Layer norm
        let ln_w = weights
            .get(&format!("{}.input_layernorm.weight", prefix))
            .ok_or_else(|| anyhow::anyhow!("Missing input_layernorm"))?;
        let normed = rms_norm(&hidden, ln_w, eps)?;

        // Self-attention
        let q_proj = weights.get(&format!("{}.self_attn.q_proj.weight", prefix)).unwrap();
        let k_proj = weights.get(&format!("{}.self_attn.k_proj.weight", prefix)).unwrap();
        let v_proj = weights.get(&format!("{}.self_attn.v_proj.weight", prefix)).unwrap();
        let o_proj = weights.get(&format!("{}.self_attn.o_proj.weight", prefix)).unwrap();

        let q = linear_3d(&normed, q_proj, None)?;
        let k = linear_3d(&normed, k_proj, None)?;
        let v = linear_3d(&normed, v_proj, None)?;

        // Q/K norms
        let q_norm_w = weights.get(&format!("{}.self_attn.q_norm.weight", prefix)).unwrap();
        let k_norm_w = weights.get(&format!("{}.self_attn.k_norm.weight", prefix)).unwrap();

        let (batch, seq, _) = q.dims3()?;
        let q = q.reshape((batch, seq, num_heads, head_dim))?;
        let k = k.reshape((batch, seq, num_kv_heads, head_dim))?;
        let v = v.reshape((batch, seq, num_kv_heads, head_dim))?;

        // Per-head RMS norm
        let q = {
            let q_flat = q.reshape((batch * seq * num_heads, head_dim))?;
            let q_normed = rms_norm_per_head(&q_flat, q_norm_w, eps)?;
            q_normed.reshape((batch, seq, num_heads, head_dim))?
        };
        let k = {
            let k_flat = k.reshape((batch * seq * num_kv_heads, head_dim))?;
            let k_normed = rms_norm_per_head(&k_flat, k_norm_w, eps)?;
            k_normed.reshape((batch, seq, num_kv_heads, head_dim))?
        };

        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        let (q, k) = apply_rope(&q, &k, &cos, &sin)?;
        let k = repeat_kv(&k, n_rep)?;
        let v = repeat_kv(&v, n_rep)?;

        let scale = (head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? / scale)?;
        let attn_weights = attn_weights.broadcast_add(&mask)?;
        let attn_probs = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_out = attn_probs.matmul(&v)?;

        let attn_out = attn_out.transpose(1, 2)?.reshape((batch, seq, num_heads * head_dim))?;
        let attn_out = linear_3d(&attn_out, o_proj, None)?;

        hidden = (hidden + attn_out)?;

        // MLP
        let post_ln_w = weights
            .get(&format!("{}.post_attention_layernorm.weight", prefix))
            .unwrap();
        let normed = rms_norm(&hidden, post_ln_w, eps)?;

        let gate_w = weights.get(&format!("{}.mlp.gate_proj.weight", prefix)).unwrap();
        let up_w = weights.get(&format!("{}.mlp.up_proj.weight", prefix)).unwrap();
        let down_w = weights.get(&format!("{}.mlp.down_proj.weight", prefix)).unwrap();

        let gate = linear_3d(&normed, gate_w, None)?;
        let up = linear_3d(&normed, up_w, None)?;
        let mlp_out = linear_3d(&(candle_nn::ops::silu(&gate)? * up)?, down_w, None)?;

        hidden = (hidden + mlp_out)?;
    }

    // Final norm
    let final_norm_w = weights
        .get("talker.code_predictor.model.norm.weight")
        .ok_or_else(|| anyhow::anyhow!("Missing code_predictor norm"))?;
    hidden = rms_norm(&hidden, final_norm_w, eps)?;

    // Extract position 1 (semantic embed position) hidden state
    let pos1_hidden = hidden.i((.., 1..2, ..))?;

    // Apply 15 lm_heads to get acoustic codes
    let mut acoustic_codes = Vec::with_capacity(15);
    for i in 0..15 {
        let lm_head_w = weights
            .get(&format!("talker.code_predictor.lm_head.{}.weight", i))
            .ok_or_else(|| anyhow::anyhow!("Missing lm_head.{}", i))?;
        let logits = linear_3d(&pos1_hidden, lm_head_w, None)?;
        let token: u32 = logits.argmax(D::Minus1)?.flatten_all()?.to_vec1::<u32>()?[0];
        acoustic_codes.push(token);
    }

    Ok(acoustic_codes)
}

fn rms_norm_per_head(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let (_num_tokens, dim) = x.dims2()?;
    let x_sq = x.sqr()?;
    let variance = (x_sq.sum(1)? / dim as f64)?;
    let eps = Tensor::new(&[eps as f32], x.device())?.broadcast_as(variance.shape())?;
    let rms = (variance + eps)?.sqrt()?;
    let x_norm = x.broadcast_div(&rms.unsqueeze(1)?)?;
    Ok(x_norm.broadcast_mul(&weight.unsqueeze(0)?)?)
}
