//! CustomVoice TTS generation
//!
//! Generates audio using the CustomVoice model with built-in speakers.
//!
//! Usage:
//!     cargo run --release --features cli --bin custom_voice_tts -- --text "Hello" --speaker ryan

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use clap::Parser;
use std::collections::HashMap;
use std::path::Path;

use qwen3_tts::{generation, models, AudioBuffer, CodePredictorConfig, Language, Speaker, TalkerConfig};

/// Generate TTS audio with CustomVoice model
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Text to synthesize
    #[arg(short, long, default_value = "Hello")]
    text: String,

    /// Speaker name (ryan, vivian, serena, aiden, etc.)
    #[arg(short = 'S', long, default_value = "ryan")]
    speaker: String,

    /// Language (english, chinese, japanese, etc.)
    #[arg(short, long, default_value = "english")]
    language: String,

    /// Random seed for reproducible generation
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Number of frames to generate
    #[arg(short, long, default_value_t = 50)]
    frames: usize,

    /// Sampling temperature
    #[arg(long, default_value_t = 0.9)]
    temperature: f64,

    /// Top-k sampling parameter
    #[arg(long, default_value_t = 50)]
    top_k: usize,

    /// Model directory (CustomVoice model)
    #[arg(short, long, default_value = "../test_data/model_customvoice")]
    model_dir: String,

    /// Output WAV file
    #[arg(short, long, default_value = "output.wav")]
    output: String,
}

/// Parse speaker name to Speaker enum
fn parse_speaker(name: &str) -> Result<Speaker> {
    match name.to_lowercase().as_str() {
        "ryan" => Ok(Speaker::Ryan),
        "vivian" => Ok(Speaker::Vivian),
        "serena" => Ok(Speaker::Serena),
        "aiden" => Ok(Speaker::Aiden),
        "uncle_fu" | "unclefu" => Ok(Speaker::UncleFu),
        "ono_anna" | "onoanna" => Ok(Speaker::OnoAnna),
        "sohee" => Ok(Speaker::Sohee),
        "eric" => Ok(Speaker::Eric),
        "dylan" => Ok(Speaker::Dylan),
        _ => anyhow::bail!("Unknown speaker: {}. Options: ryan, vivian, serena, aiden, uncle_fu, ono_anna, sohee, eric, dylan", name),
    }
}

/// Parse language name to Language enum
fn parse_language(name: &str) -> Result<Language> {
    match name.to_lowercase().as_str() {
        "english" | "en" => Ok(Language::English),
        "chinese" | "zh" => Ok(Language::Chinese),
        "japanese" | "ja" => Ok(Language::Japanese),
        "korean" | "ko" => Ok(Language::Korean),
        "german" | "de" => Ok(Language::German),
        "french" | "fr" => Ok(Language::French),
        "russian" | "ru" => Ok(Language::Russian),
        "portuguese" | "pt" => Ok(Language::Portuguese),
        "spanish" | "es" => Ok(Language::Spanish),
        "italian" | "it" => Ok(Language::Italian),
        _ => anyhow::bail!("Unknown language: {}. Options: english, chinese, japanese, korean, german, french, russian, portuguese, spanish, italian", name),
    }
}

/// Simple text to token mapping
fn text_to_ids(text: &str) -> Vec<u32> {
    match text {
        "Hello" => vec![9707],
        "Hello world" => vec![9707, 1879],
        "Hello, this is a test" => vec![9707, 11, 419, 374, 264, 1273],
        "Good morning" => vec![15571, 6017],
        "How are you?" => vec![4340, 525, 498, 30],
        _ => {
            eprintln!("Warning: Text '{}' not in tokenizer mapping, using 'Hello'", text);
            vec![9707]
        }
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    let speaker = parse_speaker(&args.speaker)?;
    let language = parse_language(&args.language)?;

    println!("=== CustomVoice TTS ===");
    println!("Text: {}", args.text);
    println!("Speaker: {:?}", speaker);
    println!("Language: {:?}", language);
    println!("Frames: {}", args.frames);
    println!("Temperature: {}", args.temperature);

    // Set seed
    generation::set_seed(args.seed);
    generation::reset_rng();
    println!("Seed: {}", args.seed);

    let device = Device::Cpu;

    // Load weights
    println!("\nLoading model from {}...", args.model_dir);
    let model_path = Path::new(&args.model_dir).join("model.safetensors");
    let weights: HashMap<String, Tensor> = candle_core::safetensors::load(&model_path, &device)?;
    let weights: HashMap<String, Tensor> = weights
        .into_iter()
        .map(|(name, tensor)| {
            let converted = if tensor.dtype() == DType::BF16 {
                tensor.to_dtype(DType::F32).unwrap()
            } else {
                tensor
            };
            (name, converted)
        })
        .collect();

    // Load decoder
    let decoder_path = Path::new(&args.model_dir).join("speech_tokenizer/model.safetensors");
    let decoder_weights: HashMap<String, Tensor> = candle_core::safetensors::load(&decoder_path, &device)?;
    let decoder_weights: HashMap<String, Tensor> = decoder_weights
        .into_iter()
        .map(|(name, tensor)| {
            let converted = if tensor.dtype() == DType::BF16 {
                tensor.to_dtype(DType::F32).unwrap()
            } else {
                tensor
            };
            (name, converted)
        })
        .collect();

    // Create models with CustomVoice config
    println!("Creating TalkerModel (CustomVoice config)...");
    let talker_config = TalkerConfig::custom_voice();
    let talker = models::TalkerModel::from_weights_with_config(&weights, talker_config, &device)?;

    println!("Creating CodePredictor (CustomVoice config)...");
    let cp_config = CodePredictorConfig::custom_voice();
    let cp_weights: HashMap<String, Tensor> = weights
        .iter()
        .filter_map(|(k, v)| {
            if k.starts_with("talker.code_predictor.") {
                Some((k.strip_prefix("talker.code_predictor.").unwrap().to_string(), v.clone()))
            } else {
                None
            }
        })
        .collect();
    let cp_vb = candle_nn::VarBuilder::from_tensors(cp_weights, DType::F32, &device);
    let code_predictor = models::CodePredictor::new(cp_config, cp_vb)?;

    println!("Creating Decoder12Hz...");
    let decoder = models::codec::Decoder12Hz::from_weights(&decoder_weights, Default::default())?;

    // Get text tokens
    let text_tokens = text_to_ids(&args.text);
    println!("\nText tokens: {:?}", text_tokens);

    // Generation config
    let gen_config = generation::GenerationConfig {
        max_new_tokens: args.frames,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: 1.0,
        repetition_penalty: 1.0,
        eos_token_id: None,
    };

    // Suppress special codec tokens (2048-3071 except EOS=2150)
    // This matches the Python implementation
    let vocab_size = 3072;
    let codec_eos = 2150;
    let suppress_mask: Vec<f32> = (0..vocab_size)
        .map(|i| {
            if i >= 2048 && i != codec_eos {
                f32::NEG_INFINITY
            } else {
                0.0
            }
        })
        .collect();
    let suppress_tensor = Tensor::from_vec(suppress_mask, (1, vocab_size), &device)?;

    // Generate semantic tokens using CustomVoice
    println!("\nGenerating {} frames with CustomVoice...", args.frames);

    // Build trailing_text_hidden: remaining text tokens (after first) + tts_eos, all projected
    // For single token text like "Hello" = [9707], remaining = [], so trailing = [tts_eos]
    let tts_eos_embed = talker.get_tts_eos_embed()?; // [1, 1, hidden_size]
    let tts_pad_embed = talker.get_tts_pad_embed()?; // [1, 1, hidden_size]

    let trailing_text_hidden = if text_tokens.len() > 1 {
        // Get projected embeddings for remaining tokens
        let remaining_embeds = talker.get_projected_text_embeddings(&text_tokens[1..])?;
        // Concatenate with tts_eos
        Tensor::cat(&[remaining_embeds, tts_eos_embed.clone()], 1)?
    } else {
        // Only tts_eos for single-token text
        tts_eos_embed.clone()
    };
    let trailing_len = trailing_text_hidden.dim(1)?;
    println!("Trailing text hidden shape: [{}, {}]", 1, trailing_len);

    let mut kv_caches = talker.new_kv_caches();
    let (hidden, logits) = talker.prefill_custom_voice(&text_tokens, speaker, language, &mut kv_caches)?;

    // Calculate offset - prefill creates:
    // 3 role prefix + 6 codec positions + 1 first text = 10 positions
    let prefill_len = 3 + 6 + 1; // = 10
    let mut offset = prefill_len;

    // Sample first semantic token (with token suppression)
    let suppressed_logits = logits.squeeze(1)?.add(&suppress_tensor)?;
    let first_token = generation::sample(&suppressed_logits, &gen_config)?;
    let first_token_id: u32 = first_token.flatten_all()?.to_vec1::<u32>()?[0];
    println!("First semantic token: {}", first_token_id);

    // Check for immediate EOS
    if first_token_id == codec_eos as u32 {
        println!("EOS detected immediately, no audio to generate");
        return Ok(());
    }

    let seq_len = hidden.dim(1)?;
    let mut past_hidden = hidden.i((.., seq_len - 1..seq_len, ..))?; // [1, 1, hidden_size]

    let mut all_codes: Vec<Vec<u32>> = vec![];
    let mut prev_semantic = first_token_id;

    // Generation loop - each iteration processes prev_semantic and predicts next
    for frame_idx in 0..args.frames {
        // Check for EOS
        if prev_semantic == codec_eos as u32 {
            println!("EOS detected at frame {}, stopping generation", frame_idx);
            break;
        }

        // Get semantic embedding
        let semantic_embed = talker.get_codec_embedding(prev_semantic)?;

        // Generate acoustic codes from [past_hidden, semantic_embed]
        let acoustic_codes = code_predictor.generate_acoustic_codes(&past_hidden, &semantic_embed)?;

        // Save this frame's codes
        let mut frame_codes = vec![prev_semantic];
        frame_codes.extend(&acoustic_codes);
        all_codes.push(frame_codes.clone());

        if frame_idx < 5 || frame_idx == args.frames - 1 {
            println!("Frame {}: semantic={}, acoustics={:?}...", frame_idx, prev_semantic, &acoustic_codes[..3]);
        } else if frame_idx == 5 {
            println!("...");
        }

        // If last frame requested, don't predict next
        if frame_idx == args.frames - 1 {
            break;
        }

        // Build input embedding: sum(semantic_embed + acoustic_embeds) + trailing_text
        let acoustic_sum = code_predictor.get_acoustic_embeddings_sum(&acoustic_codes, &device)?;
        let mut input_embed = semantic_embed.add(&acoustic_sum)?;

        // Add trailing text or tts_pad
        let text_to_add = if frame_idx < trailing_len {
            trailing_text_hidden.i((.., frame_idx..frame_idx + 1, ..))?
        } else {
            tts_pad_embed.clone()
        };
        input_embed = input_embed.add(&text_to_add)?;

        // Forward through talker
        let (hidden, logits) = talker.generate_step_with_embed(&input_embed, &mut kv_caches, offset)?;
        offset += 1;
        past_hidden = hidden;

        // Sample next semantic token
        let suppressed_logits = logits.squeeze(1)?.add(&suppress_tensor)?;
        let next_token = generation::sample(&suppressed_logits, &gen_config)?;
        let next_token_id: u32 = next_token.flatten_all()?.to_vec1::<u32>()?[0];

        prev_semantic = next_token_id;
    }

    // Check if we have any codes
    if all_codes.is_empty() {
        println!("No codes generated, nothing to decode");
        return Ok(());
    }

    // Convert to tensor [1, 16, num_frames] - decoder expects [batch, quantizers, seq_len]
    let num_frames = all_codes.len();
    let mut codes_data = vec![0i64; 16 * num_frames];
    for (frame, frame_codes) in all_codes.iter().enumerate() {
        for (q, &code) in frame_codes.iter().enumerate() {
            codes_data[q * num_frames + frame] = code as i64;
        }
    }
    let codes_tensor = Tensor::from_vec(codes_data, (1, 16, num_frames), &device)?;
    println!("\nCodes tensor shape: {:?}", codes_tensor.shape());

    // Decode to audio
    println!("Decoding to audio...");
    let waveform = decoder.decode(&codes_tensor)?;
    let audio_samples: Vec<f32> = waveform.flatten_all()?.to_vec1()?;
    let min_val = audio_samples.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = audio_samples.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let rms: f32 = (audio_samples.iter().map(|x| x * x).sum::<f32>() / audio_samples.len() as f32).sqrt();
    println!("Audio samples: {} ({:.2}s at 24kHz)", audio_samples.len(), audio_samples.len() as f64 / 24000.0);
    println!("Audio range: [{:.6}, {:.6}], RMS: {:.6}", min_val, max_val, rms);

    // Save WAV
    let audio_buffer = AudioBuffer::new(audio_samples, 24000);
    audio_buffer.save(&args.output)?;
    println!("\nSaved to: {}", args.output);

    generation::clear_seed();
    println!("Done!");

    Ok(())
}
