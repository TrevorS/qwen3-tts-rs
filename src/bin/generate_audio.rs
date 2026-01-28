//! CLI tool for generating audio with deterministic seed
//!
//! This tool generates WAV audio files using the Qwen3-TTS pipeline with a
//! specific seed, allowing direct comparison with Python output.
//!
//! Usage:
//!     cargo run --features cli --bin generate_audio -- --text "Hello" --seed 42 --frames 25
//!     cargo run --features cli --bin generate_audio -- --text "Hello" --seed 42 --duration 2.0

use anyhow::Result;
use byteorder::{LittleEndian, WriteBytesExt};
use candle_core::{DType, Device, IndexOp, Tensor};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use models::talker::{Language, Speaker, TalkerConfig};
use qwen3_tts::{
    device_info, generation, models, parse_device, tokenizer, AudioBuffer, ModelType,
    ParsedModelConfig, Qwen3TTS, SynthesisOptions,
};

/// Generate reference audio with deterministic seed for comparison
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Text to synthesize
    #[arg(short, long, default_value = "Hello")]
    text: String,

    /// Random seed for reproducible generation
    #[arg(short, long, default_value_t = 42)]
    seed: u64,

    /// Number of frames to generate (overridden by duration if specified)
    #[arg(short, long, default_value_t = 25)]
    frames: usize,

    /// Duration in seconds (overrides frames if specified)
    #[arg(short, long)]
    duration: Option<f64>,

    /// Sampling temperature
    #[arg(long, default_value_t = 0.7)]
    temperature: f64,

    /// Top-k sampling parameter
    #[arg(long, default_value_t = 50)]
    top_k: usize,

    /// Top-p (nucleus) sampling parameter
    #[arg(long, default_value_t = 0.9)]
    top_p: f64,

    /// Model directory containing model.safetensors
    #[arg(short, long, default_value = "test_data/model")]
    model_dir: String,

    /// Output directory for generated files
    #[arg(short, long, default_value = "test_data/rust_audio")]
    output_dir: String,

    /// Compare with Python reference output (if exists)
    #[arg(short, long)]
    compare: bool,

    /// Python reference directory
    #[arg(long, default_value = "test_data/reference_audio")]
    reference_dir: String,

    /// Tokenizer directory (defaults to model_dir/../tokenizer)
    #[arg(long)]
    tokenizer_dir: Option<String>,

    /// Speaker name for CustomVoice (ryan, serena, vivian, aiden, etc.)
    #[arg(long, default_value = "ryan")]
    speaker: String,

    /// Language for TTS (english, chinese, japanese, etc.)
    #[arg(long, default_value = "english")]
    language: String,

    /// Force 1.7B config (hidden=2048). Usually auto-detected from config.json.
    #[arg(long)]
    custom_voice: bool,

    /// Voice description for VoiceDesign model (e.g. "A cheerful young female voice")
    #[arg(long)]
    instruct: Option<String>,

    /// Reference audio WAV file for voice cloning (x_vector_only mode)
    #[arg(long)]
    ref_audio: Option<String>,

    /// Reference text for ICL voice cloning (requires --ref-audio)
    #[arg(long)]
    ref_text: Option<String>,

    /// Use x_vector_only mode (speaker embedding only, no ICL)
    #[arg(long)]
    x_vector_only: bool,

    /// Repetition penalty (1.0 = disabled, 1.05 = Python default)
    #[arg(long, default_value_t = 1.05)]
    repetition_penalty: f64,

    /// Output WAV file path (overrides default naming in --output-dir)
    #[arg(long)]
    output: Option<String>,

    /// Device for inference (auto, cpu, cuda, cuda:N, metal)
    #[arg(long, default_value = "auto")]
    device: String,
}

fn parse_speaker(name: &str) -> Result<Speaker> {
    match name.to_lowercase().as_str() {
        "ryan" => Ok(Speaker::Ryan),
        "serena" => Ok(Speaker::Serena),
        "vivian" => Ok(Speaker::Vivian),
        "aiden" => Ok(Speaker::Aiden),
        "uncle_fu" | "unclefu" => Ok(Speaker::UncleFu),
        "ono_anna" | "onoanna" => Ok(Speaker::OnoAnna),
        "sohee" => Ok(Speaker::Sohee),
        "eric" => Ok(Speaker::Eric),
        "dylan" => Ok(Speaker::Dylan),
        _ => anyhow::bail!("Unknown speaker: {}", name),
    }
}

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
        _ => anyhow::bail!("Unknown language: {}", name),
    }
}

/// Metadata for generated audio
#[derive(Debug, Serialize, Deserialize)]
struct GenerationMetadata {
    text: String,
    seed: u64,
    num_frames: usize,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    input_ids: Vec<u32>,
    codes_shape: Vec<usize>,
    audio_samples: usize,
    sample_rate: u32,
}

/// Validate CLI arg combinations and bail early on contradictory flags.
fn validate_args(args: &Args) -> Result<()> {
    // --instruct and --ref-audio are mutually exclusive
    if args.instruct.is_some() && args.ref_audio.is_some() {
        anyhow::bail!(
            "--instruct and --ref-audio are mutually exclusive.\n  \
             --instruct is for VoiceDesign models (text-described voices).\n  \
             --ref-audio is for Base models (voice cloning from reference audio)."
        );
    }

    // --ref-text requires --ref-audio (ref text is an ICL transcript for voice cloning)
    if args.ref_text.is_some() && args.ref_audio.is_none() {
        anyhow::bail!("--ref-text requires --ref-audio (reference text is the transcript of the reference audio for ICL voice cloning)");
    }

    // --x-vector-only requires --ref-audio (it's a voice cloning mode)
    if args.x_vector_only && args.ref_audio.is_none() {
        anyhow::bail!(
            "--x-vector-only requires --ref-audio (x_vector_only is a voice cloning mode)"
        );
    }

    // --x-vector-only + --ref-text is contradictory: x_vector_only disables ICL
    if args.x_vector_only && args.ref_text.is_some() {
        anyhow::bail!(
            "--x-vector-only and --ref-text are mutually exclusive.\n  \
             x_vector_only uses only the speaker embedding (no ICL).\n  \
             Remove --x-vector-only to use ICL mode with reference text."
        );
    }

    // --custom-voice + --ref-audio: CustomVoice models don't have a speaker encoder
    if args.custom_voice && args.ref_audio.is_some() {
        anyhow::bail!(
            "--custom-voice and --ref-audio are incompatible.\n  \
             CustomVoice models use preset speakers (--speaker), not voice cloning.\n  \
             Use a Base model for voice cloning with reference audio."
        );
    }

    // --compare only works in the preset speaker path (low-level generation)
    if args.compare && args.ref_audio.is_some() {
        anyhow::bail!(
            "--compare is only supported in the preset speaker code path.\n  \
             Voice cloning uses the high-level API which doesn't emit comparison data."
        );
    }

    Ok(())
}

/// Voice cloning path: uses the high-level Qwen3TTS API when --ref-audio is provided.
fn run_voice_clone(args: &Args) -> Result<()> {
    let ref_audio_path = args.ref_audio.as_ref().expect("--ref-audio required");

    println!("=== Voice Clone Generation ===");
    println!("Text: {}", args.text);
    println!("Reference audio: {}", ref_audio_path);
    if let Some(ref rt) = args.ref_text {
        println!("Reference text (ICL): {}", rt);
    } else {
        println!("Mode: x_vector_only (no reference text)");
    }

    // Set seed
    generation::set_seed(args.seed);
    generation::reset_rng();
    println!("Seed: {}", args.seed);

    let device = parse_device(&args.device)?;
    println!("Device: {}", device_info(&device));
    let model = Qwen3TTS::from_pretrained(&args.model_dir, device)?;

    // Load reference audio
    let ref_audio = AudioBuffer::load(ref_audio_path)?;
    println!(
        "Loaded reference audio: {:.2}s, {} samples",
        ref_audio.duration(),
        ref_audio.len()
    );

    // Create voice clone prompt (x_vector_only or ICL depending on --ref-text)
    let prompt = model.create_voice_clone_prompt(&ref_audio, args.ref_text.as_deref())?;

    // Debug: print ref_codes info if ICL mode
    if let Some(ref codes) = &prompt.ref_codes {
        let shape = codes.shape();
        println!("ref_codes shape: {:?}", shape);
        // Print first frame
        let first_frame: Vec<u32> = codes.i(0)?.to_vec1()?;
        println!("ref_codes[0] (first frame): {:?}", first_frame);
        // Print first 10 semantic codes
        let semantic: Vec<u32> = codes.i((.., 0))?.to_vec1::<u32>()?;
        println!(
            "First 10 semantic codes: {:?}",
            &semantic[..10.min(semantic.len())]
        );
    }

    let language = parse_language(&args.language)?;

    // Calculate max frames from duration if specified
    let max_frames = if let Some(duration) = args.duration {
        (duration * 12.5) as usize
    } else {
        args.frames
    };

    let options = SynthesisOptions {
        max_length: max_frames,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        repetition_penalty: args.repetition_penalty,
        ..Default::default()
    };

    println!("Generating up to {} frames...", max_frames);
    let (audio, codes) =
        model.synthesize_voice_clone_debug(&args.text, &prompt, language, Some(options))?;
    println!(
        "Generated: {:.2}s, {} samples, {} frames",
        audio.duration(),
        audio.len(),
        codes.len()
    );

    // Print semantic tokens for debugging
    let semantic_tokens: Vec<u32> = codes.iter().map(|f| f[0]).collect();
    println!("Semantic tokens: {:?}", &semantic_tokens);
    let has_eos = semantic_tokens.contains(&qwen3_tts::CODEC_EOS_TOKEN_ID);
    println!(
        "Contains EOS ({}): {}",
        qwen3_tts::CODEC_EOS_TOKEN_ID,
        has_eos
    );

    // Determine output path
    let output_path = if let Some(ref out) = args.output {
        std::path::PathBuf::from(out)
    } else {
        let output_dir = Path::new(&args.output_dir);
        fs::create_dir_all(output_dir)?;
        output_dir.join(format!("audio_seed{}_frames{}.wav", args.seed, max_frames))
    };

    // Ensure parent directory exists
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    audio.save(&output_path)?;
    println!("Saved WAV to: {}", output_path.display());

    generation::clear_seed();
    println!("Generation complete!");
    Ok(())
}

/// VoiceDesign path: uses text-described voice conditioning when --instruct is provided.
fn run_voice_design(args: &Args) -> Result<()> {
    let instruct = args.instruct.as_ref().expect("--instruct required");

    println!("=== VoiceDesign Generation ===");
    println!("Text: {}", args.text);
    println!("Instruct: {}", instruct);

    // Set seed
    generation::set_seed(args.seed);
    generation::reset_rng();
    println!("Seed: {}", args.seed);

    let device = parse_device(&args.device)?;
    println!("Device: {}", device_info(&device));
    let model = Qwen3TTS::from_pretrained(&args.model_dir, device)?;

    if !model.supports_voice_design() {
        eprintln!(
            "  WARNING: This model is not a VoiceDesign variant. \
             Voice design synthesis may produce unpredictable results."
        );
    }

    let language = parse_language(&args.language)?;

    // Calculate max frames from duration if specified
    let max_frames = if let Some(duration) = args.duration {
        (duration * 12.5) as usize
    } else {
        args.frames
    };

    let options = SynthesisOptions {
        max_length: max_frames,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        repetition_penalty: args.repetition_penalty,
        ..Default::default()
    };

    println!("Generating up to {} frames...", max_frames);
    let audio = model.synthesize_voice_design(&args.text, instruct, language, Some(options))?;
    println!(
        "Generated: {:.2}s, {} samples",
        audio.duration(),
        audio.len()
    );

    // Determine output path
    let output_path = if let Some(ref out) = args.output {
        std::path::PathBuf::from(out)
    } else {
        let output_dir = Path::new(&args.output_dir);
        fs::create_dir_all(output_dir)?;
        output_dir.join(format!("audio_seed{}_frames{}.wav", args.seed, max_frames))
    };

    // Ensure parent directory exists
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    audio.save(&output_path)?;
    println!("Saved WAV to: {}", output_path.display());

    generation::clear_seed();
    println!("Generation complete!");
    Ok(())
}

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    validate_args(&args)?;

    // Voice clone path: when --ref-audio is provided, use the high-level API
    if args.ref_audio.is_some() {
        return run_voice_clone(&args);
    }

    // VoiceDesign path: when --instruct is provided, use text-described voice conditioning
    if args.instruct.is_some() {
        return run_voice_design(&args);
    }

    // Calculate frames from duration if specified
    let num_frames = if let Some(duration) = args.duration {
        (duration * 12.5) as usize
    } else {
        args.frames
    };

    println!("=== Generating Audio (Rust) ===");
    println!("Text: {}", args.text);
    println!("Seed: {}", args.seed);
    println!("Frames: {}", num_frames);
    println!("Temperature: {}", args.temperature);
    println!("Top-k: {}", args.top_k);
    println!("Top-p: {}", args.top_p);

    // Set seed
    generation::set_seed(args.seed);
    println!(
        "\nSeed set: {} (is_seeded: {})",
        args.seed,
        generation::is_seeded()
    );

    // Reset RNG to ensure deterministic starting point
    generation::reset_rng();

    let device = parse_device(&args.device)?;
    println!("Device: {}", device_info(&device));

    // Create output directory
    let output_dir = Path::new(&args.output_dir);
    fs::create_dir_all(output_dir)?;

    // Load weights
    println!("\nLoading model weights...");
    let model_path = Path::new(&args.model_dir).join("model.safetensors");
    let weights = load_weights(&model_path, &device)?;

    let decoder_path = Path::new(&args.model_dir).join("speech_tokenizer/model.safetensors");
    let decoder_weights = load_weights(&decoder_path, &device)?;

    // Load tokenizer
    let tokenizer_dir = args.tokenizer_dir.unwrap_or_else(|| {
        Path::new(&args.model_dir)
            .parent()
            .map(|p| p.join("tokenizer"))
            .unwrap_or_else(|| Path::new("tokenizer").to_path_buf())
            .to_string_lossy()
            .to_string()
    });
    println!("Loading tokenizer from {}...", tokenizer_dir);
    let text_tokenizer = tokenizer::TextTokenizer::from_pretrained(&tokenizer_dir)?;

    // Tokenize text
    let input_ids = text_tokenizer.encode(&args.text)?;
    println!("Input IDs: {:?}", input_ids);
    // Create models — auto-detect from config.json, fall back to --custom-voice flag
    let config_path = Path::new(&args.model_dir).join("config.json");
    let parsed_config = if config_path.exists() {
        match ParsedModelConfig::from_file(&config_path) {
            Ok(cfg) => {
                println!("Detected model variant: {}", cfg.label());
                Some(cfg)
            }
            Err(e) => {
                eprintln!("Warning: failed to parse config.json: {}", e);
                None
            }
        }
    } else {
        None
    };

    // Validate CLI args against detected model variant
    if let Some(ref cfg) = parsed_config {
        match cfg.model_type {
            ModelType::Base => {
                if args.ref_audio.is_none() {
                    eprintln!();
                    eprintln!(
                        "  WARNING: This is a {} model (trained for voice cloning).",
                        cfg.label()
                    );
                    eprintln!("  Without --ref-audio, the output voice will be unpredictable.");
                    eprintln!("  Recommended usage:");
                    eprintln!("    --ref-audio <path.wav>                 (x_vector_only mode)");
                    eprintln!("    --ref-audio <path.wav> --ref-text ...  (ICL mode)");
                    eprintln!();
                }
            }
            ModelType::CustomVoice => {
                if args.ref_audio.is_some() {
                    eprintln!();
                    eprintln!(
                        "  WARNING: This is a {} model (preset speakers only).",
                        cfg.label()
                    );
                    eprintln!("  --ref-audio is ignored — CustomVoice models don't have a speaker encoder.");
                    eprintln!(
                        "  Use --speaker <name> instead. Available: ryan, serena, vivian, aiden,"
                    );
                    eprintln!("  uncle_fu, ono_anna, sohee, eric, dylan");
                    eprintln!();
                }
            }
            ModelType::VoiceDesign => {
                if args.instruct.is_none() {
                    eprintln!();
                    eprintln!(
                        "  WARNING: This is a {} model (text-described voices).",
                        cfg.label()
                    );
                    eprintln!(
                        "  Without --instruct, falling back to preset speaker prefill — voice will be unpredictable."
                    );
                    eprintln!("  Recommended usage:");
                    eprintln!("    --instruct \"A cheerful young female voice with high pitch\"");
                    eprintln!();
                }
            }
        }
    }

    let (talker_config, cp_config) = if let Some(ref cfg) = parsed_config {
        (
            TalkerConfig::from_parsed(cfg),
            models::CodePredictorConfig::from_parsed(cfg),
        )
    } else if args.custom_voice {
        println!("Using CustomVoice config (hidden=2048, MRoPE)");
        (
            TalkerConfig::custom_voice(),
            models::CodePredictorConfig::custom_voice(),
        )
    } else {
        (
            TalkerConfig::default(),
            models::CodePredictorConfig::default(),
        )
    };

    println!(
        "Creating TalkerModel (hidden={})...",
        talker_config.hidden_size
    );
    let talker = models::TalkerModel::from_weights_with_config(&weights, talker_config, &device)?;

    println!("Creating CodePredictor...");
    let cp_weights = filter_weights(&weights, "talker.code_predictor.");
    let cp_vb = candle_nn::VarBuilder::from_tensors(cp_weights, DType::F32, &device);
    let code_predictor = models::CodePredictor::new(cp_config, cp_vb)?;

    println!("Creating Decoder12Hz...");
    let decoder = models::codec::Decoder12Hz::from_weights(&decoder_weights, Default::default())?;

    // Parse speaker and language for CustomVoice prefill
    let speaker = parse_speaker(&args.speaker)?;
    let language = parse_language(&args.language)?;
    println!("\nSpeaker: {:?}, Language: {:?}", speaker, language);

    // Build trailing text embeddings:
    //   remaining text tokens (all except first) projected + tts_eos
    //   After trailing text is exhausted, tts_pad is used for each subsequent step
    let trailing_text_hidden = if input_ids.len() > 1 {
        let remaining_proj = talker.get_projected_text_embeddings(&input_ids[1..])?;
        let tts_eos_embed = talker.get_tts_eos_embed()?;
        Tensor::cat(&[&remaining_proj, &tts_eos_embed], 1)?
    } else {
        talker.get_tts_eos_embed()?
    };
    let trailing_text_len = trailing_text_hidden.dim(1)?;
    let tts_pad_embed = talker.get_tts_pad_embed()?;
    println!("Trailing text length: {} positions", trailing_text_len);

    // Generate codes
    println!("\nGenerating {} frames...", num_frames);
    let progress = ProgressBar::new(num_frames as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} frames",
            )?
            .progress_chars("#>-"),
    );

    let gen_config = generation::GenerationConfig {
        max_new_tokens: num_frames,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        repetition_penalty: args.repetition_penalty,
        eos_token_id: if args.compare {
            None // Don't stop early when comparing with Python reference
        } else {
            Some(qwen3_tts::CODEC_EOS_TOKEN_ID)
        },
        min_new_tokens: 2,
    };

    // Initialize KV caches
    let mut kv_caches = talker.new_kv_caches();

    // Prefill with CustomVoice format:
    //   [role_prefix (im_start, assistant, newline)]
    //   [tts_pad×5 + tts_bos ADDED with codec_control (think, think_bos, lang, think_eos, speaker, pad)]
    //   [first_text_proj + codec_bos]
    let (hidden, logits) =
        talker.prefill_custom_voice(&input_ids, speaker, language, &mut kv_caches)?;
    let prefill_len = hidden.dim(1)?;
    let mut offset = prefill_len;
    println!("Prefill length: {} positions", prefill_len);

    // Get last hidden state (for code predictor input)
    let mut last_hidden = hidden.i((.., prefill_len - 1..prefill_len, ..))?;

    // Track generated tokens for repetition penalty
    let mut generated_tokens: Vec<u32> = Vec::new();

    // Apply repetition penalty + token suppression and sample first semantic token
    let logits_2d = logits.squeeze(1)?;
    let logits_2d = if gen_config.repetition_penalty != 1.0 && !generated_tokens.is_empty() {
        let prev = Tensor::new(generated_tokens.as_slice(), &device)?;
        generation::apply_repetition_penalty(&logits_2d, &prev, gen_config.repetition_penalty)?
    } else {
        logits_2d
    };
    let logits_suppressed =
        generation::apply_token_suppression(&logits_2d, 3072, qwen3_tts::CODEC_EOS_TOKEN_ID)?;
    let first_token = generation::sample(&logits_suppressed, &gen_config)?;
    let mut semantic_token: u32 = first_token.flatten_all()?.to_vec1::<u32>()?[0];
    generated_tokens.push(semantic_token);
    println!("First semantic token: {}", semantic_token);

    // Collect all codes
    let mut all_codes: Vec<Vec<u32>> = Vec::new();

    // Generation loop: for each frame, generate acoustic codes, then prepare next step
    for frame_idx in 0..num_frames {
        // Check EOS before processing this frame
        if let Some(eos_id) = gen_config.eos_token_id {
            if semantic_token == eos_id {
                println!(
                    "EOS token {} at frame {} — stopping generation",
                    eos_id, frame_idx
                );
                break;
            }
        }

        // Get semantic token embedding
        let semantic_embed = talker.get_codec_embedding(semantic_token)?;

        // Generate 15 acoustic codes using code predictor
        let acoustic_codes =
            code_predictor.generate_acoustic_codes(&last_hidden, &semantic_embed)?;

        if frame_idx < 5 || frame_idx == num_frames - 1 {
            println!(
                "Frame {}: semantic={}, acoustics={:?}...",
                frame_idx,
                semantic_token,
                &acoustic_codes[..3.min(acoustic_codes.len())]
            );
        } else if frame_idx == 5 {
            println!("...");
        }

        // Save frame codes [semantic, acoustic_0, ..., acoustic_14]
        let mut frame_codes = vec![semantic_token];
        frame_codes.extend(&acoustic_codes);
        all_codes.push(frame_codes);
        progress.inc(1);

        // If this is the last frame, we're done
        if frame_idx == num_frames - 1 {
            break;
        }

        // Build input for next talker step:
        // Sum all 16 code embeddings (residual VQ pattern: semantic + 15 acoustic)
        let acoustic_embed_sum =
            code_predictor.get_acoustic_embeddings_sum(&acoustic_codes, &device)?;
        let summed = semantic_embed.add(&acoustic_embed_sum)?;

        // Add trailing text embedding (or tts_pad if trailing text is exhausted)
        let text_addition = if frame_idx < trailing_text_len {
            trailing_text_hidden.i((.., frame_idx..frame_idx + 1, ..))?
        } else {
            tts_pad_embed.clone()
        };
        let step_input = summed.add(&text_addition)?;

        // Run through talker to get next hidden state and logits
        let (h, new_logits) =
            talker.generate_step_with_embed(&step_input, &mut kv_caches, offset)?;
        offset += 1;
        last_hidden = h;

        // Sample next semantic token with repetition penalty + token suppression
        let logits_2d = new_logits.squeeze(1)?;
        let logits_2d = if gen_config.repetition_penalty != 1.0 && !generated_tokens.is_empty() {
            let prev = Tensor::new(generated_tokens.as_slice(), &device)?;
            generation::apply_repetition_penalty(&logits_2d, &prev, gen_config.repetition_penalty)?
        } else {
            logits_2d
        };
        let logits_suppressed =
            generation::apply_token_suppression(&logits_2d, 3072, qwen3_tts::CODEC_EOS_TOKEN_ID)?;
        let next_token = generation::sample(&logits_suppressed, &gen_config)?;
        semantic_token = next_token.flatten_all()?.to_vec1::<u32>()?[0];
        generated_tokens.push(semantic_token);
    }

    progress.finish_with_message("Done generating codes");

    // Convert to tensor [1, 16, num_frames]
    let codes_tensor = codes_to_tensor(&all_codes, &device)?;
    println!("\nCodes tensor shape: {:?}", codes_tensor.shape());

    // Save codes as binary
    let codes_bin_path =
        output_dir.join(format!("codes_seed{}_frames{}.bin", args.seed, num_frames));
    save_codes_binary(&all_codes, &codes_bin_path)?;
    println!("Saved binary codes to: {:?}", codes_bin_path);

    // Decode to audio
    println!("\nDecoding to audio...");
    let waveform = decoder.decode(&codes_tensor)?;
    let audio_samples: Vec<f32> = waveform.flatten_all()?.to_vec1()?;
    println!(
        "Audio samples: {} ({:.3}s at 24kHz)",
        audio_samples.len(),
        audio_samples.len() as f64 / 24000.0
    );

    // Save audio as WAV
    let wav_path = if let Some(ref out) = args.output {
        let p = std::path::PathBuf::from(out);
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent)?;
        }
        p
    } else {
        output_dir.join(format!("audio_seed{}_frames{}.wav", args.seed, num_frames))
    };
    let audio_buffer = AudioBuffer::new(audio_samples.clone(), 24000);
    audio_buffer.save(&wav_path)?;
    println!("Saved WAV to: {:?}", wav_path);

    // Save audio as binary
    let audio_bin_path =
        output_dir.join(format!("audio_seed{}_frames{}.bin", args.seed, num_frames));
    save_audio_binary(&audio_samples, &audio_bin_path)?;
    println!("Saved binary audio to: {:?}", audio_bin_path);

    // Save metadata
    let metadata = GenerationMetadata {
        text: args.text.clone(),
        seed: args.seed,
        num_frames,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        input_ids,
        codes_shape: vec![1, 16, num_frames],
        audio_samples: audio_samples.len(),
        sample_rate: 24000,
    };
    let metadata_path = output_dir.join(format!(
        "metadata_seed{}_frames{}.json",
        args.seed, num_frames
    ));
    let metadata_file = File::create(&metadata_path)?;
    serde_json::to_writer_pretty(metadata_file, &metadata)?;
    println!("Saved metadata to: {:?}", metadata_path);

    // Compare with Python reference if requested
    if args.compare {
        println!("\n=== Comparing with Python Reference ===");
        compare_with_reference(
            &args.reference_dir,
            args.seed,
            num_frames,
            &all_codes,
            &audio_samples,
        )?;
    }

    // Clear seed
    generation::clear_seed();
    println!("\nGeneration complete!");

    Ok(())
}

/// Load weights from safetensors file
fn load_weights(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    let tensors: HashMap<String, Tensor> = candle_core::safetensors::load(path, device)?;
    let tensors: HashMap<String, Tensor> = tensors
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
    Ok(tensors)
}

/// Filter weights by prefix
fn filter_weights(weights: &HashMap<String, Tensor>, prefix: &str) -> HashMap<String, Tensor> {
    weights
        .iter()
        .filter_map(|(k, v)| {
            if k.starts_with(prefix) {
                Some((k.strip_prefix(prefix).unwrap().to_string(), v.clone()))
            } else {
                None
            }
        })
        .collect()
}

/// Convert list of frame codes to tensor [batch, 16, num_frames]
fn codes_to_tensor(codes: &[Vec<u32>], device: &Device) -> Result<Tensor> {
    let num_frames = codes.len();
    if num_frames == 0 {
        return Ok(Tensor::zeros((1, 16, 0), DType::I64, device)?);
    }

    let mut data = vec![0i64; 16 * num_frames];
    for (frame, frame_codes) in codes.iter().enumerate() {
        for (q, &code) in frame_codes.iter().enumerate() {
            data[q * num_frames + frame] = code as i64;
        }
    }

    Ok(Tensor::from_vec(data, (1, 16, num_frames), device)?)
}

/// Save codes as binary (row-major: frame0_q0, frame0_q1, ..., frame1_q0, ...)
fn save_codes_binary(codes: &[Vec<u32>], path: &Path) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Write as i64 to match Python format
    for frame_codes in codes {
        for &code in frame_codes {
            writer.write_i64::<LittleEndian>(code as i64)?;
        }
    }
    writer.flush()?;
    Ok(())
}

/// Save audio samples as binary f32
fn save_audio_binary(samples: &[f32], path: &Path) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    for &sample in samples {
        writer.write_f32::<LittleEndian>(sample)?;
    }
    writer.flush()?;
    Ok(())
}

/// Compare Rust output with Python reference
fn compare_with_reference(
    reference_dir: &str,
    seed: u64,
    num_frames: usize,
    rust_codes: &[Vec<u32>],
    rust_audio: &[f32],
) -> Result<()> {
    let ref_dir = Path::new(reference_dir);

    // Load Python codes
    let py_codes_path = ref_dir.join(format!("codes_seed{}_frames{}.bin", seed, num_frames));
    if py_codes_path.exists() {
        let py_codes_data = fs::read(&py_codes_path)?;
        let py_codes: Vec<i64> = py_codes_data
            .chunks(8)
            .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        // Flatten Rust codes for comparison
        let rust_codes_flat: Vec<i64> = rust_codes
            .iter()
            .flat_map(|frame| frame.iter().map(|&c| c as i64))
            .collect();

        // Compare codes
        let codes_match = py_codes.len() == rust_codes_flat.len()
            && py_codes
                .iter()
                .zip(rust_codes_flat.iter())
                .all(|(a, b)| a == b);

        if codes_match {
            println!("Codes: MATCH (all {} values identical)", py_codes.len());
        } else {
            println!("Codes: MISMATCH");
            println!("  Python: {} values", py_codes.len());
            println!("  Rust:   {} values", rust_codes_flat.len());

            // Show first differences
            let min_len = py_codes.len().min(rust_codes_flat.len());
            let mut diff_count = 0;
            for i in 0..min_len {
                if py_codes[i] != rust_codes_flat[i] {
                    if diff_count < 5 {
                        println!(
                            "  Index {}: Python={}, Rust={}",
                            i, py_codes[i], rust_codes_flat[i]
                        );
                    }
                    diff_count += 1;
                }
            }
            if diff_count > 5 {
                println!("  ... and {} more differences", diff_count - 5);
            }
            println!("  Total differences: {}", diff_count);
        }
    } else {
        println!("Codes: Python reference not found at {:?}", py_codes_path);
    }

    // Load Python audio
    let py_audio_path = ref_dir.join(format!("audio_seed{}_frames{}.bin", seed, num_frames));
    if py_audio_path.exists() {
        let py_audio_data = fs::read(&py_audio_path)?;
        let py_audio: Vec<f32> = py_audio_data
            .chunks(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        // Calculate audio difference statistics
        let min_len = py_audio.len().min(rust_audio.len());
        if min_len > 0 {
            let mut max_diff = 0.0f32;
            let mut sum_diff = 0.0f64;
            let mut sum_sq_diff = 0.0f64;

            for i in 0..min_len {
                let diff = (py_audio[i] - rust_audio[i]).abs();
                max_diff = max_diff.max(diff);
                sum_diff += diff as f64;
                sum_sq_diff += (diff * diff) as f64;
            }

            let mean_diff = sum_diff / min_len as f64;
            let rmse = (sum_sq_diff / min_len as f64).sqrt();

            println!("\nAudio comparison ({} samples):", min_len);
            println!("  Python samples: {}", py_audio.len());
            println!("  Rust samples:   {}", rust_audio.len());
            println!("  Max difference: {:.6}", max_diff);
            println!("  Mean difference: {:.6}", mean_diff);
            println!("  RMSE: {:.6}", rmse);

            // Check if audio is essentially identical
            if max_diff < 1e-5 {
                println!("  Status: MATCH (max diff < 1e-5)");
            } else if max_diff < 1e-3 {
                println!("  Status: CLOSE (max diff < 1e-3)");
            } else {
                println!("  Status: DIFFERENT");
            }
        }
    } else {
        println!("Audio: Python reference not found at {:?}", py_audio_path);
    }

    // Load and compare metadata
    let py_meta_path = ref_dir.join(format!("metadata_seed{}_frames{}.json", seed, num_frames));
    if py_meta_path.exists() {
        let py_meta: serde_json::Value = serde_json::from_reader(File::open(&py_meta_path)?)?;
        println!("\nPython metadata: {:?}", py_meta);
    }

    Ok(())
}
