//! Qwen3-TTS usage examples.
//!
//! Demonstrates the public API: basic synthesis, voice/language selection,
//! custom options, voice cloning, and streaming.
//!
//! ```sh
//! cargo run --example tts -- --model-dir path/to/model
//! ```

use anyhow::Result;
use qwen3_tts::{AudioBuffer, Language, Qwen3TTS, Speaker, SynthesisOptions};
use std::env;

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let model_dir = env::args()
        .skip_while(|a| a != "--model-dir")
        .nth(1)
        .unwrap_or_else(|| "test_data/model".into());

    let device = qwen3_tts::auto_device()?;
    println!("Loading model from: {model_dir}");
    let model = Qwen3TTS::from_pretrained(&model_dir, device)?;

    // ── 1. Basic synthesis (default voice: Ryan, English) ────────────
    let audio = model.synthesize("Hello from Rust!", None)?;
    audio.save("output_basic.wav")?;
    println!("Basic: {:.2}s → output_basic.wav", audio.duration());

    // ── 2. Choose a speaker and language ─────────────────────────────
    let audio = model.synthesize_with_voice(
        "This uses a different voice.",
        Speaker::Serena,
        Language::English,
        None,
    )?;
    audio.save("output_serena.wav")?;
    println!("Serena: {:.2}s → output_serena.wav", audio.duration());

    // ── 3. Custom generation options ─────────────────────────────────
    let options = SynthesisOptions {
        temperature: 0.9,
        top_k: 30,
        max_length: 512,
        ..Default::default()
    };
    let audio = model.synthesize_with_voice(
        "Custom sampling parameters.",
        Speaker::Ryan,
        Language::English,
        Some(options),
    )?;
    audio.save("output_custom.wav")?;
    println!("Custom: {:.2}s → output_custom.wav", audio.duration());

    // ── 4. Voice cloning ────────────────────────────────────────────
    //
    // Reference clip: Apollo 11 — Neil Armstrong stepping onto the Moon.
    // Public domain (NASA/US government work).
    // Note: the radio transmission artifacts will carry into the clone.
    // For best results, use clean studio-quality reference audio.
    if model.supports_voice_cloning() {
        let ref_audio = AudioBuffer::load("examples/data/apollo11_one_small_step.wav")?;

        // x_vector_only mode: speaker embedding only
        let prompt = model.create_voice_clone_prompt(&ref_audio, None)?;
        let audio = model.synthesize_voice_clone(
            "We choose to go to the Moon in this decade.",
            &prompt,
            Language::English,
            None,
        )?;
        audio.save("output_clone.wav")?;
        println!(
            "Clone (x-vector): {:.2}s → output_clone.wav",
            audio.duration()
        );

        // ICL mode: speaker embedding + reference audio codes + transcript
        if model.has_speech_encoder() {
            let ref_text = "I'm going to step off the LM now. \
                            That's one small step for man, \
                            one giant leap for mankind.";
            let prompt = model.create_voice_clone_prompt(&ref_audio, Some(ref_text))?;
            let audio = model.synthesize_voice_clone(
                "We choose to go to the Moon in this decade.",
                &prompt,
                Language::English,
                None,
            )?;
            audio.save("output_clone_icl.wav")?;
            println!(
                "Clone (ICL): {:.2}s → output_clone_icl.wav",
                audio.duration()
            );
        }
    } else {
        println!("Skipping voice cloning (no speaker encoder in this model)");
    }

    // ── 5. Streaming synthesis ───────────────────────────────────────
    let options = SynthesisOptions {
        chunk_frames: 10, // ~800ms per chunk
        ..Default::default()
    };
    let mut total_samples = 0usize;
    for (i, chunk) in model
        .synthesize_streaming(
            "Streaming output, chunk by chunk.",
            Speaker::Ryan,
            Language::English,
            options,
        )?
        .enumerate()
    {
        let audio = chunk?;
        total_samples += audio.len();
        println!(
            "  chunk {i}: {} samples ({:.2}s)",
            audio.len(),
            audio.duration()
        );
    }
    println!("Streaming total: {:.2}s", total_samples as f32 / 24000.0);

    Ok(())
}
