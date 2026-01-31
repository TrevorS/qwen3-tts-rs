use anyhow::Result;
use qwen3_tts::{AudioBuffer, Language, Qwen3TTS, SynthesisOptions};

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let device = qwen3_tts::auto_device()?;
    let model = Qwen3TTS::from_pretrained("test_data/model", device)?;

    let text = "Hello world.";
    let options = SynthesisOptions {
        max_length: 5,
        temperature: 0.0,
        ..Default::default()
    };

    eprintln!("\n=== Clone 2 ===");
    let ref_audio = AudioBuffer::load("examples/data/clone_2.wav")?;
    let prompt1 = model.create_voice_clone_prompt(&ref_audio, None)?;
    let _ =
        model.synthesize_voice_clone(text, &prompt1, Language::English, Some(options.clone()))?;

    eprintln!("\n=== Sine wave ===");
    let sine: Vec<f32> = (0..24000 * 3)
        .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 24000.0).sin() * 0.5)
        .collect();
    let sine_audio = AudioBuffer::new(sine, 24000);
    let prompt2 = model.create_voice_clone_prompt(&sine_audio, None)?;
    let _ = model.synthesize_voice_clone(text, &prompt2, Language::English, Some(options))?;

    Ok(())
}
