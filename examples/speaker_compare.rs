use anyhow::Result;
use qwen3_tts::{AudioBuffer, Language, Qwen3TTS, SynthesisOptions};

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let device = qwen3_tts::auto_device()?;
    let model = Qwen3TTS::from_pretrained("test_data/model", device)?;

    let text = "Hello world.";

    // Test 1: Voice clone with temperature=0, compare Apollo vs Sine
    eprintln!("\n=== Test 1: Voice clone, temperature=0 ===");
    let options = SynthesisOptions {
        max_length: 20,
        temperature: 0.0,
        ..Default::default()
    };

    let ref_audio = AudioBuffer::load("examples/data/clone_2.wav")?;
    let prompt1 = model.create_voice_clone_prompt(&ref_audio, None)?;
    let audio1 =
        model.synthesize_voice_clone(text, &prompt1, Language::English, Some(options.clone()))?;
    eprintln!("Clone2: {} samples", audio1.len());

    let sine: Vec<f32> = (0..24000 * 3)
        .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 24000.0).sin() * 0.5)
        .collect();
    let sine_audio = AudioBuffer::new(sine, 24000);
    let prompt2 = model.create_voice_clone_prompt(&sine_audio, None)?;
    let audio2 = model.synthesize_voice_clone(text, &prompt2, Language::English, Some(options))?;
    eprintln!("Sine: {} samples", audio2.len());

    let max_diff = audio1
        .samples
        .iter()
        .zip(&audio2.samples)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("Max diff (Clone2 vs Sine, temp=0): {:.6}", max_diff);

    // Test 2: Voice clone with temperature=0.7
    eprintln!("\n=== Test 2: Voice clone, temperature=0.7 ===");
    let options2 = SynthesisOptions {
        max_length: 20,
        temperature: 0.7,
        ..Default::default()
    };
    let audio3 =
        model.synthesize_voice_clone(text, &prompt1, Language::English, Some(options2.clone()))?;
    eprintln!("Clone2 t=0.7: {} samples", audio3.len());
    let audio4 = model.synthesize_voice_clone(text, &prompt2, Language::English, Some(options2))?;
    eprintln!("Sine t=0.7: {} samples", audio4.len());

    let max_diff2 = audio3
        .samples
        .iter()
        .zip(&audio4.samples)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("Max diff (Clone2 vs Sine, temp=0.7): {:.6}", max_diff2);

    Ok(())
}
