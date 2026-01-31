use anyhow::Result;
use qwen3_tts::{AudioBuffer, Language, Qwen3TTS, SynthesisOptions};

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let device = qwen3_tts::auto_device()?;
    let model = Qwen3TTS::from_pretrained("test_data/model", device)?;

    // Text to synthesize — intentionally unrelated to the reference audio
    let text = "The quick brown fox jumps over the lazy dog near the old brick wall.";
    let options = SynthesisOptions {
        max_length: 100,
        temperature: 0.9,
        top_k: 50,
        repetition_penalty: 1.05,
        ..Default::default()
    };

    // ICL voice clone: speaker embedding + reference audio codes + transcript
    let ref_audio = AudioBuffer::load("examples/data/clone_2.wav")?;
    let ref_text = "Okay. Yeah. I resent you. I love you. I respect you. \
                    But you know what? You blew it! And thanks to you.";

    assert!(
        model.has_speech_encoder(),
        "Speech encoder required for ICL mode"
    );

    let prompt = model.create_voice_clone_prompt(&ref_audio, Some(ref_text))?;
    let (audio, codes) =
        model.synthesize_voice_clone_debug(text, &prompt, Language::English, Some(options))?;
    audio.save("output_clone_icl.wav")?;

    // Print semantic tokens to check for EOS (2150)
    let semantic_tokens: Vec<u32> = codes.iter().map(|f| f[0]).collect();
    eprintln!(
        "Semantic tokens ({} frames): {:?}",
        codes.len(),
        &semantic_tokens
    );
    let has_eos = semantic_tokens.contains(&qwen3_tts::CODEC_EOS_TOKEN_ID);
    eprintln!(
        "Contains EOS ({}): {has_eos}",
        qwen3_tts::CODEC_EOS_TOKEN_ID
    );
    eprintln!(
        "ICL clone: {:.2}s, {} samples → output_clone_icl.wav",
        audio.duration(),
        audio.len()
    );

    Ok(())
}
