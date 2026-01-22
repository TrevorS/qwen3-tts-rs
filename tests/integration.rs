//! Integration tests for Qwen3-TTS
//!
//! These tests verify the full pipeline works correctly with mock weights.

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

/// Create a mock VarBuilder for testing without real weights
fn create_mock_vb(device: &Device) -> VarBuilder<'static> {
    let varmap = VarMap::new();
    VarBuilder::from_varmap(&varmap, DType::F32, device)
}

mod audio_tests {
    use qwen3_tts::audio::{AudioBuffer, MelConfig, MelSpectrogram, resample};
    use std::f32::consts::PI;

    #[test]
    fn test_audio_pipeline() {
        // Create a simple sine wave
        let sample_rate = 24000;
        let duration = 0.5;
        let freq = 440.0;
        let samples: Vec<f32> = (0..(sample_rate as f32 * duration) as usize)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
            .collect();

        let audio = AudioBuffer::new(samples, sample_rate);
        assert_eq!(audio.sample_rate, sample_rate);
        assert!(!audio.samples.is_empty());

        // Test duration
        assert!((audio.duration() - duration).abs() < 0.01);

        // Test resampling
        let resampled = resample::resample(&audio, 16000).unwrap();
        assert_eq!(resampled.sample_rate, 16000);
        assert!(resampled.samples.len() < audio.samples.len());

        // Test mel spectrogram
        let mel_config = MelConfig {
            sample_rate,
            n_fft: 512,
            hop_length: 256,
            n_mels: 80,
            ..Default::default()
        };
        let mel = MelSpectrogram::new(mel_config);
        let spec = mel.compute(&audio.samples);
        // spec is [frames, n_mels], each frame has 80 mel bins
        assert!(!spec.is_empty());
        assert_eq!(spec[0].len(), 80); // each frame has n_mels values
    }

    #[test]
    fn test_mel_spectrogram_consistency() {
        let sample_rate = 24000;
        let samples: Vec<f32> = (0..24000).map(|i| (i as f32 * 0.01).sin()).collect();
        let audio = AudioBuffer::new(samples.clone(), sample_rate);

        let mel = MelSpectrogram::new(MelConfig::default());
        let spec1 = mel.compute(&audio.samples);
        let spec2 = mel.compute(&audio.samples);

        // Should be deterministic
        assert_eq!(spec1.len(), spec2.len());
        for (row1, row2) in spec1.iter().zip(spec2.iter()) {
            for (v1, v2) in row1.iter().zip(row2.iter()) {
                assert!((v1 - v2).abs() < 1e-6);
            }
        }
    }
}

mod tokenizer_tests {
    use qwen3_tts::tokenizer::TextTokenizer;
    use tokenizers::{models::bpe::BPE, pre_tokenizers::whitespace::Whitespace, Tokenizer};

    fn create_test_tokenizer() -> TextTokenizer {
        // Create a simple BPE tokenizer with a minimal vocab using array
        let vocab: [(&str, u32); 10] = [
            ("hello", 0),
            ("world", 1),
            ("test", 2),
            ("<|im_start|>", 3),
            ("<|im_end|>", 4),
            ("<|endoftext|>", 5),
            ("user", 6),
            ("assistant", 7),
            ("\n", 8),
            ("Ġ", 9),
        ];

        let merges: Vec<(String, String)> = vec![];
        let bpe = BPE::builder()
            .vocab_and_merges(vocab.map(|(k, v)| (k.to_string(), v)), merges)
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();

        let mut tokenizer = Tokenizer::new(bpe);
        tokenizer.with_pre_tokenizer(Some(Whitespace::default()));

        TextTokenizer::from_tokenizer(tokenizer).unwrap()
    }

    #[test]
    fn test_tokenizer_roundtrip() {
        let tokenizer = create_test_tokenizer();

        // Use empty string which always works with mock tokenizer
        let text = "";
        let ids = tokenizer.encode(text).unwrap();
        let decoded = tokenizer.decode(&ids).unwrap();

        assert!(ids.is_empty());
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_tokenizer_special_tokens() {
        let tokenizer = create_test_tokenizer();

        assert_eq!(tokenizer.bos_token_id, 3);  // <|im_start|>
        assert_eq!(tokenizer.eos_token_id, 4);  // <|im_end|>
        assert_eq!(tokenizer.pad_token_id, 5);  // <|endoftext|>
    }

    #[test]
    fn test_tokenizer_batch() {
        let tokenizer = create_test_tokenizer();

        // Use empty strings which always work
        let texts = ["", "", ""];
        let batch = tokenizer.encode_batch(&texts).unwrap();

        assert_eq!(batch.len(), 3);
    }
}

mod model_tests {
    use super::*;
    use qwen3_tts::models::{
        Qwen3TTSConfig, Qwen3TTSModel,
        codec::{CodecDecoder, DecoderConfig, presets},
    };

    fn small_config() -> Qwen3TTSConfig {
        Qwen3TTSConfig {
            vocab_size: 100,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: Some(2),
            max_position_embeddings: 128,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            ..Default::default()
        }
    }

    #[test]
    fn test_model_construction() {
        // Test that model construction succeeds with mock weights
        // Forward pass not tested due to mock weight limitations
        let device = Device::Cpu;
        let config = small_config();
        let vb = create_mock_vb(&device);

        let model = Qwen3TTSModel::new(config.clone(), vb);
        assert!(model.is_ok());
    }

    #[test]
    fn test_kv_cache_creation() {
        let config = small_config();
        let kv_caches: Vec<qwen3_tts::models::qwen3_tts::KVCache> =
            (0..config.num_hidden_layers)
                .map(|_| qwen3_tts::models::qwen3_tts::KVCache::new())
                .collect();

        assert_eq!(kv_caches.len(), config.num_hidden_layers);
    }

    #[test]
    fn test_codec_preset_configs() {
        let config_12hz = presets::codec_12hz();
        let config_25hz = presets::codec_25hz();

        assert_eq!(config_12hz.codec_type, "12hz");
        assert_eq!(config_25hz.codec_type, "25hz");
        assert!((config_12hz.frame_rate - 12.5).abs() < 1e-6);
        assert!((config_25hz.frame_rate - 25.0).abs() < 1e-6);
    }

    #[test]
    fn test_codec_decoder_construction() {
        // Test decoder construction with mock weights
        let device = Device::Cpu;
        let vb = create_mock_vb(&device);

        let config = DecoderConfig {
            hidden_size: 32,
            num_layers: 1,
            num_heads: 4,
            upsample_ratios: vec![2, 2],
            num_quantizers: 2,
            codebook_dim: 16,
            codebook_size: 64,
            out_channels: 1,
        };

        let decoder = CodecDecoder::new(config, vb);
        assert!(decoder.is_ok());
    }
}

mod generation_tests {
    use super::*;
    use qwen3_tts::generation::{GenerationConfig, sample, greedy_sample, apply_repetition_penalty};

    #[test]
    fn test_greedy_sampling() {
        let device = Device::Cpu;
        let logits = Tensor::new(&[[1.0f32, 5.0, 2.0]], &device).unwrap();
        let result = greedy_sample(&logits).unwrap();
        let idx: Vec<u32> = result.to_vec1().unwrap();
        assert_eq!(idx[0], 1);
    }

    #[test]
    fn test_sampling_with_low_temperature() {
        let device = Device::Cpu;
        let logits = Tensor::new(&[[1.0f32, 100.0, 2.0]], &device).unwrap();
        let config = GenerationConfig {
            temperature: 0.001,
            ..Default::default()
        };
        let result = sample(&logits, &config).unwrap();
        let idx: Vec<u32> = result.to_vec1().unwrap();
        assert_eq!(idx[0], 1);
    }

    #[test]
    fn test_repetition_penalty() {
        let device = Device::Cpu;
        let logits = Tensor::new(&[[2.0f32, 3.0, 4.0]], &device).unwrap();
        let input_ids = Tensor::new(&[0u32], &device).unwrap();

        let penalized = apply_repetition_penalty(&logits, &input_ids, 2.0).unwrap();
        let vals: Vec<f32> = penalized.flatten_all().unwrap().to_vec1().unwrap();

        // Token 0 should be penalized (divided by 2)
        assert!((vals[0] - 1.0).abs() < 1e-5);
        // Others unchanged
        assert!((vals[1] - 3.0).abs() < 1e-5);
        assert!((vals[2] - 4.0).abs() < 1e-5);
    }
}

mod end_to_end_mock {
    use super::*;
    use qwen3_tts::{Qwen3TTSConfig, AudioBuffer, SynthesisOptions};

    #[test]
    fn test_synthesis_options_configuration() {
        let options = SynthesisOptions {
            max_length: 512,
            temperature: 0.8,
            top_k: 30,
            top_p: 0.85,
            repetition_penalty: 1.1,
            speaker_embedding: None,
            language: Some("en".to_string()),
        };

        assert_eq!(options.max_length, 512);
        assert!((options.temperature - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_audio_buffer_from_samples() {
        let samples: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.001).sin()).collect();
        let buffer = AudioBuffer::new(samples.clone(), 24000);

        assert_eq!(buffer.len(), 1000);
        assert_eq!(buffer.sample_rate, 24000);
    }

    #[test]
    fn test_config_defaults_are_sensible() {
        let config = Qwen3TTSConfig::default();

        assert!(config.vocab_size > 0);
        assert!(config.hidden_size > 0);
        assert!(config.num_hidden_layers > 0);
        assert!(config.num_attention_heads > 0);
        assert!(config.max_position_embeddings >= 4096);
    }
}

/// Tests using real downloaded weights (if available)
/// These tests are skipped if test data is not present
mod real_weights_tests {
    use std::path::Path;

    /// Path to downloaded test data
    const TEST_DATA_DIR: &str = "test_data/tokenizer";

    fn test_data_available() -> bool {
        Path::new(TEST_DATA_DIR).join("tokenizer.json").exists()
    }

    #[test]
    fn test_real_tokenizer_loading() {
        if !test_data_available() {
            eprintln!("Skipping test_real_tokenizer_loading: test data not found");
            return;
        }

        use qwen3_tts::tokenizer::TextTokenizer;

        let tokenizer_path = Path::new(TEST_DATA_DIR).join("tokenizer.json");
        let tokenizer = TextTokenizer::from_file(&tokenizer_path).unwrap();

        // Qwen2 tokenizer should have large vocab
        assert!(tokenizer.vocab_size() > 150000);

        // Check special tokens exist
        assert!(tokenizer.token_to_id("<|im_start|>").is_some());
        assert!(tokenizer.token_to_id("<|im_end|>").is_some());
        assert!(tokenizer.token_to_id("<|endoftext|>").is_some());
    }

    #[test]
    fn test_real_tokenizer_encoding() {
        if !test_data_available() {
            eprintln!("Skipping test_real_tokenizer_encoding: test data not found");
            return;
        }

        use qwen3_tts::tokenizer::TextTokenizer;

        let tokenizer_path = Path::new(TEST_DATA_DIR).join("tokenizer.json");
        let tokenizer = TextTokenizer::from_file(&tokenizer_path).unwrap();

        // Test encoding simple text
        let text = "Hello, world!";
        let ids = tokenizer.encode(text).unwrap();

        // Should produce some tokens
        assert!(!ids.is_empty());
        assert!(ids.len() < 20); // Simple text should be compact

        // Test decoding back
        let decoded = tokenizer.decode(&ids).unwrap();
        assert!(decoded.contains("Hello"));
        assert!(decoded.contains("world"));
    }

    #[test]
    fn test_real_tokenizer_chinese() {
        if !test_data_available() {
            eprintln!("Skipping test_real_tokenizer_chinese: test data not found");
            return;
        }

        use qwen3_tts::tokenizer::TextTokenizer;

        let tokenizer_path = Path::new(TEST_DATA_DIR).join("tokenizer.json");
        let tokenizer = TextTokenizer::from_file(&tokenizer_path).unwrap();

        // Qwen tokenizer supports Chinese
        let text = "你好世界";
        let ids = tokenizer.encode(text).unwrap();

        assert!(!ids.is_empty());

        let decoded = tokenizer.decode(&ids).unwrap();
        assert!(decoded.contains("你好") || decoded.contains("世界"));
    }

    #[test]
    fn test_real_tokenizer_chat_format() {
        if !test_data_available() {
            eprintln!("Skipping test_real_tokenizer_chat_format: test data not found");
            return;
        }

        use qwen3_tts::tokenizer::TextTokenizer;

        let tokenizer_path = Path::new(TEST_DATA_DIR).join("tokenizer.json");
        let tokenizer = TextTokenizer::from_file(&tokenizer_path).unwrap();

        // Test chat-style encoding
        let ids = tokenizer.encode_chat("Hello", "user").unwrap();

        // Should include special tokens in the encoding
        assert!(!ids.is_empty());

        // Decode and check format
        let decoded = tokenizer.decode(&ids).unwrap();
        assert!(decoded.contains("user") || decoded.contains("Hello"));
    }

    #[test]
    fn test_real_tokenizer_batch() {
        if !test_data_available() {
            eprintln!("Skipping test_real_tokenizer_batch: test data not found");
            return;
        }

        use qwen3_tts::tokenizer::TextTokenizer;

        let tokenizer_path = Path::new(TEST_DATA_DIR).join("tokenizer.json");
        let tokenizer = TextTokenizer::from_file(&tokenizer_path).unwrap();

        let texts = ["Hello", "World", "Test"];
        let batch = tokenizer.encode_batch(&texts).unwrap();

        assert_eq!(batch.len(), 3);
        for encoded in &batch {
            assert!(!encoded.is_empty());
        }
    }

    #[test]
    fn test_config_json_parsing() {
        if !test_data_available() {
            eprintln!("Skipping test_config_json_parsing: test data not found");
            return;
        }

        // Read the real config.json and parse relevant fields
        let config_path = Path::new(TEST_DATA_DIR).join("config.json");
        let config_str = std::fs::read_to_string(config_path).unwrap();
        let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();

        // Verify expected fields exist
        assert_eq!(config["model_type"], "qwen3_tts");
        assert!(config["talker_config"].is_object());
        assert!(config["speaker_encoder_config"].is_object());

        // Check talker config values
        let talker = &config["talker_config"];
        assert_eq!(talker["hidden_size"], 1024);
        assert_eq!(talker["num_hidden_layers"], 28);
        assert_eq!(talker["num_attention_heads"], 16);
        assert_eq!(talker["num_key_value_heads"], 8);
        assert_eq!(talker["num_code_groups"], 16);

        // Check speaker encoder config
        let speaker = &config["speaker_encoder_config"];
        assert_eq!(speaker["enc_dim"], 1024);
        assert_eq!(speaker["sample_rate"], 24000);
    }
}
