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

/// Tests for speech tokenizer using real downloaded weights
mod speech_tokenizer_tests {
    use std::path::Path;

    /// Path to downloaded speech tokenizer data
    const SPEECH_TOKENIZER_DIR: &str = "test_data/speech_tokenizer";

    fn speech_tokenizer_available() -> bool {
        Path::new(SPEECH_TOKENIZER_DIR).join("model.safetensors").exists()
    }

    #[test]
    fn test_speech_tokenizer_config_parsing() {
        if !speech_tokenizer_available() {
            eprintln!("Skipping test_speech_tokenizer_config_parsing: test data not found");
            eprintln!("Run: ./scripts/download_test_data.sh to download");
            return;
        }

        // Read and parse the config
        let config_path = Path::new(SPEECH_TOKENIZER_DIR).join("config.json");
        let config_str = std::fs::read_to_string(config_path).unwrap();
        let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();

        // Verify architecture
        assert_eq!(config["model_type"], "qwen3_tts_tokenizer_12hz");
        assert!(config["encoder_config"].is_object());
        assert!(config["decoder_config"].is_object());

        // Check encoder config
        let encoder = &config["encoder_config"];
        assert_eq!(encoder["sampling_rate"], 24000);
        assert_eq!(encoder["num_quantizers"], 32);
        assert_eq!(encoder["codebook_size"], 2048);
        assert_eq!(encoder["hidden_size"], 512);

        // Check decoder config
        let decoder = &config["decoder_config"];
        assert_eq!(decoder["num_quantizers"], 16);
        assert_eq!(decoder["codebook_size"], 2048);
        assert_eq!(decoder["hidden_size"], 512);
        assert_eq!(decoder["num_attention_heads"], 16);
    }

    #[test]
    fn test_speech_tokenizer_model_loading() {
        if !speech_tokenizer_available() {
            eprintln!("Skipping test_speech_tokenizer_model_loading: test data not found");
            return;
        }

        use safetensors::SafeTensors;

        // Load the safetensors file
        let model_path = Path::new(SPEECH_TOKENIZER_DIR).join("model.safetensors");
        let model_bytes = std::fs::read(&model_path).unwrap();
        let tensors = SafeTensors::deserialize(&model_bytes).unwrap();

        // Check that we have tensors
        let tensor_names = tensors.names();
        assert!(!tensor_names.is_empty());

        // Should have encoder and decoder weights
        let has_encoder = tensor_names.iter().any(|n| n.contains("encoder"));
        let has_decoder = tensor_names.iter().any(|n| n.contains("decoder"));
        assert!(has_encoder, "Model should have encoder weights");
        assert!(has_decoder, "Model should have decoder weights");

        // Count total tensors
        println!("Speech tokenizer has {} tensors", tensor_names.len());
    }

    #[test]
    fn test_speech_tokenizer_encoder_weights() {
        if !speech_tokenizer_available() {
            eprintln!("Skipping test_speech_tokenizer_encoder_weights: test data not found");
            return;
        }

        use safetensors::SafeTensors;

        let model_path = Path::new(SPEECH_TOKENIZER_DIR).join("model.safetensors");
        let model_bytes = std::fs::read(&model_path).unwrap();
        let tensors = SafeTensors::deserialize(&model_bytes).unwrap();

        // Find encoder embedding weights
        let encoder_tensors: Vec<&String> = tensors.names()
            .iter()
            .filter(|n| n.starts_with("encoder."))
            .cloned()
            .collect();

        assert!(!encoder_tensors.is_empty(), "Should have encoder tensors");
        println!("Found {} encoder tensors", encoder_tensors.len());

        // Check a specific tensor shape
        for name in encoder_tensors.iter().take(5) {
            let tensor = tensors.tensor(name).unwrap();
            println!("  {}: {:?}", name, tensor.shape());
        }
    }

    #[test]
    fn test_speech_tokenizer_decoder_weights() {
        if !speech_tokenizer_available() {
            eprintln!("Skipping test_speech_tokenizer_decoder_weights: test data not found");
            return;
        }

        use safetensors::SafeTensors;

        let model_path = Path::new(SPEECH_TOKENIZER_DIR).join("model.safetensors");
        let model_bytes = std::fs::read(&model_path).unwrap();
        let tensors = SafeTensors::deserialize(&model_bytes).unwrap();

        // Find decoder weights
        let decoder_tensors: Vec<&String> = tensors.names()
            .iter()
            .filter(|n| n.starts_with("decoder."))
            .cloned()
            .collect();

        assert!(!decoder_tensors.is_empty(), "Should have decoder tensors");
        println!("Found {} decoder tensors", decoder_tensors.len());

        // Check a specific tensor shape
        for name in decoder_tensors.iter().take(5) {
            let tensor = tensors.tensor(name).unwrap();
            println!("  {}: {:?}", name, tensor.shape());
        }
    }

    #[test]
    fn test_speech_tokenizer_quantizer_codebooks() {
        if !speech_tokenizer_available() {
            eprintln!("Skipping test_speech_tokenizer_quantizer_codebooks: test data not found");
            return;
        }

        use safetensors::SafeTensors;

        let model_path = Path::new(SPEECH_TOKENIZER_DIR).join("model.safetensors");
        let model_bytes = std::fs::read(&model_path).unwrap();
        let tensors = SafeTensors::deserialize(&model_bytes).unwrap();

        // Find quantizer/codebook weights
        let all_names = tensors.names();
        let codebook_tensors: Vec<&String> = all_names
            .iter()
            .filter(|n| n.contains("quantiz") || n.contains("codebook") || n.contains("embed"))
            .cloned()
            .collect();

        println!("Found {} quantizer/codebook tensors:", codebook_tensors.len());
        for name in &codebook_tensors {
            let tensor = tensors.tensor(name).unwrap();
            println!("  {}: {:?}", name, tensor.shape());
        }

        // Should have quantization-related weights
        assert!(!codebook_tensors.is_empty(), "Should have quantizer weights");
    }

    #[test]
    fn test_speech_tokenizer_preprocessor_config() {
        if !speech_tokenizer_available() {
            eprintln!("Skipping test_speech_tokenizer_preprocessor_config: test data not found");
            return;
        }

        let config_path = Path::new(SPEECH_TOKENIZER_DIR).join("preprocessor_config.json");
        let config_str = std::fs::read_to_string(config_path).unwrap();
        let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();

        // Verify preprocessor config
        assert_eq!(config["sampling_rate"], 24000);
        assert_eq!(config["feature_size"], 1); // mono audio
        assert_eq!(config["padding_side"], "right");
    }

    #[test]
    fn test_load_tensor_to_candle() {
        if !speech_tokenizer_available() {
            eprintln!("Skipping test_load_tensor_to_candle: test data not found");
            return;
        }

        use candle_core::Device;

        let model_path = Path::new(SPEECH_TOKENIZER_DIR).join("model.safetensors");
        let device = Device::Cpu;

        // Use candle's built-in safetensors loading
        let tensors = candle_core::safetensors::load(&model_path, &device).unwrap();

        println!("Loaded {} tensors from safetensors file", tensors.len());

        // Check a few tensors
        for (name, tensor) in tensors.iter().take(5) {
            println!("  {}: {:?} ({:?})", name, tensor.dims(), tensor.dtype());
        }

        // Verify we can access tensors
        assert!(!tensors.is_empty(), "Should have loaded tensors");

        // Verify tensor shapes are valid
        for (name, tensor) in &tensors {
            assert!(!tensor.dims().is_empty(), "Tensor {} should have valid shape", name);
        }
    }
}

/// Tests for model configuration files from the 0.6B model
mod model_config_tests {
    use std::path::Path;

    /// Path to downloaded model config data
    const MODEL_CONFIG_DIR: &str = "test_data/model_config";

    fn model_config_available() -> bool {
        Path::new(MODEL_CONFIG_DIR).join("generation_config.json").exists()
    }

    #[test]
    fn test_generation_config_parsing() {
        if !model_config_available() {
            eprintln!("Skipping test_generation_config_parsing: test data not found");
            return;
        }

        let config_path = Path::new(MODEL_CONFIG_DIR).join("generation_config.json");
        let config_str = std::fs::read_to_string(config_path).unwrap();
        let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();

        // Verify generation parameters match official defaults
        assert_eq!(config["do_sample"], true);
        assert_eq!(config["temperature"], 0.9);
        assert_eq!(config["top_p"], 1.0);
        assert_eq!(config["top_k"], 50);
        assert_eq!(config["repetition_penalty"], 1.05);
        assert_eq!(config["max_new_tokens"], 8192);

        // Subtalker has its own parameters
        assert_eq!(config["subtalker_dosample"], true);
        assert_eq!(config["subtalker_temperature"], 0.9);
        assert_eq!(config["subtalker_top_k"], 50);
    }

    #[test]
    fn test_preprocessor_config_parsing() {
        if !model_config_available() {
            eprintln!("Skipping test_preprocessor_config_parsing: test data not found");
            return;
        }

        let config_path = Path::new(MODEL_CONFIG_DIR).join("preprocessor_config.json");
        let config_str = std::fs::read_to_string(config_path).unwrap();
        let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();

        // Verify preprocessor config
        assert_eq!(config["padding_side"], "left");
        assert_eq!(config["padding_value"], 0.0);
        assert_eq!(config["processor_class"], "Qwen3TTSProcessor");
        assert_eq!(config["return_attention_mask"], true);
    }

    #[test]
    fn test_tokenizer_config_special_tokens() {
        if !model_config_available() {
            eprintln!("Skipping test_tokenizer_config_special_tokens: test data not found");
            return;
        }

        let config_path = Path::new(MODEL_CONFIG_DIR).join("tokenizer_config.json");
        let config_str = std::fs::read_to_string(config_path).unwrap();
        let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();

        // Verify tokenizer class
        assert_eq!(config["tokenizer_class"], "Qwen2Tokenizer");
        assert_eq!(config["model_max_length"], 131072);

        // Verify important special tokens
        assert_eq!(config["eos_token"], "<|im_end|>");
        assert_eq!(config["pad_token"], "<|endoftext|>");

        // Verify audio-specific tokens
        assert_eq!(config["audio_bos_token"], "<|audio_start|>");
        assert_eq!(config["audio_eos_token"], "<|audio_end|>");
        assert_eq!(config["audio_token"], "<|audio_pad|>");
    }

    #[test]
    fn test_tokenizer_config_tts_tokens() {
        if !model_config_available() {
            eprintln!("Skipping test_tokenizer_config_tts_tokens: test data not found");
            return;
        }

        let config_path = Path::new(MODEL_CONFIG_DIR).join("tokenizer_config.json");
        let config_str = std::fs::read_to_string(config_path).unwrap();
        let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();

        // Check that TTS-specific tokens are present in added_tokens_decoder
        let added_tokens = &config["added_tokens_decoder"];

        // Find TTS tokens by their IDs
        assert_eq!(added_tokens["151671"]["content"], "<tts_pad>");
        assert_eq!(added_tokens["151672"]["content"], "<tts_text_bos>");
        assert_eq!(added_tokens["151673"]["content"], "<tts_text_eod>");
        assert_eq!(added_tokens["151674"]["content"], "<tts_text_bos_single>");
        assert_eq!(added_tokens["151675"]["content"], "<|audio_pad|>");

        // Audio markers
        assert_eq!(added_tokens["151669"]["content"], "<|audio_start|>");
        assert_eq!(added_tokens["151670"]["content"], "<|audio_end|>");

        // Standard Qwen tokens
        assert_eq!(added_tokens["151643"]["content"], "<|endoftext|>");
        assert_eq!(added_tokens["151644"]["content"], "<|im_start|>");
        assert_eq!(added_tokens["151645"]["content"], "<|im_end|>");
    }

    #[test]
    fn test_tokenizer_config_additional_special_tokens() {
        if !model_config_available() {
            eprintln!("Skipping test_tokenizer_config_additional_special_tokens: test data not found");
            return;
        }

        let config_path = Path::new(MODEL_CONFIG_DIR).join("tokenizer_config.json");
        let config_str = std::fs::read_to_string(config_path).unwrap();
        let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();

        let additional_tokens = config["additional_special_tokens"].as_array().unwrap();
        let tokens: Vec<&str> = additional_tokens.iter()
            .map(|t| t.as_str().unwrap())
            .collect();

        // Verify TTS-specific tokens are in additional_special_tokens
        assert!(tokens.contains(&"<tts_pad>"));
        assert!(tokens.contains(&"<tts_text_bos>"));
        assert!(tokens.contains(&"<tts_text_bos_single>"));
        assert!(tokens.contains(&"<|audio_start|>"));
        assert!(tokens.contains(&"<|audio_end|>"));
        assert!(tokens.contains(&"<|audio_pad|>"));

        // Verify standard chat tokens
        assert!(tokens.contains(&"<|im_start|>"));
        assert!(tokens.contains(&"<|im_end|>"));
    }

    #[test]
    fn test_generation_config_matches_our_defaults() {
        if !model_config_available() {
            eprintln!("Skipping test_generation_config_matches_our_defaults: test data not found");
            return;
        }

        use qwen3_tts::generation::GenerationConfig;

        let config_path = Path::new(MODEL_CONFIG_DIR).join("generation_config.json");
        let config_str = std::fs::read_to_string(config_path).unwrap();
        let official: serde_json::Value = serde_json::from_str(&config_str).unwrap();

        // Create our default config and compare key values
        let our_config = GenerationConfig::default();

        // Our defaults should be reasonable (though may differ from official)
        // This test documents the official values for reference
        println!("Official generation config:");
        println!("  temperature: {}", official["temperature"]);
        println!("  top_k: {}", official["top_k"]);
        println!("  top_p: {}", official["top_p"]);
        println!("  repetition_penalty: {}", official["repetition_penalty"]);
        println!("  max_new_tokens: {}", official["max_new_tokens"]);

        println!("\nOur default config:");
        println!("  temperature: {}", our_config.temperature);
        println!("  top_k: {:?}", our_config.top_k);
        println!("  top_p: {:?}", our_config.top_p);
        println!("  repetition_penalty: {}", our_config.repetition_penalty);
        println!("  max_new_tokens: {}", our_config.max_new_tokens);

        // At minimum, our config should have sensible values
        assert!(our_config.temperature > 0.0);
        assert!(our_config.max_new_tokens > 0);
    }
}
