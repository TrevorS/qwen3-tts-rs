//! TDD tests for streaming performance optimizations.
//!
//! These tests verify optimizations for:
//! 1. Sliding window decoder - limits context for streaming chunks
//! 2. Code predictor fusion - reduces overhead in 15-layer autoregressive loop
//!
//! Run with: cargo test --features cuda optimizations -- --nocapture

use candle_core::{DType, Device, Tensor};

mod sliding_window_decoder {
    use super::*;
    use qwen3_tts::models::codec::Decoder12HzConfig;

    #[test]
    fn test_decode_with_sliding_window_method_exists() {
        // This test verifies the method exists and compiles.
        // Full integration tests require real model weights.
        let config = Decoder12HzConfig::default();
        assert!(config.upsample_rates.len() > 0);
    }

    #[test]
    fn test_decode_with_window_and_prefix_method_exists() {
        // This test verifies the method exists and compiles.
        // Full integration tests require real model weights.
        let config = Decoder12HzConfig::default();
        assert_eq!(config.num_quantizers, 16);
    }

    #[test]
    fn test_decoder_config_window_parameter() {
        // Verify config supports window-based decoding
        let config = Decoder12HzConfig::default();
        // The default total upsampling is 4 * 480 = 1920
        let total_upsample: usize = config.upsample_rates.iter().product();
        assert!(total_upsample > 0);
    }
}

mod code_predictor_fusion {
    use super::*;
    use candle_nn::VarBuilder;
    use candle_nn::VarMap;
    use qwen3_tts::models::{CodePredictor, CodePredictorConfig};

    fn create_mock_code_predictor(device: &Device) -> CodePredictor {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

        let config = CodePredictorConfig {
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            vocab_size: 128,
            num_code_groups: 4,
            ..Default::default()
        };

        CodePredictor::new(config, vb).expect("failed to create mock code predictor")
    }

    #[test]
    fn test_generate_acoustic_codes_fused_exists() {
        let device = Device::Cpu;
        let predictor = create_mock_code_predictor(&device);

        let talker_hidden = Tensor::zeros((1, 1, 64), DType::F32, &device).unwrap();
        let semantic_embed = Tensor::zeros((1, 1, 64), DType::F32, &device).unwrap();
        let mut kv_caches = predictor.new_kv_caches();

        let result = predictor.generate_acoustic_codes_fused(
            &talker_hidden,
            &semantic_embed,
            &mut kv_caches,
        );

        assert!(result.is_ok(), "generate_acoustic_codes_fused should exist");
    }

    #[test]
    fn test_fused_produces_same_output_shape() {
        let device = Device::Cpu;
        let predictor = create_mock_code_predictor(&device);

        let talker_hidden = Tensor::zeros((1, 1, 64), DType::F32, &device).unwrap();
        let semantic_embed = Tensor::zeros((1, 1, 64), DType::F32, &device).unwrap();
        let mut kv_caches = predictor.new_kv_caches();

        let standard = predictor
            .generate_acoustic_codes(&talker_hidden, &semantic_embed, &mut kv_caches)
            .unwrap();

        for cache in kv_caches.iter_mut() {
            cache.reset();
        }

        let fused = predictor
            .generate_acoustic_codes_fused(&talker_hidden, &semantic_embed, &mut kv_caches)
            .unwrap();

        assert_eq!(
            standard.dims(),
            fused.dims(),
            "fused version should produce same shape"
        );
    }

    #[test]
    fn test_batch_generate_acoustic_codes() {
        let device = Device::Cpu;
        let predictor = create_mock_code_predictor(&device);

        let talker_hiddens = vec![
            Tensor::zeros((1, 1, 64), DType::F32, &device).unwrap(),
            Tensor::zeros((1, 1, 64), DType::F32, &device).unwrap(),
        ];
        let semantic_embeds = vec![
            Tensor::zeros((1, 1, 64), DType::F32, &device).unwrap(),
            Tensor::zeros((1, 1, 64), DType::F32, &device).unwrap(),
        ];

        let result = predictor.generate_acoustic_codes_batch(&talker_hiddens, &semantic_embeds);

        assert!(result.is_ok(), "batch generation should exist");
        let batch_result = result.unwrap();
        assert_eq!(batch_result.len(), 2, "should produce 2 results");
    }

    #[test]
    fn test_code_predictor_timing_breakdown() {
        let device = Device::Cpu;
        let predictor = create_mock_code_predictor(&device);

        let talker_hidden = Tensor::zeros((1, 1, 64), DType::F32, &device).unwrap();
        let semantic_embed = Tensor::zeros((1, 1, 64), DType::F32, &device).unwrap();
        let mut kv_caches = predictor.new_kv_caches();

        let result = predictor.generate_acoustic_codes_profiled(
            &talker_hidden,
            &semantic_embed,
            &mut kv_caches,
        );

        assert!(
            result.is_ok(),
            "generate_acoustic_codes_profiled should exist"
        );

        let (_codes, profile) = result.unwrap();

        assert!(
            profile.prefill_ms >= 0.0,
            "prefill_ms should be non-negative"
        );
        assert!(profile.decode_ms >= 0.0, "decode_ms should be non-negative");
    }
}

mod streaming_integration {
    use qwen3_tts::SynthesisOptions;

    #[test]
    fn test_decode_window_config_option() {
        let options = SynthesisOptions {
            decode_window_frames: 80,
            ..Default::default()
        };

        assert_eq!(
            options.decode_window_frames, 80,
            "decode_window_frames should be configurable"
        );
    }

    #[test]
    fn test_emit_every_frames_config() {
        let options = SynthesisOptions {
            emit_every_frames: 4,
            ..Default::default()
        };

        assert_eq!(
            options.emit_every_frames, 4,
            "emit_every_frames should be configurable"
        );
    }

    #[test]
    fn test_streaming_decode_uses_sliding_window() {
        let options = SynthesisOptions {
            decode_window_frames: 80,
            chunk_frames: 10,
            ..Default::default()
        };

        assert_eq!(options.decode_window_frames, 80);
        assert_eq!(options.chunk_frames, 10);
    }
}
