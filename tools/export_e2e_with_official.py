#!/usr/bin/env python3
"""Export E2E reference using the official Qwen3-TTS decoder."""

import sys
import types
import importlib.util
import torch
import numpy as np
from pathlib import Path

# Set up module paths to avoid problematic imports
sys.path.insert(0, '/home/trevor/Projects/Qwen3-TTS')

# Create fake parent modules
sys.modules['qwen_tts'] = types.ModuleType('qwen_tts')
sys.modules['qwen_tts.core'] = types.ModuleType('qwen_tts.core')
sys.modules['qwen_tts.core.tokenizer_12hz'] = types.ModuleType('qwen_tts.core.tokenizer_12hz')

# Load configuration module
spec_config = importlib.util.spec_from_file_location(
    'qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2',
    '/home/trevor/Projects/Qwen3-TTS/qwen_tts/core/tokenizer_12hz/configuration_qwen3_tts_tokenizer_v2.py'
)
config_module = importlib.util.module_from_spec(spec_config)
sys.modules['qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2'] = config_module
spec_config.loader.exec_module(config_module)

# Load modeling module
spec_model = importlib.util.spec_from_file_location(
    'qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2',
    '/home/trevor/Projects/Qwen3-TTS/qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py'
)
model_module = importlib.util.module_from_spec(spec_model)
sys.modules['qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2'] = model_module
spec_model.loader.exec_module(model_module)

Qwen3TTSTokenizerV2Config = config_module.Qwen3TTSTokenizerV2Config
Qwen3TTSTokenizerV2Model = model_module.Qwen3TTSTokenizerV2Model


def main():
    output_dir = Path("test_data/reference_values")
    model_dir = Path("test_data/speech_tokenizer")

    print(f"Loading official decoder from {model_dir}")

    # Load config and model
    config = Qwen3TTSTokenizerV2Config.from_pretrained(str(model_dir))
    model = Qwen3TTSTokenizerV2Model.from_pretrained(
        str(model_dir),
        config=config,
        dtype=torch.float32,
    )
    model.eval()
    print("Decoder loaded!")

    # E2E pipeline codes (from the pipeline)
    # These are the codes produced by running talker + code predictor
    codes_list = [963, 281, 1899, 1174, 221, 1049, 280, 1863, 1317, 82, 604, 1300, 1030, 342, 165, 1750]

    print(f"Using codes: {codes_list}")

    # Shape: [batch=1, seq_len=1, num_quantizers=16]
    codes = torch.tensor(codes_list, dtype=torch.long).unsqueeze(0).unsqueeze(0)
    print(f"Codes tensor shape: {codes.shape}")

    # Decode using official model
    print("Decoding with official model...")
    with torch.no_grad():
        output = model.decode(codes)

    # Get audio
    if hasattr(output, 'audio_values'):
        audio = output.audio_values
        if isinstance(audio, list):
            audio = audio[0]
    else:
        audio = output

    # Ensure tensor
    if isinstance(audio, list):
        audio = torch.cat([torch.tensor(a) for a in audio])

    print(f"Audio shape: {audio.shape}")
    print(f"Audio mean: {audio.mean().item():.6f}")
    print(f"Audio min: {audio.min().item():.6f}")
    print(f"Audio max: {audio.max().item():.6f}")

    # Reshape to [batch, channels, samples] if needed
    if audio.ndim == 1:
        audio = audio.unsqueeze(0).unsqueeze(0)
    elif audio.ndim == 2:
        audio = audio.unsqueeze(0)

    print(f"Final audio shape: {audio.shape}")

    # Save as binary
    output_path = output_dir / "e2e_audio.bin"
    np.array(audio.detach().cpu().numpy(), dtype=np.float32).tofile(output_path)
    print(f"Saved {output_path}: {audio.shape}")

    # Also generate decoder_output.bin with zeros (for test_full_decoder_12hz)
    # Use raw decoder to bypass model.decode()'s zero-filtering
    # (model.decode filters output based on semantic_token > 0)
    print("\n=== Generating decoder_output.bin with zeros ===")

    # Decoder expects [batch, num_quantizers, seq_len] = [1, 16, 2]
    codes_zeros = torch.zeros((1, 16, 2), dtype=torch.long)
    print(f"Zero codes tensor shape: {codes_zeros.shape} [batch, num_quantizers, seq_len]")

    with torch.no_grad():
        # Use raw decoder.chunked_decode() directly
        audio_zeros = model.decoder.chunked_decode(codes_zeros).squeeze(1)

    print(f"Raw decoder output shape: {audio_zeros.shape}")

    if audio_zeros.ndim == 1:
        audio_zeros = audio_zeros.unsqueeze(0).unsqueeze(0)
    elif audio_zeros.ndim == 2:
        audio_zeros = audio_zeros.unsqueeze(0)

    print(f"Zero audio shape: {audio_zeros.shape}")
    print(f"Zero audio mean: {audio_zeros.mean().item():.6f}")

    output_path_zeros = output_dir / "decoder_output.bin"
    np.array(audio_zeros.detach().cpu().numpy(), dtype=np.float32).tofile(output_path_zeros)
    print(f"Saved {output_path_zeros}: {audio_zeros.shape}")


if __name__ == "__main__":
    main()
