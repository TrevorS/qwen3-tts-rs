#!/usr/bin/env python3
"""Decode acoustic codes using the official Qwen3-TTS decoder."""

import sys
import types
import importlib.util
import torch
import numpy as np
from pathlib import Path
import scipy.io.wavfile as wavfile
from safetensors.torch import load_file

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
    import argparse
    parser = argparse.ArgumentParser(description="Decode acoustic codes using official decoder")
    parser.add_argument("--codes-file", type=str, help="Path to numpy file with acoustic codes")
    parser.add_argument("--model-dir", type=str, default="../test_data/model_customvoice/speech_tokenizer")
    parser.add_argument("--output", type=str, default="../test_data/decoded_audio.wav")
    args = parser.parse_args()

    print(f"Loading decoder from {args.model_dir}")

    # Load config and model
    config = Qwen3TTSTokenizerV2Config.from_pretrained(args.model_dir)
    model = Qwen3TTSTokenizerV2Model.from_pretrained(
        args.model_dir,
        config=config,
        dtype=torch.float32,
    )
    model.eval()
    print("Decoder loaded!")

    if args.codes_file:
        # Load codes from file
        codes = np.load(args.codes_file)
        print(f"Loaded codes from {args.codes_file}: shape {codes.shape}")
    else:
        # Use test codes - these should be [batch, seq_len, num_quantizers]
        # The model expects codes with shape [batch, seq_len, 16]
        print("No codes file provided, using test codes")
        codes = np.array([
            [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600],
        ] * 50)  # 50 frames
        print(f"Test codes shape: {codes.shape}")

    # Filter out frames with invalid semantic tokens (>= 2048 are special tokens like thinking tokens)
    # The codebook size is 2048, so semantic tokens must be in [0, 2047]
    semantic_tokens = codes[:, 0]
    valid_mask = semantic_tokens < 2048
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        print(f"Filtering out {invalid_count} frames with invalid semantic tokens >= 2048")
        codes = codes[valid_mask]
        print(f"Filtered codes shape: {codes.shape}")

    # Ensure codes are in the right shape [batch, seq_len, num_quantizers]
    if codes.ndim == 2:
        # codes could be [seq_len, num_quantizers] or [num_quantizers, seq_len]
        # If shape[1] == 16 (num_quantizers), it's already [seq_len, 16]
        if codes.shape[1] == 16:
            codes = codes[np.newaxis, ...]  # Just add batch dimension
        else:
            # codes is [num_quantizers, seq_len], transpose to [seq_len, num_quantizers]
            codes = codes.T[np.newaxis, ...]

    # Convert to tensor
    codes_tensor = torch.tensor(codes, dtype=torch.long)
    print(f"Codes tensor shape: {codes_tensor.shape}")

    # Decode
    print("Decoding...")
    with torch.no_grad():
        # The decoder expects codes with semantic token in index 0
        output = model.decode(codes_tensor)

    # The output is a Qwen3TTSTokenizerV2DecoderOutput object
    print(f"Output type: {type(output)}")
    print(f"Output keys: {output.keys() if hasattr(output, 'keys') else dir(output)}")

    # Get the audio values
    if hasattr(output, 'audio_values'):
        audio = output.audio_values
        # audio_values might be a list
        if isinstance(audio, list):
            print(f"audio_values is a list of {len(audio)} elements")
            audio = audio[0]  # Get first element
    elif isinstance(output, torch.Tensor):
        audio = output
    else:
        # Try to access as attribute
        audio = output[0] if hasattr(output, '__getitem__') else output

    # Convert to tensor if needed
    if isinstance(audio, list):
        audio = torch.cat([torch.tensor(a) for a in audio])

    print(f"Audio shape: {audio.shape}")
    print(f"Audio range: [{audio.min():.3f}, {audio.max():.3f}]")

    # Save as WAV
    audio_np = audio.squeeze().numpy()
    audio_int16 = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)
    wavfile.write(args.output, 24000, audio_int16)
    print(f"Saved audio to {args.output}")


if __name__ == "__main__":
    main()
