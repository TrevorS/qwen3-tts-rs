#!/usr/bin/env python3
"""Regenerate decoder reference audio using official Qwen3-TTS model."""

import struct
import sys
import os

# Get script directory and set base path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # qwen3-tts-rs directory
os.chdir(BASE_DIR)

# Add parent Qwen3-TTS to path
QWEN_TTS_DIR = os.path.dirname(BASE_DIR)  # Qwen3-TTS directory
sys.path.insert(0, QWEN_TTS_DIR)

import torch
import numpy as np

def main():
    device = "cpu"

    # Load codes from binary file
    print("Loading codes...")
    with open("test_data/rust_audio_final/codes_seed42_frames75.bin", "rb") as f:
        codes_bytes = f.read()

    # Convert bytes to int64
    codes_i64 = []
    for i in range(0, len(codes_bytes), 8):
        val = struct.unpack('<q', codes_bytes[i:i+8])[0]
        codes_i64.append(val)

    num_frames = len(codes_i64) // 16
    print(f"Num frames: {num_frames}")

    # Reshape to [batch, num_quantizers, seq_len]
    codes = torch.tensor(codes_i64, dtype=torch.long).reshape(num_frames, 16).T.unsqueeze(0)
    codes = codes.to(device)
    print(f"Codes shape: {codes.shape}")

    # Try loading official Qwen3-TTS tokenizer
    try:
        from qwen_tts.core.tokenizer_12hz import Qwen3TTSTokenizerV2Model

        print("Loading official Qwen3-TTS tokenizer...")
        model = Qwen3TTSTokenizerV2Model.from_pretrained("test_data/speech_tokenizer")
        model = model.to(device)
        model.eval()

        # Decode
        print("Decoding...")
        with torch.no_grad():
            audio = model.decode(codes)

        print(f"Audio shape: {audio.shape}")
        print(f"Audio samples: {audio.numel()}")
        print(f"Audio mean: {audio.mean().item():.6f}")

        # Save as binary
        audio_np = audio.squeeze().cpu().numpy().astype(np.float32)
        with open("test_data/python_decoder_audio.bin", "wb") as f:
            f.write(audio_np.tobytes())

        print(f"Saved {len(audio_np)} samples to test_data/python_decoder_audio.bin")

    except ImportError as e:
        print(f"Import error: {e}")
        print("\nPlease install qwen_tts: cd /home/trevor/Projects/Qwen3-TTS && pip install -e .")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
