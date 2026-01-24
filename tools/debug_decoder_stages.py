#!/usr/bin/env python3
"""Debug decoder stages to get reference intermediate values."""

import sys
import types
import importlib.util
import torch
import numpy as np
from pathlib import Path
from safetensors.torch import load_file

# Set up module paths
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

def print_stats(name, tensor):
    """Print min/max/mean stats for a tensor."""
    t = tensor.detach().float()
    print(f"DEBUG: {name} stats: min={t.min().item():.4f}, max={t.max().item():.4f}, mean={t.mean().item():.4f}")

def main():
    model_dir = "../test_data/speech_tokenizer"
    codes_file = "../test_data/rust_codes.npy"

    print(f"Loading decoder from {model_dir}")
    config = Qwen3TTSTokenizerV2Config.from_pretrained(model_dir)
    model = Qwen3TTSTokenizerV2Model.from_pretrained(
        model_dir,
        config=config,
        dtype=torch.float32,
    )
    model.eval()

    # Load and prepare codes
    codes = np.load(codes_file)
    print(f"Loaded codes: shape {codes.shape}")

    # Filter invalid semantic tokens
    semantic_tokens = codes[:, 0] if codes.ndim == 2 else codes[0, :, 0]
    if codes.ndim == 2:
        codes = codes.T[np.newaxis, ...]  # [1, seq, 16]

    codes_tensor = torch.tensor(codes, dtype=torch.long)
    print(f"Codes tensor shape: {codes_tensor.shape}")

    # We need to trace through the decoder with hooks
    decoder = model.decoder

    with torch.no_grad():
        print("\n--- Tracing with hooks ---")

        captured = {}

        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    out = output[0]
                else:
                    out = output
                captured[name] = out.detach().clone()
                print_stats(name, out)
            return hook

        def make_input_hook(name):
            def hook(module, input, output):
                inp = input[0] if isinstance(input, tuple) else input
                print_stats(f"{name}_input", inp)
            return hook

        # Register hooks on key layers
        hooks = []

        # quantizer
        hooks.append(decoder.quantizer.register_forward_hook(make_hook("quantizer")))

        # pre_conv - also capture input
        hooks.append(decoder.pre_conv.register_forward_hook(make_input_hook("pre_conv")))
        hooks.append(decoder.pre_conv.register_forward_hook(make_hook("pre_conv")))

        # pre_transformer input_proj
        hooks.append(decoder.pre_transformer.input_proj.register_forward_hook(make_hook("input_proj")))

        # pre_transformer norm (final)
        hooks.append(decoder.pre_transformer.norm.register_forward_hook(make_hook("transformer_norm")))

        # pre_transformer output_proj
        hooks.append(decoder.pre_transformer.output_proj.register_forward_hook(make_hook("output_proj")))

        # upsample stages
        for i, stage in enumerate(decoder.upsample):
            if hasattr(stage, '__iter__'):
                # It's a sequence
                for j, layer in enumerate(stage):
                    hooks.append(layer.register_forward_hook(make_hook(f"upsample.{i}.{j}")))
            else:
                hooks.append(stage.register_forward_hook(make_hook(f"upsample.{i}")))

        # decoder.decoder (the actual decoder blocks)
        for i, block in enumerate(decoder.decoder):
            hooks.append(block.register_forward_hook(make_hook(f"decoder.{i}")))

        # Run decode
        try:
            output = model.decode(codes_tensor)
        finally:
            # Remove hooks
            for h in hooks:
                h.remove()

        # Get audio
        if hasattr(output, 'audio_values'):
            audio = output.audio_values
            if isinstance(audio, list):
                audio = audio[0]
        else:
            audio = output

        print_stats("final_audio", audio)

if __name__ == "__main__":
    main()
