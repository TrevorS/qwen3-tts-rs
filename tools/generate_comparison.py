#!/usr/bin/env python3
"""Generate audio from Python (official model) for comparison with Rust."""

import sys
import json
import numpy as np
import torch
import scipy.io.wavfile as wav
from pathlib import Path
from safetensors.torch import load_file

# Set up paths
sys.path.insert(0, '/home/trevor/Projects/Qwen3-TTS')


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def rms_norm(x, weight, eps=1e-6):
    variance = x.pow(2).mean(-1, keepdim=True)
    return weight * x * torch.rsqrt(variance + eps)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    x = x[:, :, None, :, :].expand(b, h, n_rep, s, d)
    return x.reshape(b, h * n_rep, s, d)


def build_rope(seq_len, head_dim, theta=1000000.0):
    positions = torch.arange(seq_len)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = torch.outer(positions.float(), inv_freq)
    cos = freqs.cos().repeat(1, 2).unsqueeze(0).unsqueeze(0)
    sin = freqs.sin().repeat(1, 2).unsqueeze(0).unsqueeze(0)
    return cos, sin


def sample_token(logits, temperature=0.7, top_k=50):
    # Suppress tokens >= 2048 (decoder codebook size)
    logits = logits.clone()
    logits[2048:] = float('-inf')

    if temperature < 0.01:
        return logits.argmax().item()

    logits = logits / temperature
    if top_k > 0:
        values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(-1, indices, values)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()


def run_transformer_layer(hidden, weights, prefix, cos, sin, causal_mask,
                          num_heads, num_kv_heads, head_dim, eps=1e-6):
    batch, seq_len, _ = hidden.shape
    n_rep = num_heads // num_kv_heads
    scaling = head_dim ** -0.5

    # Input LayerNorm
    normed = rms_norm(hidden, weights[f"{prefix}.input_layernorm.weight"], eps)

    # QKV
    q = torch.nn.functional.linear(normed, weights[f"{prefix}.self_attn.q_proj.weight"])
    k = torch.nn.functional.linear(normed, weights[f"{prefix}.self_attn.k_proj.weight"])
    v = torch.nn.functional.linear(normed, weights[f"{prefix}.self_attn.v_proj.weight"])

    q = q.view(batch, seq_len, num_heads, head_dim)
    k = k.view(batch, seq_len, num_kv_heads, head_dim)
    v = v.view(batch, seq_len, num_kv_heads, head_dim)

    q = rms_norm(q, weights[f"{prefix}.self_attn.q_norm.weight"], eps)
    k = rms_norm(k, weights[f"{prefix}.self_attn.k_norm.weight"], eps)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # RoPE
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)

    # Attention
    k_exp = repeat_kv(k, n_rep)
    v_exp = repeat_kv(v, n_rep)

    attn = torch.matmul(q, k_exp.transpose(-2, -1)) * scaling
    if causal_mask is not None:
        attn = attn + causal_mask
    attn = torch.softmax(attn, dim=-1, dtype=torch.float32)
    out = torch.matmul(attn, v_exp)

    out = out.transpose(1, 2).reshape(batch, seq_len, num_heads * head_dim)
    out = torch.nn.functional.linear(out, weights[f"{prefix}.self_attn.o_proj.weight"])
    hidden = hidden + out

    # MLP
    normed = rms_norm(hidden, weights[f"{prefix}.post_attention_layernorm.weight"], eps)
    gate = torch.nn.functional.linear(normed, weights[f"{prefix}.mlp.gate_proj.weight"])
    up = torch.nn.functional.linear(normed, weights[f"{prefix}.mlp.up_proj.weight"])
    mlp_out = torch.nn.functional.linear(torch.nn.functional.silu(gate) * up,
                                          weights[f"{prefix}.mlp.down_proj.weight"])
    hidden = hidden + mlp_out

    return hidden


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    # Use absolute paths based on script location
    script_dir = Path(__file__).parent.parent  # qwen3-tts-rs directory

    output_dir = Path(args.output_dir) if args.output_dir else script_dir / "test_data" / "comparison"
    output_dir.mkdir(exist_ok=True, parents=True)

    model_dir = Path(args.model_dir) if args.model_dir else script_dir / "test_data" / "model"

    print(f"Text: {args.text}")
    print(f"Seed: {args.seed}")
    print(f"Frames: {args.frames}")
    print(f"Temperature: {args.temperature}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    from transformers import PreTrainedTokenizerFast
    tokenizer_path = script_dir / "test_data" / "tokenizer" / "tokenizer.json"
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))

    tokens = tokenizer.encode(args.text, add_special_tokens=False)
    print(f"Token IDs ({len(tokens)} tokens): {tokens}")

    # Load weights
    print("\nLoading model weights...")
    weights = load_file(str(model_dir / "model.safetensors"))
    weights = {k: v.float() for k, v in weights.items()}

    # Config
    num_heads = 16
    num_kv_heads = 8
    head_dim = 128
    num_layers = 28
    eps = 1e-6

    # Text embedding
    text_embed_w = weights["talker.model.text_embedding.weight"]
    input_ids = torch.tensor([tokens], dtype=torch.long)
    hidden = torch.nn.functional.embedding(input_ids, text_embed_w)

    # Text projection
    fc1_w = weights["talker.text_projection.linear_fc1.weight"]
    fc1_b = weights["talker.text_projection.linear_fc1.bias"]
    fc2_w = weights["talker.text_projection.linear_fc2.weight"]
    fc2_b = weights["talker.text_projection.linear_fc2.bias"]
    hidden = torch.nn.functional.linear(hidden, fc1_w, fc1_b)
    hidden = torch.nn.functional.silu(hidden)
    hidden = torch.nn.functional.linear(hidden, fc2_w, fc2_b)

    print(f"After text projection: {hidden.shape}")

    # Build RoPE and causal mask
    seq_len = hidden.shape[1]
    cos, sin = build_rope(seq_len, head_dim)
    causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)

    # Run through talker layers
    print("Running talker (28 layers)...")
    for i in range(num_layers):
        hidden = run_transformer_layer(
            hidden, weights, f"talker.model.layers.{i}",
            cos, sin, causal_mask, num_heads, num_kv_heads, head_dim, eps
        )
        if i % 7 == 0:
            print(f"  Layer {i}: mean={hidden.mean().item():.6f}")

    # Final norm
    hidden = rms_norm(hidden, weights["talker.model.norm.weight"], eps)

    # Get logits for first semantic token
    logits = torch.nn.functional.linear(hidden[:, -1:, :], weights["talker.codec_head.weight"])
    first_token = sample_token(logits[0, 0], args.temperature)
    print(f"\nFirst semantic token: {first_token}")

    # Generate frames
    all_codes = []
    codec_embed_w = weights["talker.model.codec_embedding.weight"]

    # Collect code predictor weights
    cp_weights = {k: v for k, v in weights.items() if k.startswith("talker.code_predictor.")}

    for frame_idx in range(args.frames):
        if frame_idx == 0:
            semantic_token = first_token
            last_hidden = hidden[:, -1:, :]
        else:
            # Get next hidden from codec embedding
            prev_embed = codec_embed_w[semantic_token].unsqueeze(0).unsqueeze(0)
            # For simplicity, just use the last hidden state (no KV cache in this simplified version)
            # This is approximate but works for demonstration
            last_hidden = prev_embed

            # Run a quick forward through layers for the new token
            new_cos, new_sin = build_rope(1, head_dim)
            for i in range(num_layers):
                last_hidden = run_transformer_layer(
                    last_hidden, weights, f"talker.model.layers.{i}",
                    new_cos, new_sin, None, num_heads, num_kv_heads, head_dim, eps
                )
            last_hidden = rms_norm(last_hidden, weights["talker.model.norm.weight"], eps)

            logits = torch.nn.functional.linear(last_hidden, weights["talker.codec_head.weight"])
            semantic_token = sample_token(logits[0, 0], args.temperature)

        # Generate acoustic codes using code predictor
        # Use code predictor's codec_embedding.0 for semantic token embedding
        cp_semantic_embed_w = weights["talker.code_predictor.model.codec_embedding.0.weight"]
        semantic_embed = cp_semantic_embed_w[semantic_token].unsqueeze(0).unsqueeze(0)

        # Code predictor input: [last_hidden, semantic_embed] - both 1024-dim
        # Note: last_hidden is already 1024-dim from text projection
        cp_input = torch.cat([last_hidden, semantic_embed], dim=1)
        cp_cos, cp_sin = build_rope(2, head_dim)
        cp_mask = torch.triu(torch.full((2, 2), float("-inf")), diagonal=1)

        # Code predictor has different head config: 16 q_heads but 1024 hidden
        # q_proj: [2048, 1024] -> 16 heads × 128 head_dim = 2048 output
        # k_proj: [1024, 1024] -> 8 kv_heads × 128 head_dim = 1024 output
        # So cp uses 16 query heads but GQA with 8 kv heads
        cp_hidden = cp_input
        for i in range(5):
            cp_hidden = run_transformer_layer(
                cp_hidden, weights, f"talker.code_predictor.model.layers.{i}",
                cp_cos, cp_sin, cp_mask, 16, 8, head_dim, eps
            )
        cp_hidden = rms_norm(cp_hidden, weights["talker.code_predictor.model.norm.weight"], eps)

        # Get acoustic tokens
        acoustic_codes = []
        for i in range(15):
            ac_logits = torch.nn.functional.linear(
                cp_hidden[:, 1:2, :],
                weights[f"talker.code_predictor.lm_head.{i}.weight"]
            )
            acoustic_codes.append(ac_logits.argmax(dim=-1).item())

        frame_codes = [semantic_token] + acoustic_codes
        all_codes.append(frame_codes)

        if frame_idx < 5 or frame_idx == args.frames - 1:
            print(f"Frame {frame_idx}: semantic={semantic_token}, acoustics={acoustic_codes[:3]}...")
        elif frame_idx == 5:
            print("...")

    # Convert to codes tensor
    codes_array = np.array(all_codes, dtype=np.int64).T
    print(f"\nCodes shape: {codes_array.shape}")

    # Decode using official decoder
    print("Decoding to audio...")

    # Import and use official decoder
    import types
    import importlib.util

    sys.modules['qwen_tts'] = types.ModuleType('qwen_tts')
    sys.modules['qwen_tts.core'] = types.ModuleType('qwen_tts.core')
    sys.modules['qwen_tts.core.tokenizer_12hz'] = types.ModuleType('qwen_tts.core.tokenizer_12hz')

    spec_config = importlib.util.spec_from_file_location(
        'qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2',
        '/home/trevor/Projects/Qwen3-TTS/qwen_tts/core/tokenizer_12hz/configuration_qwen3_tts_tokenizer_v2.py'
    )
    config_module = importlib.util.module_from_spec(spec_config)
    sys.modules['qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2'] = config_module
    spec_config.loader.exec_module(config_module)

    spec_model = importlib.util.spec_from_file_location(
        'qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2',
        '/home/trevor/Projects/Qwen3-TTS/qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py'
    )
    model_module = importlib.util.module_from_spec(spec_model)
    sys.modules['qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2'] = model_module
    spec_model.loader.exec_module(model_module)

    st_config = config_module.Qwen3TTSTokenizerV2Config.from_pretrained(str(model_dir / "speech_tokenizer"))
    st_model = model_module.Qwen3TTSTokenizerV2Model.from_pretrained(
        str(model_dir / "speech_tokenizer"),
        config=st_config,
        dtype=torch.float32,
    )
    st_model.eval()

    codes_tensor = torch.tensor(codes_array, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        audio = st_model.decoder.chunked_decode(codes_tensor).squeeze(1)
    audio = audio.squeeze(0).numpy()

    print(f"Audio shape: {audio.shape}")
    print(f"Audio duration: {len(audio) / 24000:.2f}s")
    print(f"Audio range: [{audio.min():.4f}, {audio.max():.4f}]")

    # Save outputs
    codes_path = output_dir / f"python_codes_seed{args.seed}.npy"
    np.save(codes_path, codes_array)
    print(f"\nSaved codes: {codes_path}")

    audio_path = output_dir / f"python_audio_seed{args.seed}.wav"
    audio_clipped = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767).astype(np.int16)
    wav.write(str(audio_path), 24000, audio_int16)
    print(f"Saved audio: {audio_path}")

    tokens_path = output_dir / f"token_ids_seed{args.seed}.json"
    with open(tokens_path, "w") as f:
        json.dump({"text": args.text, "tokens": tokens, "seed": args.seed, "frames": args.frames}, f)
    print(f"Saved tokens: {tokens_path}")

    audio_bin_path = output_dir / f"python_audio_seed{args.seed}.bin"
    audio.astype(np.float32).tofile(audio_bin_path)
    print(f"Saved raw audio: {audio_bin_path}")


if __name__ == "__main__":
    main()
