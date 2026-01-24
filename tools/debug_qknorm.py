#!/usr/bin/env python3
"""Debug QK normalization by comparing with official model values."""

import json
import torch
from pathlib import Path
from safetensors.torch import load_file


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply RMS normalization."""
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return x * weight


def resize_mlp(x: torch.Tensor, weights: dict, prefix: str) -> torch.Tensor:
    """Apply the ResizeMLP (2-layer MLP with SiLU)."""
    hidden = torch.nn.functional.linear(
        x,
        weights[f"{prefix}.linear_fc1.weight"],
        weights[f"{prefix}.linear_fc1.bias"]
    )
    hidden = hidden * torch.sigmoid(hidden)
    out = torch.nn.functional.linear(
        hidden,
        weights[f"{prefix}.linear_fc2.weight"],
        weights[f"{prefix}.linear_fc2.bias"]
    )
    return out


def main():
    model_dir = Path("../test_data/model_customvoice")

    # Load config
    with open(model_dir / "config.json") as f:
        config = json.load(f)

    talker_config = config["talker_config"]
    head_dim = talker_config["head_dim"]  # 128
    n_heads = talker_config["num_attention_heads"]  # 16
    n_kv_heads = talker_config["num_key_value_heads"]  # 8
    eps = talker_config["rms_norm_eps"]

    # Load weights
    print("Loading weights...")
    weights = load_file(model_dir / "model.safetensors")
    weights = {k: v.float() for k, v in weights.items()}

    # Build same prefill input as in our generation script
    text_embed_weight = weights["talker.model.text_embedding.weight"]
    codec_embed_weight = weights["talker.model.codec_embedding.weight"]

    # Get special token IDs
    im_start_id = config["im_start_token_id"]  # 151644
    assistant_id = config["assistant_token_id"]  # 77091
    newline_id = 198
    tts_bos_id = config["tts_bos_token_id"]  # 151672
    tts_eos_id = config["tts_eos_token_id"]  # 151673
    tts_pad_id = config["tts_pad_token_id"]  # 151671

    codec_bos_id = talker_config["codec_bos_id"]  # 2149
    codec_pad_id = talker_config["codec_pad_id"]  # 2148
    codec_think_id = talker_config["codec_think_id"]  # 2154
    codec_think_bos_id = talker_config["codec_think_bos_id"]  # 2156
    codec_think_eos_id = talker_config["codec_think_eos_id"]  # 2157
    language_id = talker_config["codec_language_id"]["english"]  # 2050
    speaker_id = talker_config["spk_id"]["ryan"]  # 3061

    text_token = 9707  # "Hello"

    # Build input embeddings (same as generate_customvoice_correct.py)
    # Role prefix
    role_prefix_embed = resize_mlp(
        text_embed_weight[[im_start_id, assistant_id, newline_id]].unsqueeze(0),
        weights, "talker.text_projection"
    )

    # TTS special embeds
    tts_special_embeds = resize_mlp(
        text_embed_weight[[tts_bos_id, tts_eos_id, tts_pad_id]].unsqueeze(0),
        weights, "talker.text_projection"
    )
    tts_bos_embed = tts_special_embeds[:, 0:1, :]
    tts_pad_embed = tts_special_embeds[:, 2:3, :]

    # Codec control: [think, think_bos, lang, think_eos, speaker, pad, bos]
    codec_prefix = [codec_think_id, codec_think_bos_id, language_id, codec_think_eos_id]
    codec_prefix_embed = codec_embed_weight[codec_prefix].unsqueeze(0)
    speaker_embed = codec_embed_weight[speaker_id].unsqueeze(0).unsqueeze(0)
    codec_pad_bos_embed = codec_embed_weight[[codec_pad_id, codec_bos_id]].unsqueeze(0)
    codec_control_embed = torch.cat([codec_prefix_embed, speaker_embed, codec_pad_bos_embed], dim=1)

    # Text control: [pad, pad, pad, pad, pad, bos]
    text_control_embed = torch.cat([
        tts_pad_embed.expand(-1, 5, -1),
        tts_bos_embed,
    ], dim=1)

    # Combined control (without last codec position)
    control_embed = text_control_embed + codec_control_embed[:, :-1, :]

    # First text with codec_bos
    first_text_embed = resize_mlp(
        text_embed_weight[text_token].unsqueeze(0).unsqueeze(0),
        weights, "talker.text_projection"
    )
    first_input = first_text_embed + codec_control_embed[:, -1:, :]

    # Full input: [role_prefix, control, first_text]
    initial_input = torch.cat([role_prefix_embed, control_embed, first_input], dim=1)
    print(f"Initial input shape: {initial_input.shape}")  # Should be [1, 10, 2048]

    # Now run through layer 0 and extract intermediate values
    prefix = "talker.model.layers.0"

    # Input layernorm
    hidden_states = rms_norm(initial_input, weights[f"{prefix}.input_layernorm.weight"], eps)
    print(f"\nAfter input_layernorm[-1,:5]: {hidden_states[0, -1, :5].tolist()}")

    # Q, K projections
    q = torch.nn.functional.linear(hidden_states, weights[f"{prefix}.self_attn.q_proj.weight"])
    k = torch.nn.functional.linear(hidden_states, weights[f"{prefix}.self_attn.k_proj.weight"])

    print(f"After q_proj[-1,:5]: {q[0, -1, :5].tolist()}")
    print(f"After k_proj[-1,:5]: {k[0, -1, :5].tolist()}")

    # Reshape
    batch, seq_len = initial_input.shape[:2]
    q = q.view(batch, seq_len, n_heads, head_dim)
    k = k.view(batch, seq_len, n_kv_heads, head_dim)

    print(f"\nQ shape after view: {q.shape}")  # [1, 10, 16, 128]
    print(f"K shape after view: {k.shape}")  # [1, 10, 8, 128]

    # Check QK norm weight shapes
    q_norm_weight = weights[f"{prefix}.self_attn.q_norm.weight"]
    k_norm_weight = weights[f"{prefix}.self_attn.k_norm.weight"]
    print(f"\nq_norm.weight shape: {q_norm_weight.shape}")
    print(f"k_norm.weight shape: {k_norm_weight.shape}")

    # Apply QK norm (on head_dim dimension which is the last)
    q_normed = rms_norm(q, q_norm_weight, eps)
    k_normed = rms_norm(k, k_norm_weight, eps)

    print(f"\nOur q_norm[-1,0,:5]: {q_normed[0, -1, 0, :5].tolist()}")
    print(f"Our k_norm[-1,0,:5]: {k_normed[0, -1, 0, :5].tolist()}")

    print("\n=== Official model values (from earlier debug) ===")
    print(f"Official q_norm[-1,0,:5]: [0.4429, -0.9468, 0.1208, -0.3787, 1.4232]")
    print(f"Official k_norm[-1,0,:5]: [-4.6627, 0.6511, 4.9487, 0.3390, -1.7821]")


if __name__ == "__main__":
    main()
