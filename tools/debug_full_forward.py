#!/usr/bin/env python3
"""Compare full forward pass between our implementation and official model."""

import json
import torch
from pathlib import Path
from safetensors.torch import load_file


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply RMS normalization."""
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return x * weight


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU activation function."""
    return x * torch.sigmoid(x)


def resize_mlp(x: torch.Tensor, weights: dict, prefix: str) -> torch.Tensor:
    """Apply the ResizeMLP (2-layer MLP with SiLU)."""
    hidden = torch.nn.functional.linear(
        x,
        weights[f"{prefix}.linear_fc1.weight"],
        weights[f"{prefix}.linear_fc1.bias"]
    )
    hidden = silu(hidden)
    out = torch.nn.functional.linear(
        hidden,
        weights[f"{prefix}.linear_fc2.weight"],
        weights[f"{prefix}.linear_fc2.bias"]
    )
    return out


def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, interleaved=True):
    """Apply 3D Multi-modal RoPE (M-RoPE)."""
    if interleaved:
        def apply_interleaved_rope(x, modality_num):
            x_t = x[0].clone()
            for i, n in enumerate(mrope_section[1:], 1):
                beg_idx = i
                end_idx = n * modality_num
                x_t[..., beg_idx:end_idx:modality_num] = x[i, ..., beg_idx:end_idx:modality_num]
            return x_t

        dim = cos.shape[-1]
        modality_num = len(mrope_section)
        cos = torch.cat([apply_interleaved_rope(cos[..., :dim // 2], modality_num)] * 2, dim=-1).unsqueeze(1)
        sin = torch.cat([apply_interleaved_rope(sin[..., :dim // 2], modality_num)] * 2, dim=-1).unsqueeze(1)
    else:
        mrope_section_doubled = mrope_section * 2
        cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section_doubled, dim=-1))], dim=-1).unsqueeze(1)
        sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section_doubled, dim=-1))], dim=-1).unsqueeze(1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def build_rope_cache(max_seq_len: int, head_dim: int, theta: float = 1000000.0):
    """Build rotary position embedding cache."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return torch.cos(emb), torch.sin(emb)


def transformer_layer(
    hidden_states: torch.Tensor,
    weights: dict,
    prefix: str,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    intermediate_size: int,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    position_ids: torch.Tensor,
    eps: float = 1e-6,
    mrope_section: list = None,
):
    """Single transformer layer forward pass."""
    batch, seq_len, hidden_size = hidden_states.shape

    # Input layernorm
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, weights[f"{prefix}.input_layernorm.weight"], eps)

    # Self attention
    q = torch.nn.functional.linear(hidden_states, weights[f"{prefix}.self_attn.q_proj.weight"])
    k = torch.nn.functional.linear(hidden_states, weights[f"{prefix}.self_attn.k_proj.weight"])
    v = torch.nn.functional.linear(hidden_states, weights[f"{prefix}.self_attn.v_proj.weight"])

    q = q.view(batch, seq_len, n_heads, head_dim)
    k = k.view(batch, seq_len, n_kv_heads, head_dim)
    v = v.view(batch, seq_len, n_kv_heads, head_dim)

    # Apply QK norm
    q = rms_norm(q, weights[f"{prefix}.self_attn.q_norm.weight"], eps)
    k = rms_norm(k, weights[f"{prefix}.self_attn.k_norm.weight"], eps)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Apply M-RoPE
    cos = rope_cos[position_ids]
    sin = rope_sin[position_ids]
    q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, interleaved=True)

    # Repeat KV for GQA
    if n_kv_heads < n_heads:
        n_rep = n_heads // n_kv_heads
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)

    # Attention
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output = torch.matmul(attn_weights, v)

    attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
    attn_output = torch.nn.functional.linear(attn_output, weights[f"{prefix}.self_attn.o_proj.weight"])

    hidden_states = residual + attn_output

    # MLP
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, weights[f"{prefix}.post_attention_layernorm.weight"], eps)

    gate = torch.nn.functional.linear(hidden_states, weights[f"{prefix}.mlp.gate_proj.weight"])
    up = torch.nn.functional.linear(hidden_states, weights[f"{prefix}.mlp.up_proj.weight"])
    hidden_states = silu(gate) * up
    hidden_states = torch.nn.functional.linear(hidden_states, weights[f"{prefix}.mlp.down_proj.weight"])

    hidden_states = residual + hidden_states

    return hidden_states


def main():
    model_dir = Path("../test_data/model_customvoice")

    with open(model_dir / "config.json") as f:
        config = json.load(f)

    talker_config = config["talker_config"]
    n_layers = talker_config["num_hidden_layers"]
    head_dim = talker_config["head_dim"]
    n_heads = talker_config["num_attention_heads"]
    n_kv_heads = talker_config["num_key_value_heads"]
    intermediate_size = talker_config["intermediate_size"]
    eps = talker_config["rms_norm_eps"]
    rope_theta = talker_config["rope_theta"]
    mrope_section = talker_config.get("mrope_section", [24, 20, 20])
    vocab_size = talker_config["vocab_size"]

    print(f"Model: {n_layers} layers, head_dim={head_dim}, n_heads={n_heads}")

    print("Loading weights...")
    weights = load_file(model_dir / "model.safetensors")
    weights = {k: v.float() for k, v in weights.items()}

    # Build input embeddings
    text_embed_weight = weights["talker.model.text_embedding.weight"]
    codec_embed_weight = weights["talker.model.codec_embedding.weight"]

    im_start_id = config["im_start_token_id"]
    assistant_id = config["assistant_token_id"]
    newline_id = 198
    tts_bos_id = config["tts_bos_token_id"]
    tts_eos_id = config["tts_eos_token_id"]
    tts_pad_id = config["tts_pad_token_id"]

    codec_bos_id = talker_config["codec_bos_id"]
    codec_pad_id = talker_config["codec_pad_id"]
    codec_think_id = talker_config["codec_think_id"]
    codec_think_bos_id = talker_config["codec_think_bos_id"]
    codec_think_eos_id = talker_config["codec_think_eos_id"]
    language_id = talker_config["codec_language_id"]["english"]
    speaker_id = talker_config["spk_id"]["ryan"]
    text_token = 9707

    # Build input embeddings
    role_prefix_embed = resize_mlp(
        text_embed_weight[[im_start_id, assistant_id, newline_id]].unsqueeze(0),
        weights, "talker.text_projection"
    )

    tts_special_embeds = resize_mlp(
        text_embed_weight[[tts_bos_id, tts_eos_id, tts_pad_id]].unsqueeze(0),
        weights, "talker.text_projection"
    )
    tts_bos_embed = tts_special_embeds[:, 0:1, :]
    tts_pad_embed = tts_special_embeds[:, 2:3, :]

    codec_prefix = [codec_think_id, codec_think_bos_id, language_id, codec_think_eos_id]
    codec_prefix_embed = codec_embed_weight[codec_prefix].unsqueeze(0)
    speaker_embed = codec_embed_weight[speaker_id].unsqueeze(0).unsqueeze(0)
    codec_pad_bos_embed = codec_embed_weight[[codec_pad_id, codec_bos_id]].unsqueeze(0)
    codec_control_embed = torch.cat([codec_prefix_embed, speaker_embed, codec_pad_bos_embed], dim=1)

    text_control_embed = torch.cat([
        tts_pad_embed.expand(-1, 5, -1),
        tts_bos_embed,
    ], dim=1)

    control_embed = text_control_embed + codec_control_embed[:, :-1, :]

    first_text_embed = resize_mlp(
        text_embed_weight[text_token].unsqueeze(0).unsqueeze(0),
        weights, "talker.text_projection"
    )
    first_input = first_text_embed + codec_control_embed[:, -1:, :]

    initial_input = torch.cat([role_prefix_embed, control_embed, first_input], dim=1)
    print(f"Initial input shape: {initial_input.shape}")

    # Build RoPE cache
    rope_cos, rope_sin = build_rope_cache(8192, head_dim, rope_theta)

    batch, seq_len = initial_input.shape[:2]
    pos_1d = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    position_ids = pos_1d.unsqueeze(0).expand(3, -1, -1)

    # Run through all layers
    hidden_states = initial_input
    for layer_idx in range(n_layers):
        prefix = f"talker.model.layers.{layer_idx}"
        hidden_states = transformer_layer(
            hidden_states, weights, prefix,
            n_heads, n_kv_heads, head_dim, intermediate_size,
            rope_cos, rope_sin, position_ids, eps,
            mrope_section=mrope_section,
        )
        if layer_idx in [0, 11, 23]:  # Check key layers
            print(f"After layer {layer_idx}[-1,:5]: {hidden_states[0, -1, :5].tolist()}")

    # Final norm
    hidden_states = rms_norm(hidden_states, weights["talker.model.norm.weight"], eps)
    print(f"After final norm[-1,:5]: {hidden_states[0, -1, :5].tolist()}")

    # Codec head
    logits = torch.nn.functional.linear(hidden_states[:, -1:, :], weights["talker.codec_head.weight"])
    print(f"Logits shape: {logits.shape}")
    print(f"Logits[-1,:10]: {logits[0, -1, :10].tolist()}")
    print(f"Logits max: {logits[0, -1].max().item():.4f} at {logits[0, -1].argmax().item()}")

    # Apply suppression like in generation
    codec_eos_id = talker_config["codec_eos_token_id"]  # 2150
    suppress_start = vocab_size - 1024  # 2048
    suppress_tokens = [i for i in range(suppress_start, vocab_size) if i != codec_eos_id]

    logits_suppressed = logits.clone()
    logits_suppressed[:, :, suppress_tokens] = float('-inf')

    print(f"After suppression max: {logits_suppressed[0, -1].max().item():.4f} at {logits_suppressed[0, -1].argmax().item()}")

    # Sample
    probs = torch.softmax(logits_suppressed[0, -1], dim=-1)
    top5 = torch.topk(probs, 5)
    print(f"Top 5 tokens: {list(zip(top5.indices.tolist(), top5.values.tolist()))}")


if __name__ == "__main__":
    main()
