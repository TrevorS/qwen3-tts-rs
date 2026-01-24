#!/usr/bin/env python3
"""Compare our forward pass with official model's forward."""

import json
import torch
from pathlib import Path
from safetensors.torch import load_file
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration, apply_multimodal_rotary_pos_emb


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return x * weight


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def resize_mlp(x: torch.Tensor, weights: dict, prefix: str) -> torch.Tensor:
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
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def my_apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, interleaved=True):
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
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return torch.cos(emb), torch.sin(emb)


def main():
    model_dir = Path("/home/trevor/Projects/Qwen3-TTS/qwen3-tts-rs/test_data/model_customvoice")

    print("Loading official model...")
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        str(model_dir),
        dtype=torch.float32,
    )
    model.eval()

    print("Loading weights...")
    weights = load_file(model_dir / "model.safetensors")
    weights = {k: v.float() for k, v in weights.items()}

    config = model.config
    talker_config = config.talker_config
    talker = model.talker

    # Build input (same as compare_constructions.py which works)
    im_start = config.im_start_token_id
    assistant = config.assistant_token_id
    newline = 198
    hello_token = 9707

    with torch.no_grad():
        text_embed = talker.model.text_embedding.weight
        codec_embed = talker.model.codec_embedding.weight

        role_prefix = talker.text_projection(
            text_embed[[im_start, assistant, newline]].unsqueeze(0)
        )

        tts_special = talker.text_projection(
            text_embed[[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]].unsqueeze(0)
        )
        tts_bos, tts_eos, tts_pad = tts_special.chunk(3, dim=1)

        language_id = talker_config.codec_language_id["english"]
        speaker_id = talker_config.spk_id["ryan"]

        codec_prefix = codec_embed[[
            talker_config.codec_think_id,
            talker_config.codec_think_bos_id,
            language_id,
            talker_config.codec_think_eos_id,
        ]].unsqueeze(0)
        speaker = codec_embed[speaker_id].unsqueeze(0).unsqueeze(0)
        codec_pad_bos = codec_embed[[talker_config.codec_pad_id, talker_config.codec_bos_id]].unsqueeze(0)

        codec_input = torch.cat([codec_prefix, speaker, codec_pad_bos], dim=1)

        _talker_input = torch.cat((
            tts_pad.expand(-1, codec_input.shape[1] - 2, -1),
            tts_bos,
        ), dim=1) + codec_input[:, :-1]

        talker_input = torch.cat([role_prefix, _talker_input], dim=1)

        first_text = talker.text_projection(
            text_embed[hello_token].unsqueeze(0).unsqueeze(0)
        )
        talker_input = torch.cat([
            talker_input,
            first_text + codec_input[:, -1:]
        ], dim=1)

        print(f"Input shape: {talker_input.shape}")

        # Position IDs
        batch, seq_len = talker_input.shape[:2]
        pos_1d = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        position_ids = pos_1d.unsqueeze(0).expand(3, -1, -1)

        # === Official forward ===
        print("\n=== Official forward ===")
        outputs = talker.model(
            inputs_embeds=talker_input,
            position_ids=position_ids,
            use_cache=False,
        )
        hidden_official = outputs.last_hidden_state  # Already includes final norm!
        logits_official = talker.codec_head(hidden_official[:, -1:])
        print(f"Official logits max: {logits_official[0, -1].max().item():.4f} at {logits_official[0, -1].argmax().item()}")

        # === My forward pass (layer by layer) ===
        print("\n=== My forward ===")
        n_layers = talker_config.num_hidden_layers
        head_dim = talker_config.head_dim
        n_heads = talker_config.num_attention_heads
        n_kv_heads = talker_config.num_key_value_heads
        eps = talker_config.rms_norm_eps
        mrope_section = talker_config.rope_scaling["mrope_section"]

        hidden_mine = talker_input.clone()

        for layer_idx in range(n_layers):
            layer = talker.model.layers[layer_idx]

            # Input layernorm
            residual = hidden_mine
            hidden_mine = layer.input_layernorm(hidden_mine)

            # Self attention using official components
            attn = layer.self_attn

            q = attn.q_proj(hidden_mine)
            k = attn.k_proj(hidden_mine)
            v = attn.v_proj(hidden_mine)

            hidden_shape = (batch, seq_len, n_heads, head_dim)
            q = q.view(hidden_shape)
            k = k.view(batch, seq_len, n_kv_heads, head_dim)
            v = v.view(batch, seq_len, n_kv_heads, head_dim)

            q = attn.q_norm(q)
            k = attn.k_norm(k)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            cos, sin = talker.model.rotary_emb(v, position_ids=position_ids)
            q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=0)

            n_rep = n_heads // n_kv_heads
            k_exp = k.repeat_interleave(n_rep, dim=1)
            v_exp = v.repeat_interleave(n_rep, dim=1)

            attn_weights = torch.matmul(q, k_exp.transpose(-2, -1)) / (head_dim ** 0.5)

            # Apply causal mask
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
            attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

            attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32)
            attn_output = torch.matmul(attn_weights, v_exp)

            attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
            attn_output = attn.o_proj(attn_output)

            hidden_mine = residual + attn_output

            # MLP
            residual = hidden_mine
            hidden_mine = layer.post_attention_layernorm(hidden_mine)
            mlp_out = layer.mlp(hidden_mine)
            hidden_mine = residual + mlp_out

            if layer_idx in [0, n_layers - 1]:
                diff = (hidden_mine - outputs.last_hidden_state if layer_idx == 0 else hidden_mine).abs().max().item()
                # print(f"After layer {layer_idx}, my[-1,:5]: {hidden_mine[0, -1, :5].tolist()}")

        # Final norm
        hidden_mine = talker.model.norm(hidden_mine)
        logits_mine = talker.codec_head(hidden_mine[:, -1:])
        print(f"My logits max: {logits_mine[0, -1].max().item():.4f} at {logits_mine[0, -1].argmax().item()}")

        # Compare
        hidden_diff = (hidden_official - hidden_mine).abs().max().item()
        logits_diff = (logits_official - logits_mine).abs().max().item()
        print(f"\nHidden diff: {hidden_diff}")
        print(f"Logits diff: {logits_diff}")


if __name__ == "__main__":
    main()
