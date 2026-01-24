#!/usr/bin/env python3
"""Get full forward pass values from official model for comparison."""

import torch
from pathlib import Path
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration

def main():
    model_dir = Path("../test_data/model_customvoice")

    print("Loading official model...")
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        str(model_dir),
        dtype=torch.float32,
    )
    model.eval()
    print("Official model loaded!")

    talker = model.talker
    config = model.config.talker_config

    im_start_id = 151644
    assistant_id = 77091
    newline_id = 198
    tts_bos_id = 151672
    tts_eos_id = 151673
    tts_pad_id = 151671

    codec_bos_id = config.codec_bos_id
    codec_pad_id = config.codec_pad_id
    codec_think_id = config.codec_think_id
    codec_think_bos_id = config.codec_think_bos_id
    codec_think_eos_id = config.codec_think_eos_id
    language_id = config.codec_language_id["english"]
    speaker_id = config.spk_id["ryan"]
    text_token = 9707

    text_embed = talker.model.text_embedding.weight
    codec_embed = talker.model.codec_embedding.weight

    # Build input embeddings
    role_prefix_embed = talker.text_projection(
        text_embed[[im_start_id, assistant_id, newline_id]].unsqueeze(0)
    )

    tts_special_embeds = talker.text_projection(
        text_embed[[tts_bos_id, tts_eos_id, tts_pad_id]].unsqueeze(0)
    )
    tts_bos_embed = tts_special_embeds[:, 0:1, :]
    tts_pad_embed = tts_special_embeds[:, 2:3, :]

    codec_prefix = [codec_think_id, codec_think_bos_id, language_id, codec_think_eos_id]
    codec_prefix_embed = codec_embed[codec_prefix].unsqueeze(0)
    speaker_embed = codec_embed[speaker_id].unsqueeze(0).unsqueeze(0)
    codec_pad_bos_embed = codec_embed[[codec_pad_id, codec_bos_id]].unsqueeze(0)
    codec_control_embed = torch.cat([codec_prefix_embed, speaker_embed, codec_pad_bos_embed], dim=1)

    text_control_embed = torch.cat([
        tts_pad_embed.expand(-1, 5, -1),
        tts_bos_embed,
    ], dim=1)

    control_embed = text_control_embed + codec_control_embed[:, :-1, :]

    first_text_embed = talker.text_projection(
        text_embed[text_token].unsqueeze(0).unsqueeze(0)
    )
    first_input = first_text_embed + codec_control_embed[:, -1:, :]

    initial_input = torch.cat([role_prefix_embed, control_embed, first_input], dim=1)
    print(f"Initial input shape: {initial_input.shape}")

    # Position IDs
    batch, seq_len, _ = initial_input.shape
    pos_1d = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    position_ids = pos_1d.unsqueeze(0).expand(3, -1, -1)

    n_layers = config.num_hidden_layers
    head_dim = config.head_dim
    n_heads = config.num_attention_heads
    n_kv_heads = config.num_key_value_heads
    eps = config.rms_norm_eps
    mrope_section = config.rope_scaling["mrope_section"]

    from qwen_tts.core.models.modeling_qwen3_tts import apply_multimodal_rotary_pos_emb

    with torch.no_grad():
        hidden_states = initial_input

        for layer_idx in range(n_layers):
            layer = talker.model.layers[layer_idx]

            # Input layernorm
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            # Self attention
            attn = layer.self_attn

            q = attn.q_proj(hidden_states)
            k = attn.k_proj(hidden_states)
            v = attn.v_proj(hidden_states)

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
            attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32)
            attn_output = torch.matmul(attn_weights, v_exp)

            attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
            attn_output = attn.o_proj(attn_output)

            hidden_states = residual + attn_output

            # MLP
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            mlp_out = layer.mlp(hidden_states)
            hidden_states = residual + mlp_out

            if layer_idx in [0, 11, 23]:
                print(f"After layer {layer_idx}[-1,:5]: {hidden_states[0, -1, :5].tolist()}")

        # Final norm
        hidden_states = talker.model.norm(hidden_states)
        print(f"After final norm[-1,:5]: {hidden_states[0, -1, :5].tolist()}")

        # Codec head
        logits = talker.codec_head(hidden_states[:, -1:, :])
        print(f"Logits shape: {logits.shape}")
        print(f"Logits[-1,:10]: {logits[0, -1, :10].tolist()}")
        print(f"Logits max: {logits[0, -1].max().item():.4f} at {logits[0, -1].argmax().item()}")

        # Apply suppression
        vocab_size = config.vocab_size
        codec_eos_id = config.codec_eos_token_id
        suppress_start = vocab_size - 1024
        suppress_tokens = [i for i in range(suppress_start, vocab_size) if i != codec_eos_id]

        logits_suppressed = logits.clone()
        logits_suppressed[:, :, suppress_tokens] = float('-inf')

        print(f"After suppression max: {logits_suppressed[0, -1].max().item():.4f} at {logits_suppressed[0, -1].argmax().item()}")

        probs = torch.softmax(logits_suppressed[0, -1], dim=-1)
        top5 = torch.topk(probs, 5)
        print(f"Top 5 tokens: {list(zip(top5.indices.tolist(), top5.values.tolist()))}")


if __name__ == "__main__":
    main()
