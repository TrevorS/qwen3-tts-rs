#!/usr/bin/env python3
"""Get intermediate values from official model for comparison."""

import torch
from pathlib import Path
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration

def main():
    model_dir = Path("../test_data/model_customvoice")

    print("Loading official model...")
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float32,
    )
    model.eval()
    print("Official model loaded!")

    # Get the talker model
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

    # Get embeddings
    text_embed = talker.model.text_embedding.weight
    codec_embed = talker.model.codec_embedding.weight

    # Build input embeddings using text_projection
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

    # Access layer 0 directly
    layer0 = talker.model.layers[0]
    eps = config.rms_norm_eps
    head_dim = config.head_dim
    n_heads = config.num_attention_heads
    n_kv_heads = config.num_key_value_heads

    with torch.no_grad():
        # Input layernorm
        residual = initial_input
        hidden_states = layer0.input_layernorm(initial_input)

        # Self attention
        attn = layer0.self_attn

        # Q, K, V projections
        q = attn.q_proj(hidden_states)
        k = attn.k_proj(hidden_states)
        v = attn.v_proj(hidden_states)

        # Reshape
        hidden_shape = (batch, seq_len, n_heads, head_dim)
        q = q.view(hidden_shape)
        k = k.view(batch, seq_len, n_kv_heads, head_dim)
        v = v.view(batch, seq_len, n_kv_heads, head_dim)

        # Apply Q/K norm
        q_normed = attn.q_norm(q)
        k_normed = attn.k_norm(k)

        # Transpose for attention
        q_t = q_normed.transpose(1, 2)
        k_t = k_normed.transpose(1, 2)
        v_t = v.transpose(1, 2)

        print(f"Q after transpose shape: {q_t.shape}")

        # Get RoPE embeddings from official model (rotary_emb is in talker.model, not attention)
        cos, sin = talker.model.rotary_emb(v_t, position_ids=position_ids)
        print(f"Position IDs: {position_ids[:, 0, :].tolist()}")
        print(f"cos shape: {cos.shape}")

        # Access the M-RoPE section from config
        mrope_section = config.rope_scaling["mrope_section"]
        print(f"mrope_section: {mrope_section}")

        # Apply M-RoPE using the model's method
        # The official model uses apply_multimodal_rotary_pos_emb from the module
        from qwen_tts.core.models.modeling_qwen3_tts import apply_multimodal_rotary_pos_emb

        q_rope, k_rope = apply_multimodal_rotary_pos_emb(
            q_t, k_t, cos, sin,
            mrope_section,
            unsqueeze_dim=0
        )

        print(f"\nAfter RoPE q[-1,0,:5]: {q_rope[0, 0, -1, :5].tolist()}")
        print(f"After RoPE k[-1,0,:5]: {k_rope[0, 0, -1, :5].tolist()}")

        # Continue with attention
        n_rep = n_heads // n_kv_heads
        k_exp = k_rope.repeat_interleave(n_rep, dim=1)
        v_exp = v_t.repeat_interleave(n_rep, dim=1)

        attn_weights = torch.matmul(q_rope, k_exp.transpose(-2, -1)) / (head_dim ** 0.5)
        print(f"\nAttn weights shape: {attn_weights.shape}")
        print(f"Attn weights[-1,-1,:]: {attn_weights[0, 0, -1, :].tolist()}")

        attn_weights_soft = torch.softmax(attn_weights, dim=-1, dtype=torch.float32)
        print(f"Softmax attn[-1,-1,:]: {attn_weights_soft[0, 0, -1, :].tolist()}")

        attn_output = torch.matmul(attn_weights_soft, v_exp)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        attn_output = attn.o_proj(attn_output)

        hidden_states_after_attn = residual + attn_output
        print(f"\nAfter attn+residual[-1,:5]: {hidden_states_after_attn[0, -1, :5].tolist()}")

        # MLP
        residual = hidden_states_after_attn
        hidden_states = layer0.post_attention_layernorm(hidden_states_after_attn)
        mlp_out = layer0.mlp(hidden_states)
        hidden_states = residual + mlp_out
        print(f"After MLP layer0[-1,:5]: {hidden_states[0, -1, :5].tolist()}")


if __name__ == "__main__":
    main()
