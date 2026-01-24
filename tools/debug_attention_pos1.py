#!/usr/bin/env python3
"""Debug attention computation at position 1 where divergence occurs."""

import torch
from pathlib import Path
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration, apply_multimodal_rotary_pos_emb


def main():
    model_dir = Path("/home/trevor/Projects/Qwen3-TTS/qwen3-tts-rs/test_data/model_customvoice")

    print("Loading official model...")
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        str(model_dir),
        dtype=torch.float32,
    )
    model.eval()

    config = model.config
    talker_config = config.talker_config
    talker = model.talker

    # Build input
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

        batch, seq_len = talker_input.shape[:2]
        pos_1d = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        position_ids = pos_1d.unsqueeze(0).expand(3, -1, -1)

        n_layers = talker_config.num_hidden_layers
        head_dim = talker_config.head_dim
        n_heads = talker_config.num_attention_heads
        n_kv_heads = talker_config.num_key_value_heads
        eps = talker_config.rms_norm_eps
        mrope_section = talker_config.rope_scaling["mrope_section"]

        # === Compare layer 0 attention in detail ===
        layer = talker.model.layers[0]

        # Input layernorm
        hidden = layer.input_layernorm(talker_input)

        attn = layer.self_attn

        # Q, K, V projections
        q = attn.q_proj(hidden)
        k = attn.k_proj(hidden)
        v = attn.v_proj(hidden)

        print(f"\n=== After projection (before reshape) ===")
        print(f"Q[0,1,:5]: {q[0, 1, :5].tolist()}")
        print(f"K[0,1,:5]: {k[0, 1, :5].tolist()}")
        print(f"V[0,1,:5]: {v[0, 1, :5].tolist()}")

        # Reshape
        hidden_shape = (batch, seq_len, n_heads, head_dim)
        q = q.view(hidden_shape)
        k = k.view(batch, seq_len, n_kv_heads, head_dim)
        v = v.view(batch, seq_len, n_kv_heads, head_dim)

        print(f"\n=== After reshape ===")
        print(f"Q shape: {q.shape}")
        print(f"Q[0,1,0,:5]: {q[0, 1, 0, :5].tolist()}")  # pos 1, head 0

        # QK norm
        q = attn.q_norm(q)
        k = attn.k_norm(k)

        print(f"\n=== After QK norm ===")
        print(f"Q[0,1,0,:5]: {q[0, 1, 0, :5].tolist()}")
        print(f"K[0,1,0,:5]: {k[0, 1, 0, :5].tolist()}")

        # Transpose for attention
        q = q.transpose(1, 2)  # [batch, heads, seq, dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        print(f"\n=== After transpose ===")
        print(f"Q shape: {q.shape}")
        print(f"Q[0,0,1,:5]: {q[0, 0, 1, :5].tolist()}")  # head 0, pos 1

        # RoPE
        cos, sin = talker.model.rotary_emb(v, position_ids=position_ids)
        print(f"\n=== RoPE embeddings ===")
        print(f"cos shape: {cos.shape}")
        print(f"sin shape: {sin.shape}")
        print(f"cos[0,0,1,:5]: {cos[0, 0, 1, :5].tolist()}")  # modality 0, batch 0, pos 1
        print(f"cos[1,0,1,:5]: {cos[1, 0, 1, :5].tolist()}")  # modality 1
        print(f"cos[2,0,1,:5]: {cos[2, 0, 1, :5].tolist()}")  # modality 2

        q_before_rope = q.clone()
        k_before_rope = k.clone()

        q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=0)

        print(f"\n=== After RoPE ===")
        print(f"Q[0,0,1,:5]: {q[0, 0, 1, :5].tolist()}")
        print(f"K[0,0,1,:5]: {k[0, 0, 1, :5].tolist()}")

        # Expand KV for GQA
        n_rep = n_heads // n_kv_heads
        k_exp = k.repeat_interleave(n_rep, dim=1)
        v_exp = v.repeat_interleave(n_rep, dim=1)

        print(f"\n=== After KV expansion ===")
        print(f"K_exp shape: {k_exp.shape}")

        # Attention scores
        attn_weights = torch.matmul(q, k_exp.transpose(-2, -1)) / (head_dim ** 0.5)

        print(f"\n=== Attention weights (before mask/softmax) ===")
        print(f"attn_weights shape: {attn_weights.shape}")
        print(f"attn_weights[0,0,1,:]: {attn_weights[0, 0, 1, :].tolist()}")  # head 0, query pos 1

        # Apply causal mask!
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        print(f"\n=== Attention weights (after causal mask) ===")
        print(f"attn_weights[0,0,1,:]: {attn_weights[0, 0, 1, :].tolist()}")  # head 0, query pos 1

        attn_weights_softmax = torch.softmax(attn_weights, dim=-1, dtype=torch.float32)

        print(f"\n=== Attention weights (after softmax) ===")
        print(f"attn_weights[0,0,1,:]: {attn_weights_softmax[0, 0, 1, :].tolist()}")

        # Attention output
        attn_output = torch.matmul(attn_weights_softmax, v_exp)

        print(f"\n=== Attention output (before o_proj) ===")
        print(f"attn_output[0,0,1,:5]: {attn_output[0, 0, 1, :5].tolist()}")

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        print(f"\n=== After reshape (before o_proj) ===")
        print(f"attn_output shape: {attn_output.shape}")
        print(f"attn_output[0,1,:5]: {attn_output[0, 1, :5].tolist()}")

        attn_output = attn.o_proj(attn_output)

        print(f"\n=== After o_proj ===")
        print(f"attn_output[0,1,:5]: {attn_output[0, 1, :5].tolist()}")

        # Now compare with official attention forward
        print("\n" + "="*60)
        print("=== COMPARING WITH OFFICIAL ATTENTION ===")
        print("="*60)

        # Run official attention
        hidden_for_official = layer.input_layernorm(talker_input)

        cos_official, sin_official = talker.model.rotary_emb(hidden_for_official, position_ids=position_ids)

        attn_result = layer.self_attn(
            hidden_states=hidden_for_official,
            position_embeddings=(cos_official, sin_official),
            attention_mask=None,
        )
        attn_output_official = attn_result[0]

        print(f"\nOfficial attn output[0,1,:5]: {attn_output_official[0, 1, :5].tolist()}")
        print(f"My attn output[0,1,:5]: {attn_output[0, 1, :5].tolist()}")

        diff = (attn_output_official - attn_output).abs()
        max_diff = diff.max().item()
        max_diff_idx = diff.argmax()
        max_pos = max_diff_idx // diff.shape[-1]
        max_dim = max_diff_idx % diff.shape[-1]
        print(f"\nMax diff: {max_diff:.6f}")
        print(f"Max diff at position {max_pos.item()}, dim {max_dim.item()}")

        if max_diff > 1e-4:
            print(f"\n=== Investigating position {max_pos.item()} ===")
            pos = max_pos.item()
            print(f"Official[0,{pos},:10]: {attn_output_official[0, pos, :10].tolist()}")
            print(f"Mine[0,{pos},:10]: {attn_output[0, pos, :10].tolist()}")

            # Check the specific dimension
            dim = max_dim.item()
            print(f"\nAt dim {dim}:")
            print(f"  Official: {attn_output_official[0, pos, dim].item():.6f}")
            print(f"  Mine: {attn_output[0, pos, dim].item():.6f}")


if __name__ == "__main__":
    main()
