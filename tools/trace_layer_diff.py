#!/usr/bin/env python3
"""Trace where the forward pass diverges layer by layer."""

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

        # Run official forward and capture intermediate states
        # We need to hook into the model to get layer outputs
        layer_outputs_official = []

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                layer_outputs_official.append(output[0].clone())
            else:
                layer_outputs_official.append(output.clone())

        handles = []
        for layer in talker.model.layers:
            h = layer.register_forward_hook(hook_fn)
            handles.append(h)

        # Run official forward
        outputs = talker.model(
            inputs_embeds=talker_input,
            position_ids=position_ids,
            use_cache=False,
        )

        for h in handles:
            h.remove()

        print(f"\nCaptured {len(layer_outputs_official)} layer outputs")

        # Now run my forward and compare at each layer
        hidden_mine = talker_input.clone()

        for layer_idx in range(n_layers):
            layer = talker.model.layers[layer_idx]

            # Input layernorm
            residual = hidden_mine
            hidden_mine = layer.input_layernorm(hidden_mine)

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

            # Compare with official
            diff = (layer_outputs_official[layer_idx] - hidden_mine).abs().max().item()
            # Print every layer and don't break
            print(f"Layer {layer_idx}: max diff = {diff:.6f}")
            if layer_idx in [0, 13, 27] or diff > 0.1:
                print(f"  Official[-1,:5]: {layer_outputs_official[layer_idx][0, -1, :5].tolist()}")
                print(f"  Mine[-1,:5]: {hidden_mine[0, -1, :5].tolist()}")

        # Compare with actual model output
        outputs = talker.model(
            inputs_embeds=talker_input,
            position_ids=position_ids,
            use_cache=False,
        )
        official_last_hidden = outputs.last_hidden_state
        print(f"\n=== Final comparison ===")
        print(f"outputs.last_hidden_state[-1,:5]: {official_last_hidden[0, -1, :5].tolist()}")
        print(f"My hidden[-1,:5]: {hidden_mine[0, -1, :5].tolist()}")

        last_hidden_diff = (official_last_hidden - hidden_mine).abs().max().item()
        print(f"Last hidden diff: {last_hidden_diff:.6f}")

        # After norm
        hidden_official_normed = talker.model.norm(official_last_hidden)
        hidden_mine_normed = talker.model.norm(hidden_mine)
        normed_diff = (hidden_official_normed - hidden_mine_normed).abs().max().item()
        print(f"After norm diff: {normed_diff:.6f}")


if __name__ == "__main__":
    main()
