#!/usr/bin/env python3
"""Debug frame 1 generation - compare with Rust."""

import torch
from pathlib import Path
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration

# Hook storage
debug_data = {}

def make_attn_hook(layer_idx):
    """Create a hook to capture Q/K before and after RoPE."""
    def hook(module, args, output):
        # Only care about layer 0 and single-token generation
        hidden_states = args[0]
        if hidden_states.shape[1] == 1 and layer_idx == 0:
            # This is called AFTER the forward - we need a pre-hook to get Q/K before RoPE
            pass
        return output
    return hook

def make_pre_attn_hook(layer_idx):
    """Pre-hook to capture Q/K before RoPE is applied."""
    def hook(module, args, kwargs):
        # Get hidden_states from kwargs (HF transformers passes it as kwarg)
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and len(args) > 0:
            hidden_states = args[0]

        if hidden_states is None:
            return

        # Single token generation, layer 0
        if hidden_states.shape[1] == 1 and layer_idx == 0:
            head_dim = module.head_dim

            # Compute Q and K directly (before RoPE)
            q_proj = module.q_proj(hidden_states)
            k_proj = module.k_proj(hidden_states)

            # Apply q_norm and k_norm
            q_normed = module.q_norm(q_proj.view(*hidden_states.shape[:-1], -1, head_dim))
            k_normed = module.k_norm(k_proj.view(*hidden_states.shape[:-1], -1, head_dim))

            debug_data["q_sum_before_rope"] = q_normed.sum().item()
            debug_data["k_sum_before_rope"] = k_normed.sum().item()

            # Also get position embeddings to compute after RoPE
            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is not None:
                cos, sin = position_embeddings
                debug_data["cos_sum"] = cos.sum().item()
                debug_data["sin_sum"] = sin.sum().item()
    return hook


def make_layer_output_hook(layer_idx):
    """Hook to capture hidden state after each layer."""
    def hook(module, args, kwargs, output):
        hidden_states = output[0]  # First element is hidden states
        if hidden_states.shape[1] == 1:  # Single token generation
            debug_data[f"layer_{layer_idx}_output_sum"] = hidden_states.sum().item()
    return hook


def main():
    script_dir = Path(__file__).parent.parent
    model_dir = (script_dir / "test_data/model_customvoice").resolve()

    torch.manual_seed(42)

    print("Loading model...")
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        str(model_dir),
        dtype=torch.float32,
    )
    model.eval()

    config = model.config
    talker_config = config.talker_config

    text_tokens = [9707]  # "Hello"
    speaker_id = talker_config.spk_id["ryan"]
    language_id = talker_config.codec_language_id["english"]

    im_start = config.im_start_token_id
    assistant = config.assistant_token_id
    newline = 198

    with torch.no_grad():
        talker = model.talker
        text_embed_layer = talker.get_text_embeddings()
        codec_embed_layer = talker.get_input_embeddings()

        # Get TTS special embeddings
        tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(
            text_embed_layer(
                torch.tensor([[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]])
            )
        ).chunk(3, dim=1)

        # Build codec prefill for CustomVoice
        codec_prefill = [
            talker_config.codec_think_id,
            talker_config.codec_think_bos_id,
            language_id,
            talker_config.codec_think_eos_id,
        ]
        codec_input_0 = codec_embed_layer(torch.tensor([codec_prefill]))
        codec_input_1 = codec_embed_layer(torch.tensor([[talker_config.codec_pad_id, talker_config.codec_bos_id]]))
        speaker_embed = codec_embed_layer(torch.tensor([speaker_id])).unsqueeze(0)
        codec_input = torch.cat([codec_input_0, speaker_embed, codec_input_1], dim=1)

        # Role prefix
        role_prefix_embed = talker.text_projection(
            text_embed_layer(torch.tensor([[im_start, assistant, newline]]))
        )

        # Codec positions with tts_pad/tts_bos
        num_codec = codec_input.shape[1] - 1
        tts_text = torch.cat([
            tts_pad_embed.expand(-1, num_codec - 1, -1),
            tts_bos_embed
        ], dim=1)
        codec_hidden = tts_text + codec_input[:, :-1]

        # Combine role prefix and codec
        talker_input_embed = torch.cat([role_prefix_embed, codec_hidden], dim=1)

        # Add first text token with codec_bos
        first_text_embed = talker.text_projection(
            text_embed_layer(torch.tensor([text_tokens[:1]]))
        )
        talker_input_embed = torch.cat([
            talker_input_embed,
            first_text_embed + codec_input[:, -1:]
        ], dim=1)

        print(f"Prefill input shape: {talker_input_embed.shape}")

        # Position IDs
        batch, seq_len = talker_input_embed.shape[:2]
        pos_1d = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        position_ids = pos_1d.unsqueeze(0).expand(3, -1, -1)

        # Forward pass through talker (prefill)
        outputs = talker.model(
            inputs_embeds=talker_input_embed,
            position_ids=position_ids,
            use_cache=True,
        )
        hidden_states = outputs.last_hidden_state
        hidden_states = talker.model.norm(hidden_states)
        past_kv = outputs.past_key_values

        # Debug: print KV cache stats for layer 0
        k_cache_0 = past_kv[0][0]  # [batch, kv_heads, seq_len, head_dim]
        v_cache_0 = past_kv[0][1]
        print(f"DEBUG: KV cache layer 0 K shape: {k_cache_0.shape}")
        print(f"DEBUG: KV cache layer 0 K sum: {k_cache_0.sum().item():.4f}")
        print(f"DEBUG: KV cache layer 0 V sum: {v_cache_0.sum().item():.4f}")

        # Register hooks for frame 1 generation debugging
        hooks = []
        for layer_idx, layer in enumerate(talker.model.layers):
            hook = layer.self_attn.register_forward_pre_hook(make_pre_attn_hook(layer_idx), with_kwargs=True)
            hooks.append(hook)
            # Also register output hook on the layer itself
            hook2 = layer.register_forward_hook(make_layer_output_hook(layer_idx), with_kwargs=True)
            hooks.append(hook2)

        # Get logits and suppress special tokens
        logits = talker.codec_head(hidden_states[:, -1:])
        suppress_start = talker_config.vocab_size - 1024
        suppress_tokens = [i for i in range(suppress_start, talker_config.vocab_size)
                         if i != talker_config.codec_eos_token_id]
        logits[:, :, suppress_tokens] = float('-inf')

        # Sample first semantic token (greedy with very low temp)
        probs = torch.softmax(logits / 0.0001, dim=-1)
        first_token = torch.multinomial(probs.view(-1), 1).item()
        print(f"First semantic token: {first_token}")

        # === Frame 1 generation ===
        prev_token = first_token

        # Embed previous semantic token
        input_embed = codec_embed_layer(torch.tensor([[prev_token]]))
        embed_sum = input_embed.sum().item()
        print(f"Frame 1 DEBUG: prev_token={prev_token}, input_embed_sum={embed_sum:.4f}")

        # Position
        new_pos = seq_len + 1 - 1  # frame_idx = 1
        pos_ids = torch.tensor([[[new_pos]]]).expand(3, -1, -1)
        print(f"Frame 1 DEBUG: position={new_pos}")

        # Forward through talker
        outputs = talker.model(
            inputs_embeds=input_embed,
            position_ids=pos_ids,
            past_key_values=past_kv,
            use_cache=True,
        )
        hidden_states = outputs.last_hidden_state

        # Print layer 0 Q/K debug info from hook
        print(f"Frame 1 DEBUG Layer 0: Q_sum_before_rope={debug_data.get('q_sum_before_rope', 'N/A'):.4f}")
        print(f"Frame 1 DEBUG Layer 0: K_sum_before_rope={debug_data.get('k_sum_before_rope', 'N/A'):.4f}")
        print(f"Frame 1 DEBUG Layer 0: cos_sum={debug_data.get('cos_sum', 'N/A'):.4f}")
        print(f"Frame 1 DEBUG Layer 0: sin_sum={debug_data.get('sin_sum', 'N/A'):.4f}")

        # Print layer-by-layer hidden state sums
        print("Frame 1 DEBUG: Layer-by-layer hidden state sums:")
        for i in range(28):  # 28 layers
            key = f"layer_{i}_output_sum"
            if key in debug_data:
                print(f"  Layer {i}: {debug_data[key]:.4f}")

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Debug: print hidden state sum (note: model.forward already applies norm)
        hidden_sum_after_model = hidden_states.sum().item()
        print(f"Frame 1 DEBUG: hidden_sum after model (already normed)={hidden_sum_after_model:.4f}")

        # Get logits - note: model already applied norm, so don't apply again
        # The hidden_states from model.forward are already normalized
        print(f"Frame 1 DEBUG: codec_head weight shape: {talker.codec_head.weight.shape}")
        print(f"Frame 1 DEBUG: codec_head weight sum: {talker.codec_head.weight.sum().item():.4f}")
        print(f"Frame 1 DEBUG: codec_head bias: {talker.codec_head.bias}")

        hidden_for_logits = hidden_states[:, -1:]
        print(f"Frame 1 DEBUG: hidden_for_logits shape: {hidden_for_logits.shape}")
        print(f"Frame 1 DEBUG: hidden_for_logits sum: {hidden_for_logits.sum().item():.4f}")

        logits = talker.codec_head(hidden_for_logits)
        print(f"Frame 1 DEBUG: logits shape: {logits.shape}")
        print(f"Frame 1 DEBUG: logits sum: {logits.sum().item():.4f}")

        # Check specific logit values
        logits_flat = logits.squeeze()
        print(f"Frame 1 DEBUG: logits[0:5]: {logits_flat[0:5].tolist()}")
        print(f"Frame 1 DEBUG: logits[210]: {logits_flat[210].item():.4f}")
        print(f"Frame 1 DEBUG: logits[415]: {logits_flat[415].item():.4f}")
        print(f"Frame 1 DEBUG: logits[1028]: {logits_flat[1028].item():.4f}")

        # Top tokens before suppression
        logits_flat = logits.squeeze()
        top_vals, top_idxs = logits_flat.topk(5)
        print(f"Frame 1 DEBUG: top 5 tokens BEFORE suppression:")
        for i in range(5):
            print(f"  {top_idxs[i].item()}: {top_vals[i].item():.4f}")

        # Apply suppression
        logits[:, :, suppress_tokens] = float('-inf')
        logits_flat = logits.squeeze()
        top_vals, top_idxs = logits_flat.topk(5)
        print(f"Frame 1 DEBUG: top 5 tokens AFTER suppression:")
        for i in range(5):
            print(f"  {top_idxs[i].item()}: {top_vals[i].item():.4f}")


if __name__ == "__main__":
    main()
