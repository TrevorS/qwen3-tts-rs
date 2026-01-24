#!/usr/bin/env python3
"""Generate TTS audio using the CustomVoice model - CORRECT implementation.

This follows the official Qwen3-TTS architecture where:
1. Text and codec embeddings are ADDED together
2. Text embeddings go through text_projection first
3. Code predictor is called DURING generation, not after
4. All 16 embeddings are SUMMED for residual VQ pattern
"""

import argparse
import json
import torch
import numpy as np
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
    """Apply the ResizeMLP (2-layer MLP with SiLU).

    Structure: linear_fc1 -> SiLU -> linear_fc2
    """
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
    """Apply 3D Multi-modal RoPE (M-RoPE) used by Qwen3-TTS.

    For pure text/audio (no vision), all 3 position dimensions are identical,
    but the RoPE is still applied with the mrope_section split.

    Args:
        q, k: Query and key tensors [batch, heads, seq_len, head_dim]
        cos, sin: RoPE cos/sin values [3, batch, seq_len, head_dim]
        mrope_section: Section sizes for temporal/height/width [24, 20, 20]
        interleaved: Whether to use interleaved rope (True for Qwen3-TTS)
    """
    if interleaved:
        # Interleaved M-RoPE: interleave the sections from different modalities
        def apply_interleaved_rope(x, modality_num):
            # x shape: [3, batch, seq_len, head_dim/2]
            x_t = x[0].clone()
            for i, n in enumerate(mrope_section[1:], 1):
                beg_idx = i
                end_idx = n * modality_num
                x_t[..., beg_idx:end_idx:modality_num] = x[i, ..., beg_idx:end_idx:modality_num]
            return x_t

        dim = cos.shape[-1]
        modality_num = len(mrope_section)  # 3
        cos = torch.cat([apply_interleaved_rope(cos[..., :dim // 2], modality_num)] * 2, dim=-1).unsqueeze(1)
        sin = torch.cat([apply_interleaved_rope(sin[..., :dim // 2], modality_num)] * 2, dim=-1).unsqueeze(1)
    else:
        # Non-interleaved M-RoPE
        mrope_section_doubled = mrope_section * 2
        cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section_doubled, dim=-1))], dim=-1).unsqueeze(1)
        sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section_doubled, dim=-1))], dim=-1).unsqueeze(1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def build_rope_cache_3d(max_seq_len: int, head_dim: int, theta: float = 1000000.0):
    """Build 3D rotary position embedding cache for M-RoPE.

    Returns cos/sin with shape [max_seq_len, head_dim] that will be
    expanded to 3D during forward pass.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return torch.cos(emb), torch.sin(emb)


def get_rope_embeddings_3d(cos_cache, sin_cache, position_ids):
    """Get 3D RoPE embeddings for given position IDs.

    Args:
        cos_cache, sin_cache: Cached values [max_seq_len, head_dim]
        position_ids: 3D position IDs [3, batch, seq_len]

    Returns:
        cos, sin: [3, batch, seq_len, head_dim]
    """
    # For pure text, all 3 dimensions have the same position
    # position_ids shape: [3, batch, seq_len]
    cos = cos_cache[position_ids]  # [3, batch, seq_len, head_dim]
    sin = sin_cache[position_ids]  # [3, batch, seq_len, head_dim]
    return cos, sin


# Keep old function for code predictor which uses standard RoPE
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """Apply standard 1D rotary position embeddings (for code predictor)."""
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def build_rope_cache(max_seq_len: int, head_dim: int, theta: float = 1000000.0):
    """Build standard 1D rotary position embedding cache."""
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
    attention_mask: torch.Tensor = None,
    past_key_value: tuple = None,
    use_cache: bool = False,
    use_mrope: bool = False,
    mrope_section: list = None,
):
    """Single transformer layer forward pass with optional KV cache.

    Args:
        use_mrope: Whether to use 3D Multi-modal RoPE (for talker)
        mrope_section: Section sizes for M-RoPE [24, 20, 20] (for talker)
    """
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

    # Apply QK normalization (RMSNorm on head_dim)
    q = rms_norm(q, weights[f"{prefix}.self_attn.q_norm.weight"], eps)
    k = rms_norm(k, weights[f"{prefix}.self_attn.k_norm.weight"], eps)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Apply RoPE
    if use_mrope and mrope_section is not None:
        # 3D M-RoPE for talker
        # position_ids should be [3, batch, seq_len]
        cos = rope_cos[position_ids]  # [3, batch, seq_len, head_dim]
        sin = rope_sin[position_ids]  # [3, batch, seq_len, head_dim]
        q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, interleaved=True)
    else:
        # Standard 1D RoPE for code predictor
        q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin, position_ids)

    # Handle KV cache
    if past_key_value is not None:
        past_k, past_v = past_key_value
        k = torch.cat([past_k, k], dim=2)
        v = torch.cat([past_v, v], dim=2)

    new_past_key_value = (k, v) if use_cache else None

    # Repeat KV for GQA
    if n_kv_heads < n_heads:
        n_rep = n_heads // n_kv_heads
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)

    # Attention
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
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

    if use_cache:
        return hidden_states, new_past_key_value
    return hidden_states


def talker_forward(
    inputs_embeds: torch.Tensor,
    weights: dict,
    config: dict,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor = None,
    past_key_values: list = None,
    use_cache: bool = False,
):
    """Run forward pass through talker transformer.

    Uses 3D M-RoPE with position_ids shape [3, batch, seq_len].
    For pure text/audio, all 3 position dimensions are identical.
    """
    talker_config = config["talker_config"]
    n_layers = talker_config["num_hidden_layers"]
    n_heads = talker_config["num_attention_heads"]
    n_kv_heads = talker_config["num_key_value_heads"]
    head_dim = talker_config["head_dim"]
    intermediate_size = talker_config["intermediate_size"]
    eps = talker_config["rms_norm_eps"]

    # M-RoPE configuration
    mrope_section = talker_config.get("mrope_section", [24, 20, 20])

    hidden_states = inputs_embeds
    new_past_key_values = [] if use_cache else None

    for layer_idx in range(n_layers):
        prefix = f"talker.model.layers.{layer_idx}"
        past_kv = past_key_values[layer_idx] if past_key_values else None

        if use_cache:
            hidden_states, new_kv = transformer_layer(
                hidden_states, weights, prefix,
                n_heads, n_kv_heads, head_dim, intermediate_size,
                rope_cos, rope_sin, position_ids, eps,
                attention_mask=attention_mask,
                past_key_value=past_kv,
                use_cache=True,
                use_mrope=True,
                mrope_section=mrope_section,
            )
            new_past_key_values.append(new_kv)
        else:
            hidden_states = transformer_layer(
                hidden_states, weights, prefix,
                n_heads, n_kv_heads, head_dim, intermediate_size,
                rope_cos, rope_sin, position_ids, eps,
                attention_mask=attention_mask,
                use_mrope=True,
                mrope_section=mrope_section,
            )

    hidden_states = rms_norm(hidden_states, weights["talker.model.norm.weight"], eps)

    return hidden_states, new_past_key_values


def code_predictor_generate(
    past_hidden: torch.Tensor,
    last_id_hidden: torch.Tensor,
    weights: dict,
    config: dict,
    temperature: float = 1.0,
    top_k: int = 50,
):
    """Generate 15 acoustic codes using the code predictor.

    Args:
        past_hidden: Hidden state from talker [1, 1, 2048]
        last_id_hidden: Embedding of the semantic token [1, 1, 2048]
        weights: Model weights
        config: Model config

    Returns:
        acoustic_codes: [15] tensor of acoustic codes
        acoustic_embeds: List of 15 embeddings [1, 1, 2048] each
    """
    cp_config = config["talker_config"]["code_predictor_config"]
    hidden_size = cp_config["hidden_size"]  # 1024
    n_layers = cp_config["num_hidden_layers"]  # 5
    n_heads = cp_config["num_attention_heads"]  # 16
    n_kv_heads = cp_config["num_key_value_heads"]  # 8
    head_dim = cp_config["head_dim"]  # 128
    intermediate_size = cp_config["intermediate_size"]  # 3072
    eps = cp_config["rms_norm_eps"]
    rope_theta = cp_config["rope_theta"]

    # Build RoPE cache for code predictor
    rope_cos, rope_sin = build_rope_cache(256, head_dim, rope_theta)

    # Initial input: [past_hidden, last_id_hidden]
    inputs_embeds = torch.cat([past_hidden, last_id_hidden], dim=1)  # [1, 2, 2048]

    # Project to code predictor hidden size
    small_to_mtp = weights["talker.code_predictor.small_to_mtp_projection.weight"]
    small_to_mtp_b = weights["talker.code_predictor.small_to_mtp_projection.bias"]
    inputs_embeds = torch.nn.functional.linear(inputs_embeds, small_to_mtp, small_to_mtp_b)
    # Now [1, 2, 1024]

    position_ids = torch.arange(2, dtype=torch.long).unsqueeze(0)

    # Run through code predictor layers (prefill)
    hidden_states = inputs_embeds
    past_key_values = []
    for layer_idx in range(n_layers):
        prefix = f"talker.code_predictor.model.layers.{layer_idx}"
        hidden_states, new_kv = transformer_layer(
            hidden_states, weights, prefix,
            n_heads, n_kv_heads, head_dim, intermediate_size,
            rope_cos, rope_sin, position_ids, eps,
            use_cache=True,
        )
        past_key_values.append(new_kv)

    hidden_states = rms_norm(hidden_states, weights["talker.code_predictor.model.norm.weight"], eps)

    # Generate 15 acoustic codes autoregressively
    acoustic_codes = []
    acoustic_embeds = []

    # Get first code logits (use lm_head.0 on the last position)
    logits = torch.nn.functional.linear(
        hidden_states[:, -1:, :],
        weights["talker.code_predictor.lm_head.0.weight"]
    )
    logits = logits[:, -1, :] / temperature
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
    probs = torch.softmax(logits, dim=-1)
    code = torch.multinomial(probs, num_samples=1).item()
    acoustic_codes.append(code)

    # Get embedding for this code (using code_predictor's embedding layer 0)
    # Note: embeddings output 2048 dim, then projected to 1024
    embed_weight = weights["talker.code_predictor.model.codec_embedding.0.weight"]
    code_embed = embed_weight[code].unsqueeze(0).unsqueeze(0)  # [1, 1, 2048]
    acoustic_embeds.append(code_embed)

    # Generate remaining 14 codes
    for step in range(1, 15):
        # Project embedding to code predictor dim
        inputs_embeds = torch.nn.functional.linear(code_embed, small_to_mtp, small_to_mtp_b)

        position_ids = torch.tensor([[2 + step]], dtype=torch.long)

        # Run through layers with KV cache
        hidden_states = inputs_embeds
        new_past_key_values = []
        for layer_idx in range(n_layers):
            prefix = f"talker.code_predictor.model.layers.{layer_idx}"
            hidden_states, new_kv = transformer_layer(
                hidden_states, weights, prefix,
                n_heads, n_kv_heads, head_dim, intermediate_size,
                rope_cos, rope_sin, position_ids, eps,
                past_key_value=past_key_values[layer_idx],
                use_cache=True,
            )
            new_past_key_values.append(new_kv)
        past_key_values = new_past_key_values

        hidden_states = rms_norm(hidden_states, weights["talker.code_predictor.model.norm.weight"], eps)

        # Get logits for this code group
        logits = torch.nn.functional.linear(
            hidden_states[:, -1:, :],
            weights[f"talker.code_predictor.lm_head.{step}.weight"]
        )
        logits = logits[:, -1, :] / temperature
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        probs = torch.softmax(logits, dim=-1)
        code = torch.multinomial(probs, num_samples=1).item()
        acoustic_codes.append(code)

        # Get embedding for next step
        embed_weight = weights[f"talker.code_predictor.model.codec_embedding.{step}.weight"]
        code_embed = embed_weight[code].unsqueeze(0).unsqueeze(0)
        acoustic_embeds.append(code_embed)

    return torch.tensor(acoustic_codes), acoustic_embeds


def generate_with_custom_voice(
    text_tokens: list,
    speaker_id: int,
    weights: dict,
    config: dict,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    top_k: int = 50,
    language: str = "english",
):
    """Generate speech using the CustomVoice model.

    This follows the official architecture:
    1. Build input with text + codec embeddings added together
    2. For each generation step, call code predictor to get acoustic codes
    3. Sum all 16 embeddings for residual VQ pattern
    """
    talker_config = config["talker_config"]
    hidden_size = talker_config["hidden_size"]
    head_dim = talker_config["head_dim"]
    rope_theta = talker_config["rope_theta"]

    # Special token IDs
    codec_bos_id = talker_config["codec_bos_id"]  # 2149
    codec_eos_id = talker_config["codec_eos_token_id"]  # 2150
    codec_pad_id = talker_config["codec_pad_id"]  # 2148
    codec_think_id = talker_config["codec_think_id"]  # 2154
    codec_nothink_id = talker_config["codec_nothink_id"]  # 2155
    codec_think_bos_id = talker_config["codec_think_bos_id"]  # 2156
    codec_think_eos_id = talker_config["codec_think_eos_id"]  # 2157

    # Language ID
    language_id = talker_config["codec_language_id"].get(language.lower(), 2050)

    # TTS special token IDs (from main config)
    tts_bos_id = config["tts_bos_token_id"]  # 151672
    tts_eos_id = config["tts_eos_token_id"]  # 151673
    tts_pad_id = config["tts_pad_token_id"]  # 151671

    print(f"Special tokens: bos={codec_bos_id}, eos={codec_eos_id}, pad={codec_pad_id}")
    print(f"TTS tokens: bos={tts_bos_id}, eos={tts_eos_id}, pad={tts_pad_id}")
    print(f"Language: {language} (ID: {language_id})")
    print(f"Speaker ID: {speaker_id}")

    # Get embeddings
    text_embed_weight = weights["talker.model.text_embedding.weight"]
    codec_embed_weight = weights["talker.model.codec_embedding.weight"]

    # Build RoPE cache
    rope_cos, rope_sin = build_rope_cache(8192, head_dim, rope_theta)

    # Role prefix tokens: <|im_start|>assistant\n
    im_start_id = config["im_start_token_id"]  # 151644
    assistant_id = config["assistant_token_id"]  # 77091
    newline_id = 198  # \n token

    # Step 1: Get TTS special embeddings via text_projection
    tts_special_embeds = resize_mlp(
        text_embed_weight[[tts_bos_id, tts_eos_id, tts_pad_id]].unsqueeze(0),
        weights,
        "talker.text_projection"
    )
    tts_bos_embed = tts_special_embeds[:, 0:1, :]  # [1, 1, 2048]
    tts_eos_embed = tts_special_embeds[:, 1:2, :]
    tts_pad_embed = tts_special_embeds[:, 2:3, :]

    # Step 2: Build role prefix: [im_start, assistant, \n]
    role_prefix_ids = [im_start_id, assistant_id, newline_id]
    role_prefix_embed = resize_mlp(
        text_embed_weight[role_prefix_ids].unsqueeze(0),
        weights,
        "talker.text_projection"
    )  # [1, 3, 2048]

    # Step 3: Build codec prefix
    # Try nothink mode: [codec_nothink_id, codec_think_bos_id, codec_think_eos_id]
    # Or with language: [codec_think_id, codec_think_bos_id, language_id, codec_think_eos_id]
    use_nothink = False  # Set to True to try nothink mode
    if use_nothink:
        codec_prefix = [codec_nothink_id, codec_think_bos_id, codec_think_eos_id]
    else:
        codec_prefix = [codec_think_id, codec_think_bos_id, language_id, codec_think_eos_id]
    codec_prefix_embed = codec_embed_weight[codec_prefix].unsqueeze(0)

    # Speaker embedding
    speaker_embed = codec_embed_weight[speaker_id].unsqueeze(0).unsqueeze(0)  # [1, 1, 2048]

    # Codec pad and bos
    codec_pad_bos_embed = codec_embed_weight[[codec_pad_id, codec_bos_id]].unsqueeze(0)  # [1, 2, 2048]

    # Full codec control sequence: [think, think_bos, lang, think_eos, speaker, pad, bos]
    codec_control_embed = torch.cat([
        codec_prefix_embed,  # [1, 4, 2048]
        speaker_embed,        # [1, 1, 2048]
        codec_pad_bos_embed,  # [1, 2, 2048]
    ], dim=1)  # [1, 7, 2048]

    # Step 4: Build text control sequence
    # tts_pad repeated for control positions (excluding last 2: pad and bos), then tts_bos
    text_control_embed = torch.cat([
        tts_pad_embed.expand(-1, codec_control_embed.shape[1] - 2, -1),  # [1, 5, 2048]
        tts_bos_embed,  # [1, 1, 2048]
    ], dim=1)  # [1, 6, 2048]

    # Step 5: Combine text + codec for control sequence (ADD them!)
    # Exclude last codec position (bos) which will be added with first text token
    control_embed = text_control_embed + codec_control_embed[:, :-1, :]
    # [1, 6, 2048]

    # Step 6: Add first text token with codec_bos
    if len(text_tokens) > 0:
        first_text_embed = resize_mlp(
            text_embed_weight[text_tokens[0]].unsqueeze(0).unsqueeze(0),
            weights,
            "talker.text_projection"
        )  # [1, 1, 2048]
        first_input = first_text_embed + codec_control_embed[:, -1:, :]  # ADD with codec_bos
    else:
        first_input = tts_pad_embed + codec_control_embed[:, -1:, :]

    # Full initial input: [role_prefix, control_embed, first_text]
    initial_input = torch.cat([role_prefix_embed, control_embed, first_input], dim=1)
    # [1, 3 + 6 + 1 = 10, 2048]

    print(f"Initial input shape: {initial_input.shape}")

    # Trailing text for streaming mode
    if len(text_tokens) > 1:
        trailing_text_ids = text_tokens[1:]
        trailing_text_embed = resize_mlp(
            text_embed_weight[trailing_text_ids].unsqueeze(0),
            weights,
            "talker.text_projection"
        )
        trailing_text_hidden = torch.cat([trailing_text_embed, tts_eos_embed], dim=1)
    else:
        trailing_text_hidden = tts_eos_embed

    print(f"Trailing text shape: {trailing_text_hidden.shape}")

    # Step 6: Run initial prefill through talker
    seq_len = initial_input.shape[1]
    # 3D position IDs for M-RoPE: [3, batch, seq_len]
    # For pure text/audio, all 3 dimensions have the same positions
    pos_1d = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)  # [1, seq_len]
    position_ids = pos_1d.unsqueeze(0).expand(3, -1, -1)  # [3, 1, seq_len]

    hidden_states, past_key_values = talker_forward(
        initial_input, weights, config, rope_cos, rope_sin, position_ids,
        use_cache=True,
    )

    # Suppress special tokens >= 2048 (except EOS 2150) during generation
    # This matches the official code: suppress range(vocab_size - 1024, vocab_size) except eos
    vocab_size = talker_config["vocab_size"]  # 3072
    suppress_start = vocab_size - 1024  # 2048
    suppress_tokens = [i for i in range(suppress_start, vocab_size) if i != codec_eos_id]

    # Get first semantic token
    logits = torch.nn.functional.linear(hidden_states[:, -1:, :], weights["talker.codec_head.weight"])
    logits = logits[:, -1, :] / temperature
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
    # Suppress special tokens
    logits[:, suppress_tokens] = float('-inf')
    probs = torch.softmax(logits, dim=-1)
    semantic_token = torch.multinomial(probs, num_samples=1).item()

    print(f"First semantic token: {semantic_token}")

    # Generation loop
    all_semantic_tokens = [semantic_token]
    all_acoustic_codes = []
    generation_step = 0
    current_pos = seq_len

    for step in range(max_new_tokens):
        if semantic_token == codec_eos_id:
            print(f"Hit EOS at step {step}")
            break

        # Get semantic token embedding
        last_id_hidden = codec_embed_weight[semantic_token].unsqueeze(0).unsqueeze(0)  # [1, 1, 2048]
        past_hidden = hidden_states[:, -1:, :]  # [1, 1, 2048]

        # Generate 15 acoustic codes using code predictor
        acoustic_codes, acoustic_embeds = code_predictor_generate(
            past_hidden, last_id_hidden, weights, config, temperature, top_k
        )
        all_acoustic_codes.append([semantic_token] + acoustic_codes.tolist())

        if step % 10 == 0:
            print(f"Step {step}: semantic={semantic_token}, acoustic[0:3]={acoustic_codes[:3].tolist()}")

        # Sum all 16 embeddings (residual VQ pattern)
        all_embeds = [last_id_hidden] + acoustic_embeds  # 16 embeddings
        summed_embeds = torch.cat(all_embeds, dim=1).sum(1, keepdim=True)  # [1, 1, 2048]

        # Add trailing text if available
        if generation_step < trailing_text_hidden.shape[1]:
            inputs_embeds = summed_embeds + trailing_text_hidden[:, generation_step:generation_step+1, :]
        else:
            inputs_embeds = summed_embeds + tts_pad_embed

        # Run through talker with KV cache
        # Position should be current_pos (after prefill of seq_len tokens, next is seq_len)
        # 3D position IDs for M-RoPE: [3, batch, seq_len]
        position_ids = torch.tensor([[[current_pos]]] * 3, dtype=torch.long)
        current_pos += 1  # Increment AFTER using

        hidden_states, past_key_values = talker_forward(
            inputs_embeds, weights, config, rope_cos, rope_sin, position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )

        # Sample next semantic token
        logits = torch.nn.functional.linear(hidden_states[:, -1:, :], weights["talker.codec_head.weight"])
        logits = logits[:, -1, :] / temperature
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        # Suppress special tokens
        logits[:, suppress_tokens] = float('-inf')
        probs = torch.softmax(logits, dim=-1)
        semantic_token = torch.multinomial(probs, num_samples=1).item()

        all_semantic_tokens.append(semantic_token)
        generation_step += 1

    print(f"Generated {len(all_semantic_tokens)} semantic tokens")
    print(f"Generated {len(all_acoustic_codes)} frames with acoustic codes")

    # Build output codes: [semantic, a1, a2, ..., a15] per frame
    codes = np.array(all_acoustic_codes, dtype=np.int64)
    return all_semantic_tokens, codes


def main():
    parser = argparse.ArgumentParser(description="Generate TTS with CustomVoice model (correct implementation)")
    parser.add_argument("--text", type=str, default="Hello", help="Text to synthesize")
    parser.add_argument("--speaker", type=str, default="ryan", help="Speaker name")
    parser.add_argument("--language", type=str, default="english", help="Language")
    parser.add_argument("--model-dir", type=str, default="../test_data/model_customvoice", help="Model directory")
    parser.add_argument("--output", type=str, default="../test_data/customvoice_correct.npy", help="Output codes file")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max frames to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_dir = Path(args.model_dir)
    print(f"Loading model from {model_dir}")

    # Load config
    with open(model_dir / "config.json") as f:
        config = json.load(f)

    # Get speaker ID
    speaker_ids = config["talker_config"]["spk_id"]
    if args.speaker.lower() not in speaker_ids:
        print(f"Unknown speaker: {args.speaker}")
        print(f"Available speakers: {list(speaker_ids.keys())}")
        return
    speaker_id = speaker_ids[args.speaker.lower()]
    print(f"Speaker: {args.speaker} (ID: {speaker_id})")

    # Simple text to token mapping
    text_tokens_map = {
        "Hello": [9707],
        "Hello world": [9707, 1917],
        "Hello, this is a test": [9707, 11, 419, 374, 264, 1273],
        "Hi": [13347],
    }
    if args.text not in text_tokens_map:
        print(f"Text '{args.text}' not in hardcoded mapping, using 'Hello'")
        text_tokens = text_tokens_map["Hello"]
    else:
        text_tokens = text_tokens_map[args.text]
    print(f"Text tokens: {text_tokens}")

    # Load model weights
    print("Loading model weights...")
    weights = load_file(model_dir / "model.safetensors")
    # Convert all weights to float32 for consistent computation
    weights = {k: v.float() for k, v in weights.items()}
    print(f"Loaded {len(weights)} weight tensors")

    # Generate
    print("\n=== Generating ===")
    semantic_tokens, codes = generate_with_custom_voice(
        text_tokens,
        speaker_id,
        weights,
        config,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        language=args.language,
    )

    # Save codes
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), codes)
    print(f"\nSaved codes to {output_path}: shape {codes.shape}")
    print(f"First few frames:\n{codes[:5]}")

    print("\nTo decode to audio, run:")
    print(f"  uv run python decode_with_official.py --codes-file {output_path}")


if __name__ == "__main__":
    main()
