#!/usr/bin/env python3
"""Export end-to-end reference values for full TTS pipeline validation.

This script runs the complete pipeline:
1. Text tokenization
2. Talker (28-layer transformer) -> semantic tokens
3. Code Predictor (5-layer transformer) -> acoustic tokens
4. Decoder -> audio waveform
"""

import json
import numpy as np
import torch
from pathlib import Path
from safetensors.torch import load_file


def export_e2e_reference():
    """Export reference values for full TTS pipeline."""

    device = torch.device("cpu")
    output_dir = Path("test_data/reference_values")
    output_dir.mkdir(exist_ok=True)

    # Load model weights
    print("Loading model weights...")
    weights = load_file("test_data/model/model.safetensors")
    weights = {k: v.float() for k, v in weights.items()}

    # Load speech tokenizer weights
    print("Loading speech tokenizer weights...")
    decoder_weights = load_file("test_data/speech_tokenizer/model.safetensors")
    decoder_weights = {k: v.float() for k, v in decoder_weights.items()}

    # Load configs
    with open("test_data/speech_tokenizer/config.json") as f:
        decoder_config = json.load(f)["decoder_config"]

    # Test input: simple tokens
    input_ids = torch.tensor([[9707, 11, 419, 374, 264]], dtype=torch.long)  # "Hello, this is a"
    batch_size, seq_len = input_ids.shape
    print(f"Input IDs: {input_ids.tolist()}")

    # ===== Config values =====
    hidden_size = 1024
    num_heads = 16
    num_kv_heads = 8
    head_dim = 128
    num_layers = 28
    rope_theta = 1000000.0
    eps = 1e-6
    n_rep = num_heads // num_kv_heads

    # Code predictor config
    cp_num_layers = 5
    cp_hidden_size = 1024
    cp_num_heads = 16
    cp_num_kv_heads = 8
    cp_head_dim = 128
    num_code_groups = 16

    # ===== Helper functions =====
    def rms_norm(x, weight, eps):
        variance = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + eps)
        return weight * x_norm

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def repeat_kv(x, n_rep):
        if n_rep == 1:
            return x
        batch, n_kv_heads, seq, hd = x.shape
        x = x[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, seq, hd)
        return x.reshape(batch, n_kv_heads * n_rep, seq, hd)

    # ===== 1. Text Embedding & Projection =====
    print("\n=== Step 1: Text Embedding & Projection ===")
    text_embed_w = weights["talker.model.text_embedding.weight"]
    text_embeddings = torch.nn.functional.embedding(input_ids, text_embed_w)

    fc1_w = weights["talker.text_projection.linear_fc1.weight"]
    fc1_b = weights["talker.text_projection.linear_fc1.bias"]
    fc2_w = weights["talker.text_projection.linear_fc2.weight"]
    fc2_b = weights["talker.text_projection.linear_fc2.bias"]

    hidden = torch.nn.functional.linear(text_embeddings, fc1_w, fc1_b)
    hidden = torch.nn.functional.silu(hidden)
    hidden = torch.nn.functional.linear(hidden, fc2_w, fc2_b)
    print(f"After text projection: {hidden.shape}")

    # ===== 2. Talker (28-layer transformer) =====
    print("\n=== Step 2: Talker (28 layers) ===")

    # Build RoPE
    positions = torch.arange(seq_len).unsqueeze(0)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = torch.outer(positions.float().squeeze(0), inv_freq)
    cos = freqs.cos().repeat(1, 2).unsqueeze(0).unsqueeze(0)
    sin = freqs.sin().repeat(1, 2).unsqueeze(0).unsqueeze(0)

    # Causal mask
    causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)

    scaling = head_dim ** -0.5

    for layer_idx in range(num_layers):
        # Input LayerNorm
        input_ln_w = weights[f"talker.model.layers.{layer_idx}.input_layernorm.weight"]
        normed = rms_norm(hidden, input_ln_w, eps)

        # QKV projections
        q_proj_w = weights[f"talker.model.layers.{layer_idx}.self_attn.q_proj.weight"]
        k_proj_w = weights[f"talker.model.layers.{layer_idx}.self_attn.k_proj.weight"]
        v_proj_w = weights[f"talker.model.layers.{layer_idx}.self_attn.v_proj.weight"]
        q_norm_w = weights[f"talker.model.layers.{layer_idx}.self_attn.q_norm.weight"]
        k_norm_w = weights[f"talker.model.layers.{layer_idx}.self_attn.k_norm.weight"]

        q = torch.nn.functional.linear(normed, q_proj_w)
        q = q.view(batch_size, seq_len, num_heads, head_dim)
        q = rms_norm(q, q_norm_w, eps)
        q = q.transpose(1, 2)

        k = torch.nn.functional.linear(normed, k_proj_w)
        k = k.view(batch_size, seq_len, num_kv_heads, head_dim)
        k = rms_norm(k, k_norm_w, eps)
        k = k.transpose(1, 2)

        v = torch.nn.functional.linear(normed, v_proj_w)
        v = v.view(batch_size, seq_len, num_kv_heads, head_dim)
        v = v.transpose(1, 2)

        # RoPE
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)

        # Attention
        k_exp = repeat_kv(k, n_rep)
        v_exp = repeat_kv(v, n_rep)

        attn_weights = torch.matmul(q, k_exp.transpose(2, 3)) * scaling
        attn_weights = attn_weights + causal_mask
        attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_output = torch.matmul(attn_probs, v_exp)

        # O projection
        o_proj_w = weights[f"talker.model.layers.{layer_idx}.self_attn.o_proj.weight"]
        attn_output_flat = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        attn_proj = torch.nn.functional.linear(attn_output_flat, o_proj_w)
        hidden = hidden + attn_proj

        # MLP
        post_ln_w = weights[f"talker.model.layers.{layer_idx}.post_attention_layernorm.weight"]
        mlp_input = rms_norm(hidden, post_ln_w, eps)

        gate_w = weights[f"talker.model.layers.{layer_idx}.mlp.gate_proj.weight"]
        up_w = weights[f"talker.model.layers.{layer_idx}.mlp.up_proj.weight"]
        down_w = weights[f"talker.model.layers.{layer_idx}.mlp.down_proj.weight"]

        gate = torch.nn.functional.linear(mlp_input, gate_w)
        up = torch.nn.functional.linear(mlp_input, up_w)
        mlp_hidden = torch.nn.functional.silu(gate) * up
        mlp_output = torch.nn.functional.linear(mlp_hidden, down_w)
        hidden = hidden + mlp_output

        if layer_idx % 7 == 0:
            print(f"  Layer {layer_idx}: mean={hidden.mean().item():.6f}")

    # Final norm
    final_norm_w = weights["talker.model.norm.weight"]
    hidden = rms_norm(hidden, final_norm_w, eps)

    # Codec head -> semantic tokens
    codec_head_w = weights["talker.codec_head.weight"]
    codec_logits = torch.nn.functional.linear(hidden, codec_head_w)
    semantic_tokens = codec_logits.argmax(dim=-1)
    print(f"Semantic tokens: {semantic_tokens.tolist()}")

    # ===== 3. Code Predictor (5-layer transformer) =====
    print("\n=== Step 3: Code Predictor ===")

    # Get last hidden state and semantic token
    last_hidden = hidden[:, -1:, :]  # [1, 1, 1024]
    last_semantic = semantic_tokens[:, -1]  # scalar

    # Get semantic embedding
    codec_embed_w = weights["talker.model.codec_embedding.weight"]
    semantic_embed = codec_embed_w[last_semantic].unsqueeze(0)  # [1, 1, 1024]

    # Code predictor input: [hidden_state, semantic_embed]
    cp_input = torch.cat([last_hidden, semantic_embed], dim=1)  # [1, 2, 1024]
    cp_seq_len = cp_input.shape[1]

    # Build RoPE for code predictor
    cp_positions = torch.arange(cp_seq_len).unsqueeze(0)
    cp_inv_freq = 1.0 / (rope_theta ** (torch.arange(0, cp_head_dim, 2, dtype=torch.float32) / cp_head_dim))
    cp_freqs = torch.outer(cp_positions.float().squeeze(0), cp_inv_freq)
    cp_cos = cp_freqs.cos().repeat(1, 2).unsqueeze(0).unsqueeze(0)
    cp_sin = cp_freqs.sin().repeat(1, 2).unsqueeze(0).unsqueeze(0)

    # Causal mask
    cp_causal = torch.triu(torch.full((cp_seq_len, cp_seq_len), float("-inf")), diagonal=1)

    cp_hidden = cp_input

    for layer_idx in range(cp_num_layers):
        prefix = f"talker.code_predictor.model.layers.{layer_idx}"

        input_ln_w = weights[f"{prefix}.input_layernorm.weight"]
        normed = rms_norm(cp_hidden, input_ln_w, eps)

        q_proj_w = weights[f"{prefix}.self_attn.q_proj.weight"]
        k_proj_w = weights[f"{prefix}.self_attn.k_proj.weight"]
        v_proj_w = weights[f"{prefix}.self_attn.v_proj.weight"]
        q_norm_w = weights[f"{prefix}.self_attn.q_norm.weight"]
        k_norm_w = weights[f"{prefix}.self_attn.k_norm.weight"]

        q = torch.nn.functional.linear(normed, q_proj_w)
        q = q.view(1, cp_seq_len, cp_num_heads, cp_head_dim)
        q = rms_norm(q, q_norm_w, eps)
        q = q.transpose(1, 2)

        k = torch.nn.functional.linear(normed, k_proj_w)
        k = k.view(1, cp_seq_len, cp_num_kv_heads, cp_head_dim)
        k = rms_norm(k, k_norm_w, eps)
        k = k.transpose(1, 2)

        v = torch.nn.functional.linear(normed, v_proj_w)
        v = v.view(1, cp_seq_len, cp_num_kv_heads, cp_head_dim)
        v = v.transpose(1, 2)

        # RoPE
        q = (q * cp_cos) + (rotate_half(q) * cp_sin)
        k = (k * cp_cos) + (rotate_half(k) * cp_sin)

        # Attention
        k_exp = repeat_kv(k, n_rep)
        v_exp = repeat_kv(v, n_rep)

        attn_weights = torch.matmul(q, k_exp.transpose(2, 3)) * scaling
        attn_weights = attn_weights + cp_causal
        attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_output = torch.matmul(attn_probs, v_exp)

        o_proj_w = weights[f"{prefix}.self_attn.o_proj.weight"]
        attn_output_flat = attn_output.transpose(1, 2).reshape(1, cp_seq_len, -1)
        attn_proj = torch.nn.functional.linear(attn_output_flat, o_proj_w)
        cp_hidden = cp_hidden + attn_proj

        # MLP
        post_ln_w = weights[f"{prefix}.post_attention_layernorm.weight"]
        mlp_input = rms_norm(cp_hidden, post_ln_w, eps)

        gate_w = weights[f"{prefix}.mlp.gate_proj.weight"]
        up_w = weights[f"{prefix}.mlp.up_proj.weight"]
        down_w = weights[f"{prefix}.mlp.down_proj.weight"]

        gate = torch.nn.functional.linear(mlp_input, gate_w)
        up = torch.nn.functional.linear(mlp_input, up_w)
        mlp_hidden = torch.nn.functional.silu(gate) * up
        mlp_output = torch.nn.functional.linear(mlp_hidden, down_w)
        cp_hidden = cp_hidden + mlp_output

    # Final norm
    cp_norm_w = weights["talker.code_predictor.model.norm.weight"]
    cp_hidden = rms_norm(cp_hidden, cp_norm_w, eps)

    # Generate acoustic tokens using lm_heads
    acoustic_tokens = []
    for i in range(num_code_groups - 1):  # 15 acoustic heads
        lm_head_w = weights[f"talker.code_predictor.lm_head.{i}.weight"]
        # Position 1 (semantic embed) predicts acoustic token 0
        logits = torch.nn.functional.linear(cp_hidden[:, 1:2, :], lm_head_w)
        token = logits.argmax(dim=-1).item()
        acoustic_tokens.append(token)

    print(f"Acoustic tokens: {acoustic_tokens}")

    # ===== 4. Build codes tensor for decoder =====
    print("\n=== Step 4: Build codes for decoder ===")

    # For this test, use the last semantic token and the 15 acoustic tokens
    # Shape: [batch, num_quantizers, seq_len=1]
    codes = torch.zeros((1, 16, 1), dtype=torch.long)
    codes[0, 0, 0] = last_semantic.item()
    for i, tok in enumerate(acoustic_tokens):
        codes[0, i + 1, 0] = tok

    print(f"Codes shape: {codes.shape}")
    print(f"Codes: {codes.squeeze().tolist()}")

    # ===== 5. Decoder =====
    print("\n=== Step 5: Decoder ===")

    # 5.1 Quantizer decode - normalize by cluster_usage.clamp(min=epsilon) as per official implementation
    epsilon = 1e-7
    first_embedding_sum = decoder_weights["decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"]
    first_cluster_usage = decoder_weights["decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage"]
    first_codebook = first_embedding_sum / first_cluster_usage.clamp(min=epsilon).unsqueeze(-1)

    rest_codebooks = []
    for i in range(15):
        embedding_sum = decoder_weights[f"decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum"]
        cluster_usage = decoder_weights[f"decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.cluster_usage"]
        rest_codebooks.append(embedding_sum / cluster_usage.clamp(min=epsilon).unsqueeze(-1))

    embeddings = []
    embeddings.append(first_codebook[codes[:, 0, :]])
    for i in range(15):
        embeddings.append(rest_codebooks[i][codes[:, i + 1, :]])

    quantized = torch.stack(embeddings, dim=0).sum(dim=0)  # [batch, seq, 256]

    output_proj_w = decoder_weights["decoder.quantizer.rvq_first.output_proj.weight"].squeeze(-1)
    quantized = torch.nn.functional.linear(quantized, output_proj_w)
    print(f"Quantized: {quantized.shape}")

    # 5.2 Pre-conv
    pre_conv_w = decoder_weights["decoder.pre_conv.conv.weight"]
    pre_conv_b = decoder_weights["decoder.pre_conv.conv.bias"]
    kernel_size = pre_conv_w.shape[2]
    x = quantized.transpose(1, 2)
    x = torch.nn.functional.pad(x, (kernel_size - 1, 0))
    x = torch.nn.functional.conv1d(x, pre_conv_w, pre_conv_b)
    print(f"Pre-conv: {x.shape}")

    # 5.3 Pre-transformer
    dec_hidden_size = decoder_config["hidden_size"]
    dec_num_layers = decoder_config["num_hidden_layers"]
    dec_num_heads = decoder_config["num_attention_heads"]
    dec_head_dim = decoder_config["head_dim"]
    dec_eps = decoder_config["rms_norm_eps"]
    dec_rope_theta = decoder_config["rope_theta"]

    input_proj_w = decoder_weights["decoder.pre_transformer.input_proj.weight"]
    input_proj_b = decoder_weights["decoder.pre_transformer.input_proj.bias"]

    dec_seq_len = x.shape[2]
    x = x.transpose(1, 2)
    x = torch.nn.functional.linear(x, input_proj_w, input_proj_b)

    # Build RoPE for decoder
    dec_positions = torch.arange(dec_seq_len).unsqueeze(0)
    dec_inv_freq = 1.0 / (dec_rope_theta ** (torch.arange(0, dec_head_dim, 2, dtype=torch.float32) / dec_head_dim))
    dec_freqs = torch.outer(dec_positions.float().squeeze(0), dec_inv_freq)
    dec_cos = dec_freqs.cos().repeat(1, 2).unsqueeze(0).unsqueeze(0)
    dec_sin = dec_freqs.sin().repeat(1, 2).unsqueeze(0).unsqueeze(0)

    dec_causal = torch.triu(torch.full((dec_seq_len, dec_seq_len), float("-inf")), diagonal=1)

    dec_scaling = dec_head_dim ** -0.5

    for layer_idx in range(dec_num_layers):
        prefix = f"decoder.pre_transformer.layers.{layer_idx}"

        ln_w = decoder_weights[f"{prefix}.input_layernorm.weight"]
        normed = rms_norm(x, ln_w, dec_eps)

        q_proj_w = decoder_weights[f"{prefix}.self_attn.q_proj.weight"]
        k_proj_w = decoder_weights[f"{prefix}.self_attn.k_proj.weight"]
        v_proj_w = decoder_weights[f"{prefix}.self_attn.v_proj.weight"]
        o_proj_w = decoder_weights[f"{prefix}.self_attn.o_proj.weight"]

        q = torch.nn.functional.linear(normed, q_proj_w)
        k = torch.nn.functional.linear(normed, k_proj_w)
        v = torch.nn.functional.linear(normed, v_proj_w)

        q = q.view(1, dec_seq_len, dec_num_heads, dec_head_dim).transpose(1, 2)
        k = k.view(1, dec_seq_len, dec_num_heads, dec_head_dim).transpose(1, 2)
        v = v.view(1, dec_seq_len, dec_num_heads, dec_head_dim).transpose(1, 2)

        q = (q * dec_cos) + (rotate_half(q) * dec_sin)
        k = (k * dec_cos) + (rotate_half(k) * dec_sin)

        attn = torch.matmul(q, k.transpose(2, 3)) * dec_scaling
        attn = attn + dec_causal
        attn = torch.nn.functional.softmax(attn, dim=-1)
        attn_out = torch.matmul(attn, v)

        attn_out = attn_out.transpose(1, 2).reshape(1, dec_seq_len, -1)
        attn_out = torch.nn.functional.linear(attn_out, o_proj_w)

        attn_scale = decoder_weights[f"{prefix}.self_attn_layer_scale.scale"]
        attn_out = attn_out * attn_scale
        x = x + attn_out

        post_ln_w = decoder_weights[f"{prefix}.post_attention_layernorm.weight"]
        mlp_input = rms_norm(x, post_ln_w, dec_eps)

        gate_w = decoder_weights[f"{prefix}.mlp.gate_proj.weight"]
        up_w = decoder_weights[f"{prefix}.mlp.up_proj.weight"]
        down_w = decoder_weights[f"{prefix}.mlp.down_proj.weight"]

        gate = torch.nn.functional.linear(mlp_input, gate_w)
        up = torch.nn.functional.linear(mlp_input, up_w)
        mlp_out = torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, down_w)

        mlp_scale = decoder_weights[f"{prefix}.mlp_layer_scale.scale"]
        mlp_out = mlp_out * mlp_scale
        x = x + mlp_out

    # Final norm before output projection
    final_norm_w = decoder_weights["decoder.pre_transformer.norm.weight"]
    x = rms_norm(x, final_norm_w, dec_eps)

    # Output projection
    output_proj_w = decoder_weights["decoder.pre_transformer.output_proj.weight"]
    output_proj_b = decoder_weights["decoder.pre_transformer.output_proj.bias"]
    x = torch.nn.functional.linear(x, output_proj_w, output_proj_b)
    print(f"Pre-transformer output: {x.shape}")

    # 5.4 Transpose for conv
    x = x.transpose(1, 2)

    # 5.5 Upsample stages
    for stage in range(2):
        conv_w = decoder_weights[f"decoder.upsample.{stage}.0.conv.weight"]
        conv_b = decoder_weights[f"decoder.upsample.{stage}.0.conv.bias"]
        kernel_size = conv_w.shape[2]
        stride = kernel_size

        x = torch.nn.functional.conv_transpose1d(x, conv_w, conv_b, stride=stride)

        # ConvNeXtBlock
        dw_w = decoder_weights[f"decoder.upsample.{stage}.1.dwconv.conv.weight"]
        dw_b = decoder_weights[f"decoder.upsample.{stage}.1.dwconv.conv.bias"]
        dw_kernel = dw_w.shape[2]
        x_dw = torch.nn.functional.pad(x, (dw_kernel - 1, 0))
        x_dw = torch.nn.functional.conv1d(x_dw, dw_w, dw_b, groups=x.shape[1])

        x_dw = x_dw.transpose(1, 2)
        norm_w = decoder_weights[f"decoder.upsample.{stage}.1.norm.weight"]
        norm_b = decoder_weights[f"decoder.upsample.{stage}.1.norm.bias"]
        x_norm = torch.nn.functional.layer_norm(x_dw, (x_dw.shape[-1],), norm_w, norm_b, eps=1e-6)

        pw1_w = decoder_weights[f"decoder.upsample.{stage}.1.pwconv1.weight"]
        pw1_b = decoder_weights[f"decoder.upsample.{stage}.1.pwconv1.bias"]
        x_pw1 = torch.nn.functional.linear(x_norm, pw1_w, pw1_b)

        x_gelu = torch.nn.functional.gelu(x_pw1)

        pw2_w = decoder_weights[f"decoder.upsample.{stage}.1.pwconv2.weight"]
        pw2_b = decoder_weights[f"decoder.upsample.{stage}.1.pwconv2.bias"]
        x_pw2 = torch.nn.functional.linear(x_gelu, pw2_w, pw2_b)

        gamma = decoder_weights[f"decoder.upsample.{stage}.1.gamma"]
        x_scaled = (x_pw2 * gamma).transpose(1, 2)

        x = x + x_scaled

    print(f"After upsample: {x.shape}")

    # 5.6 Decoder blocks
    def snake_beta(x, alpha, beta, eps=1e-9):
        alpha = alpha.view(1, -1, 1)
        beta = beta.view(1, -1, 1)
        alpha = torch.exp(alpha)
        beta = torch.exp(beta)
        return x + (1.0 / (beta + eps)) * torch.pow(torch.sin(x * alpha), 2)

    # decoder.0
    conv0_w = decoder_weights["decoder.decoder.0.conv.weight"]
    conv0_b = decoder_weights["decoder.decoder.0.conv.bias"]
    kernel_size = conv0_w.shape[2]
    x = torch.nn.functional.pad(x, (kernel_size - 1, 0))
    x = torch.nn.functional.conv1d(x, conv0_w, conv0_b)

    upsample_rates = decoder_config["upsample_rates"]

    for block_idx in range(1, 5):
        prefix = f"decoder.decoder.{block_idx}.block"
        rate = upsample_rates[block_idx - 1]

        # SnakeBeta
        snake_alpha = decoder_weights[f"{prefix}.0.alpha"]
        snake_beta_param = decoder_weights[f"{prefix}.0.beta"]
        x = snake_beta(x, snake_alpha, snake_beta_param)

        # TransConv
        conv_w = decoder_weights[f"{prefix}.1.conv.weight"]
        conv_b = decoder_weights[f"{prefix}.1.conv.bias"]
        kernel_size = conv_w.shape[2]
        stride = rate

        x = torch.nn.functional.conv_transpose1d(x, conv_w, conv_b, stride=stride)
        # Trim from right side only for exact input * stride upsampling
        trim = kernel_size - stride
        if trim > 0:
            x = x[..., :-trim]

        # ResidualUnits
        for unit_idx, dilation in enumerate([1, 3, 9], start=2):
            unit_prefix = f"{prefix}.{unit_idx}"
            residual = x

            a1_alpha = decoder_weights[f"{unit_prefix}.act1.alpha"]
            a1_beta = decoder_weights[f"{unit_prefix}.act1.beta"]
            x = snake_beta(x, a1_alpha, a1_beta)

            c1_w = decoder_weights[f"{unit_prefix}.conv1.conv.weight"]
            c1_b = decoder_weights[f"{unit_prefix}.conv1.conv.bias"]
            k1 = c1_w.shape[2]
            pad1 = dilation * (k1 - 1)
            x_padded = torch.nn.functional.pad(x, (pad1, 0))
            x = torch.nn.functional.conv1d(x_padded, c1_w, c1_b, dilation=dilation)

            a2_alpha = decoder_weights[f"{unit_prefix}.act2.alpha"]
            a2_beta = decoder_weights[f"{unit_prefix}.act2.beta"]
            x = snake_beta(x, a2_alpha, a2_beta)

            c2_w = decoder_weights[f"{unit_prefix}.conv2.conv.weight"]
            c2_b = decoder_weights[f"{unit_prefix}.conv2.conv.bias"]
            x = torch.nn.functional.conv1d(x, c2_w, c2_b)

            x = x + residual

    # decoder.5 - final SnakeBeta
    snake5_alpha = decoder_weights["decoder.decoder.5.alpha"]
    snake5_beta = decoder_weights["decoder.decoder.5.beta"]
    x = snake_beta(x, snake5_alpha, snake5_beta)

    # decoder.6 - final conv
    conv6_w = decoder_weights["decoder.decoder.6.conv.weight"]
    conv6_b = decoder_weights["decoder.decoder.6.conv.bias"]
    kernel_size = conv6_w.shape[2]
    x = torch.nn.functional.pad(x, (kernel_size - 1, 0))
    x = torch.nn.functional.conv1d(x, conv6_w, conv6_b)

    # Clamp to [-1, 1]
    audio = x.clamp(-1, 1)

    print(f"\n=== Final Output ===")
    print(f"Audio shape: {audio.shape}")
    print(f"Audio samples: {audio.shape[2]}")
    print(f"Audio mean: {audio.mean().item():.6f}")
    print(f"Audio min: {audio.min().item():.6f}")
    print(f"Audio max: {audio.max().item():.6f}")

    # Save outputs
    save_tensor(output_dir / "e2e_audio.bin", audio)

    # Save intermediate values
    save_tensor(output_dir / "e2e_semantic_tokens.bin", semantic_tokens.float())
    save_tensor(output_dir / "e2e_codes.bin", codes.float())

    # Save metadata
    metadata = {
        "input_ids": input_ids.tolist(),
        "semantic_tokens": semantic_tokens.tolist(),
        "acoustic_tokens": acoustic_tokens,
        "codes": codes.squeeze().tolist(),
        "audio_shape": list(audio.shape),
        "audio_samples": audio.shape[2],
        "audio_mean": audio.mean().item(),
    }

    with open(output_dir / "e2e_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n=== Saved end-to-end reference values to {output_dir} ===")


def save_tensor(path, tensor):
    """Save tensor as raw float32 binary."""
    np.array(tensor.detach().cpu().numpy(), dtype=np.float32).tofile(path)
    print(f"  Saved {path.name}: {tensor.shape}")


if __name__ == "__main__":
    export_e2e_reference()
