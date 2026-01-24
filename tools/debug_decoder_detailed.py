#!/usr/bin/env python3
"""Detailed decoder debugging - compare Python vs Rust at each stage."""

import struct
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import load_file


def snake_beta(x, alpha, beta):
    """SnakeBeta activation: x + (1/beta) * sin(alpha * x)^2"""
    return x + (1.0 / beta.unsqueeze(0).unsqueeze(2)) * torch.sin(alpha.unsqueeze(0).unsqueeze(2) * x).pow(2)


def rms_norm(x, weight, eps=1e-5):
    """RMS normalization."""
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return x * weight


def apply_rope(q, k, cos, sin):
    """Apply rotary position embedding."""
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


def full_python_decode(codes, weights):
    """Full Python decoder implementation with normalized codebooks."""
    batch_size = codes.shape[0]
    seq_len = codes.shape[2]
    codebook_dim = 256
    eps = 1e-5

    # 1. Quantizer decode with normalization
    first_embedding_sum = weights["decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"]
    first_cluster_usage = weights["decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage"]
    first_codebook = first_embedding_sum / first_cluster_usage.unsqueeze(-1)

    embeddings = []
    first_codes = codes[:, 0, :].flatten() % first_codebook.shape[0]
    first_embed = first_codebook[first_codes].view(batch_size, seq_len, codebook_dim)
    embeddings.append(first_embed)

    for i in range(15):
        embedding_sum = weights[f"decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum"]
        cluster_usage = weights[f"decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.cluster_usage"]
        cb = embedding_sum / cluster_usage.unsqueeze(-1)
        c = codes[:, i + 1, :].flatten()
        embed = cb[c].view(batch_size, seq_len, codebook_dim)
        embeddings.append(embed)

    quantized = sum(embeddings)
    print(f"Quantized: shape={quantized.shape}, mean={quantized.mean():.4f}, range=[{quantized.min():.4f}, {quantized.max():.4f}]")

    # Output projection
    output_proj_w = weights["decoder.quantizer.rvq_first.output_proj.weight"].squeeze(2)
    quantized = F.linear(quantized, output_proj_w)
    print(f"After output proj: shape={quantized.shape}, mean={quantized.mean():.4f}")

    # 2. Pre-conv
    x = quantized.transpose(1, 2)  # [batch, channels, seq]
    pre_conv_w = weights["decoder.pre_conv.conv.weight"]
    pre_conv_b = weights["decoder.pre_conv.conv.bias"]
    kernel_size = pre_conv_w.shape[2]
    x = F.pad(x, (kernel_size - 1, 0))
    x = F.conv1d(x, pre_conv_w, pre_conv_b)
    print(f"After pre_conv: shape={x.shape}, mean={x.mean():.4f}")

    # 3. Pre-transformer
    input_proj_w = weights["decoder.pre_transformer.input_proj.weight"]
    input_proj_b = weights["decoder.pre_transformer.input_proj.bias"]
    hidden = x.transpose(1, 2)  # [batch, seq, channels]
    hidden = F.linear(hidden, input_proj_w, input_proj_b)
    print(f"After input proj: shape={hidden.shape}, mean={hidden.mean():.4f}")

    # Transformer layers
    num_layers = 8
    num_heads = 16
    head_dim = 64
    dec_seq_len = hidden.shape[1]

    # Build RoPE
    positions = torch.arange(dec_seq_len).float()
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = positions.unsqueeze(1) @ inv_freq.unsqueeze(0)
    cos = freqs.cos().repeat(1, 2).unsqueeze(0).unsqueeze(0)
    sin = freqs.sin().repeat(1, 2).unsqueeze(0).unsqueeze(0)

    causal_mask = torch.triu(torch.full((dec_seq_len, dec_seq_len), float('-inf')), diagonal=1)

    for layer_idx in range(num_layers):
        prefix = f'decoder.pre_transformer.layers.{layer_idx}'

        # Input layernorm
        ln_w = weights[f'{prefix}.input_layernorm.weight']
        normed = rms_norm(hidden, ln_w)

        # Self attention
        q = F.linear(normed, weights[f'{prefix}.self_attn.q_proj.weight'])
        k = F.linear(normed, weights[f'{prefix}.self_attn.k_proj.weight'])
        v = F.linear(normed, weights[f'{prefix}.self_attn.v_proj.weight'])

        q = q.view(batch_size, dec_seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, dec_seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, dec_seq_len, num_heads, head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn = attn + causal_mask
        attn = F.softmax(attn, dim=-1)
        attn_out = attn @ v

        attn_out = attn_out.transpose(1, 2).reshape(batch_size, dec_seq_len, num_heads * head_dim)
        attn_out = F.linear(attn_out, weights[f'{prefix}.self_attn.o_proj.weight'])

        attn_scale = weights[f'{prefix}.self_attn_layer_scale.scale']
        attn_out = attn_out * attn_scale
        hidden = hidden + attn_out

        # MLP
        ln2_w = weights[f'{prefix}.post_attention_layernorm.weight']
        mlp_input = rms_norm(hidden, ln2_w)

        gate = F.linear(mlp_input, weights[f'{prefix}.mlp.gate_proj.weight'])
        up = F.linear(mlp_input, weights[f'{prefix}.mlp.up_proj.weight'])
        mlp_out = F.linear(F.silu(gate) * up, weights[f'{prefix}.mlp.down_proj.weight'])

        mlp_scale = weights[f'{prefix}.mlp_layer_scale.scale']
        mlp_out = mlp_out * mlp_scale
        hidden = hidden + mlp_out

    # Final norm
    final_ln_w = weights['decoder.pre_transformer.norm.weight']
    hidden = rms_norm(hidden, final_ln_w)

    # Output projection
    output_proj_w = weights['decoder.pre_transformer.output_proj.weight']
    output_proj_b = weights['decoder.pre_transformer.output_proj.bias']
    hidden = F.linear(hidden, output_proj_w, output_proj_b)
    print(f"After pre_transformer: shape={hidden.shape}, mean={hidden.mean():.4f}")

    # 4. Upsampling
    hidden = hidden.transpose(1, 2)  # [batch, channels, seq]

    for stage_idx in range(2):
        prefix = f'decoder.upsample.{stage_idx}'
        stride = [2, 2][stage_idx]

        # Trans conv
        conv_w = weights[f'{prefix}.0.conv.weight']
        conv_b = weights[f'{prefix}.0.conv.bias']
        hidden = F.conv_transpose1d(hidden, conv_w, conv_b, stride=stride)

        # ConvNeXt block
        dwconv_w = weights[f'{prefix}.1.dwconv.conv.weight']
        dwconv_b = weights[f'{prefix}.1.dwconv.conv.bias']
        norm_w = weights[f'{prefix}.1.norm.weight']
        norm_b = weights[f'{prefix}.1.norm.bias']
        pwconv1_w = weights[f'{prefix}.1.pwconv1.weight']
        pwconv1_b = weights[f'{prefix}.1.pwconv1.bias']
        pwconv2_w = weights[f'{prefix}.1.pwconv2.weight']
        pwconv2_b = weights[f'{prefix}.1.pwconv2.bias']
        gamma = weights[f'{prefix}.1.gamma']

        residual = hidden
        kernel_size = dwconv_w.shape[2]
        x_padded = F.pad(hidden, (kernel_size - 1, 0))
        x = F.conv1d(x_padded, dwconv_w, dwconv_b, groups=dwconv_w.shape[0])
        x = x.transpose(1, 2)
        x = F.layer_norm(x, (x.shape[-1],), norm_w, norm_b)
        x = F.linear(x, pwconv1_w, pwconv1_b)
        x = F.gelu(x)
        x = F.linear(x, pwconv2_w, pwconv2_b)
        x = gamma * x
        x = x.transpose(1, 2)
        hidden = residual + x

    print(f"After upsample: shape={hidden.shape}, mean={hidden.mean():.4f}")

    # 5. Decoder blocks
    init_conv_w = weights['decoder.decoder.0.conv.weight']
    init_conv_b = weights['decoder.decoder.0.conv.bias']
    kernel_size = init_conv_w.shape[2]
    hidden = F.pad(hidden, (kernel_size - 1, 0))
    hidden = F.conv1d(hidden, init_conv_w, init_conv_b)
    print(f"After decoder.0: shape={hidden.shape}, mean={hidden.mean():.4f}")

    upsample_rates = [8, 5, 4, 3]
    for block_idx in range(4):
        prefix = f'decoder.decoder.{block_idx + 1}'
        stride = upsample_rates[block_idx]

        # SnakeBeta
        alpha = weights[f'{prefix}.block.0.alpha']
        beta = weights[f'{prefix}.block.0.beta']
        hidden = snake_beta(hidden, alpha, beta)

        # Trans conv
        conv_w = weights[f'{prefix}.block.1.conv.weight']
        conv_b = weights[f'{prefix}.block.1.conv.bias']
        hidden = F.conv_transpose1d(hidden, conv_w, conv_b, stride=stride)

        # Residual units
        for res_idx in range(2, 5):
            res_prefix = f'{prefix}.block.{res_idx}'

            act1_alpha = weights[f'{res_prefix}.act1.alpha']
            act1_beta = weights[f'{res_prefix}.act1.beta']
            x = snake_beta(hidden, act1_alpha, act1_beta)

            conv1_w = weights[f'{res_prefix}.conv1.conv.weight']
            conv1_b = weights[f'{res_prefix}.conv1.conv.bias']
            k = conv1_w.shape[2]
            x = F.pad(x, (k - 1, 0))
            x = F.conv1d(x, conv1_w, conv1_b)

            act2_alpha = weights[f'{res_prefix}.act2.alpha']
            act2_beta = weights[f'{res_prefix}.act2.beta']
            x = snake_beta(x, act2_alpha, act2_beta)

            conv2_w = weights[f'{res_prefix}.conv2.conv.weight']
            conv2_b = weights[f'{res_prefix}.conv2.conv.bias']
            k = conv2_w.shape[2]
            x = F.pad(x, (k - 1, 0))
            x = F.conv1d(x, conv2_w, conv2_b)

            hidden = hidden + x

        print(f"After decoder.{block_idx + 1}: shape={hidden.shape}, mean={hidden.mean():.4f}")

    # 6. Final layers
    final_alpha = weights['decoder.decoder.5.alpha']
    final_beta = weights['decoder.decoder.5.beta']
    hidden = snake_beta(hidden, final_alpha, final_beta)

    final_conv_w = weights['decoder.decoder.6.conv.weight']
    final_conv_b = weights['decoder.decoder.6.conv.bias']
    k = final_conv_w.shape[2]
    hidden = F.pad(hidden, (k - 1, 0))
    hidden = F.conv1d(hidden, final_conv_w, final_conv_b)

    print(f"Final output: shape={hidden.shape}, range=[{hidden.min():.4f}, {hidden.max():.4f}]")

    # Apply tanh to bound output
    hidden = torch.tanh(hidden)
    print(f"After tanh: range=[{hidden.min():.4f}, {hidden.max():.4f}]")

    return hidden


def main():
    # Use simple test codes (zeros)
    batch_size = 1
    num_quantizers = 16
    seq_len = 10  # 10 frames = ~0.8s of audio

    codes = torch.zeros((batch_size, num_quantizers, seq_len), dtype=torch.long)
    print(f"Test codes shape: {codes.shape}")

    # Load weights
    weights = load_file('test_data/speech_tokenizer/model.safetensors')
    weights = {k: v.float() for k, v in weights.items()}

    print("\n=== Python Decoder ===")
    python_audio = full_python_decode(codes, weights)

    # Save Python output
    output_dir = Path('test_data/debug_decoder')
    output_dir.mkdir(exist_ok=True)
    python_audio.squeeze().detach().numpy().astype(np.float32).tofile(output_dir / 'python_audio.bin')

    # Also save as WAV
    import scipy.io.wavfile as wav
    audio_np = python_audio.squeeze().detach().numpy()
    wav.write(output_dir / 'python_audio.wav', 24000, audio_np.astype(np.float32))

    print(f"\nSaved to {output_dir}")
    print(f"Python audio: {len(audio_np)} samples, range [{audio_np.min():.4f}, {audio_np.max():.4f}]")


if __name__ == '__main__':
    main()
