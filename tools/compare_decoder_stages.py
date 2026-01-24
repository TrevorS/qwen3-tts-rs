#!/usr/bin/env python3
"""Compare Python and Rust decoder at each stage."""

import struct
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import load_file


def snake_beta(x, alpha, beta):
    """SnakeBeta activation: x + (1/exp(beta)) * sin(exp(alpha) * x)^2."""
    # Alpha and beta are exponentiated per official implementation
    alpha_exp = alpha.exp().unsqueeze(0).unsqueeze(2)
    beta_exp = beta.exp().unsqueeze(0).unsqueeze(2)
    return x + (1.0 / (beta_exp + 1e-9)) * torch.sin(alpha_exp * x).pow(2)


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


def main():
    # Load Rust codes
    rust_dir = Path('test_data/rust_audio_fixed2')
    rust_codes_file = rust_dir / 'codes_seed42_frames75.bin'

    if not rust_codes_file.exists():
        print("Rust codes not found, using zeros")
        codes = torch.zeros((1, 16, 10), dtype=torch.long)
    else:
        data = rust_codes_file.read_bytes()
        codes_flat = struct.unpack(f'{len(data)//8}q', data)
        num_frames = len(codes_flat) // 16
        codes = torch.tensor(codes_flat, dtype=torch.long).view(num_frames, 16).T.unsqueeze(0)
        print(f"Loaded codes: {codes.shape}")

    # Load weights
    weights = load_file('test_data/speech_tokenizer/model.safetensors')
    weights = {k: v.float() for k, v in weights.items()}

    batch_size = codes.shape[0]
    seq_len = codes.shape[2]
    codebook_dim = 256
    epsilon = 1e-7

    # 1. Quantizer decode
    print("\n=== 1. Quantizer Decode ===")
    first_embedding_sum = weights["decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"]
    first_cluster_usage = weights["decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage"]
    first_codebook = first_embedding_sum / first_cluster_usage.clamp(min=epsilon).unsqueeze(-1)

    embeddings = []
    first_codes = codes[:, 0, :].flatten() % first_codebook.shape[0]
    first_embed = first_codebook[first_codes].view(batch_size, seq_len, codebook_dim)
    embeddings.append(first_embed)

    for i in range(15):
        embedding_sum = weights[f"decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum"]
        cluster_usage = weights[f"decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.cluster_usage"]
        cb = embedding_sum / cluster_usage.clamp(min=epsilon).unsqueeze(-1)
        c = codes[:, i + 1, :].flatten()
        embed = cb[c].view(batch_size, seq_len, codebook_dim)
        embeddings.append(embed)

    quantized = sum(embeddings)
    output_proj_w = weights["decoder.quantizer.rvq_first.output_proj.weight"].squeeze(2)
    quantized = F.linear(quantized, output_proj_w)
    print(f"Quantized: shape={quantized.shape}, mean={quantized.mean():.6f}, std={quantized.std():.6f}")

    # 2. Pre-conv
    print("\n=== 2. Pre-conv ===")
    x = quantized.transpose(1, 2)
    pre_conv_w = weights["decoder.pre_conv.conv.weight"]
    pre_conv_b = weights["decoder.pre_conv.conv.bias"]
    kernel_size = pre_conv_w.shape[2]
    x = F.pad(x, (kernel_size - 1, 0))
    x = F.conv1d(x, pre_conv_w, pre_conv_b)
    print(f"Pre-conv: shape={x.shape}, mean={x.mean():.6f}, std={x.std():.6f}")

    # 3. Pre-transformer
    print("\n=== 3. Pre-transformer ===")
    input_proj_w = weights["decoder.pre_transformer.input_proj.weight"]
    input_proj_b = weights["decoder.pre_transformer.input_proj.bias"]
    hidden = x.transpose(1, 2)
    hidden = F.linear(hidden, input_proj_w, input_proj_b)

    num_layers = 8
    num_heads = 16
    head_dim = 64
    dec_seq_len = hidden.shape[1]

    positions = torch.arange(dec_seq_len).float()
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = positions.unsqueeze(1) @ inv_freq.unsqueeze(0)
    cos = freqs.cos().repeat(1, 2).unsqueeze(0).unsqueeze(0)
    sin = freqs.sin().repeat(1, 2).unsqueeze(0).unsqueeze(0)
    causal_mask = torch.triu(torch.full((dec_seq_len, dec_seq_len), float('-inf')), diagonal=1)

    for layer_idx in range(num_layers):
        prefix = f'decoder.pre_transformer.layers.{layer_idx}'
        ln_w = weights[f'{prefix}.input_layernorm.weight']
        normed = rms_norm(hidden, ln_w)

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

        ln2_w = weights[f'{prefix}.post_attention_layernorm.weight']
        mlp_input = rms_norm(hidden, ln2_w)

        gate = F.linear(mlp_input, weights[f'{prefix}.mlp.gate_proj.weight'])
        up = F.linear(mlp_input, weights[f'{prefix}.mlp.up_proj.weight'])
        mlp_out = F.linear(F.silu(gate) * up, weights[f'{prefix}.mlp.down_proj.weight'])

        mlp_scale = weights[f'{prefix}.mlp_layer_scale.scale']
        mlp_out = mlp_out * mlp_scale
        hidden = hidden + mlp_out

    final_ln_w = weights['decoder.pre_transformer.norm.weight']
    hidden = rms_norm(hidden, final_ln_w)
    output_proj_w = weights['decoder.pre_transformer.output_proj.weight']
    output_proj_b = weights['decoder.pre_transformer.output_proj.bias']
    hidden = F.linear(hidden, output_proj_w, output_proj_b)
    print(f"Pre-transformer: shape={hidden.shape}, mean={hidden.mean():.6f}, std={hidden.std():.6f}")

    # 4. Upsampling
    print("\n=== 4. Upsampling ===")
    hidden = hidden.transpose(1, 2)

    for stage_idx in range(2):
        prefix = f'decoder.upsample.{stage_idx}'
        stride = [2, 2][stage_idx]

        conv_w = weights[f'{prefix}.0.conv.weight']
        conv_b = weights[f'{prefix}.0.conv.bias']

        # Trans conv
        hidden = F.conv_transpose1d(hidden, conv_w, conv_b, stride=stride)
        print(f"  Stage {stage_idx} after trans_conv: {hidden.shape}, mean={hidden.mean():.6f}")

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
        k = dwconv_w.shape[2]
        x_padded = F.pad(hidden, (k - 1, 0))
        x = F.conv1d(x_padded, dwconv_w, dwconv_b, groups=dwconv_w.shape[0])
        x = x.transpose(1, 2)
        x = F.layer_norm(x, (x.shape[-1],), norm_w, norm_b)
        x = F.linear(x, pwconv1_w, pwconv1_b)
        x = F.gelu(x)
        x = F.linear(x, pwconv2_w, pwconv2_b)
        x = gamma * x
        x = x.transpose(1, 2)
        hidden = residual + x
        print(f"  Stage {stage_idx} after convnext: {hidden.shape}, mean={hidden.mean():.6f}")

    # 5. Decoder blocks
    print("\n=== 5. Decoder Blocks ===")
    init_conv_w = weights['decoder.decoder.0.conv.weight']
    init_conv_b = weights['decoder.decoder.0.conv.bias']
    k = init_conv_w.shape[2]
    hidden = F.pad(hidden, (k - 1, 0))
    hidden = F.conv1d(hidden, init_conv_w, init_conv_b)
    print(f"After decoder.0: {hidden.shape}, mean={hidden.mean():.6f}, std={hidden.std():.6f}")

    upsample_rates = [8, 5, 4, 3]
    for block_idx in range(4):
        prefix = f'decoder.decoder.{block_idx + 1}'
        stride = upsample_rates[block_idx]

        alpha = weights[f'{prefix}.block.0.alpha']
        beta = weights[f'{prefix}.block.0.beta']
        hidden = snake_beta(hidden, alpha, beta)
        snake_mean = hidden.mean().item()

        conv_w = weights[f'{prefix}.block.1.conv.weight']
        conv_b = weights[f'{prefix}.block.1.conv.bias']
        kernel_size = conv_w.shape[2]
        hidden = F.conv_transpose1d(hidden, conv_w, conv_b, stride=stride)
        # Trim (kernel_size - stride) from right for exact upsampling (output = input * stride)
        # Raw output = (input-1)*stride + kernel, trim = kernel - stride
        trim = kernel_size - stride
        if trim > 0:
            hidden = hidden[..., :-trim]
        trans_mean = hidden.mean().item()

        # Dilations for the 3 residual units (matching official implementation)
        dilations = [1, 3, 9]
        for res_idx, dilation in zip(range(2, 5), dilations):
            res_prefix = f'{prefix}.block.{res_idx}'

            act1_alpha = weights[f'{res_prefix}.act1.alpha']
            act1_beta = weights[f'{res_prefix}.act1.beta']
            x = snake_beta(hidden, act1_alpha, act1_beta)

            conv1_w = weights[f'{res_prefix}.conv1.conv.weight']
            conv1_b = weights[f'{res_prefix}.conv1.conv.bias']
            k = conv1_w.shape[2]
            # Causal padding with dilation: padding = (kernel_size - 1) * dilation
            padding = (k - 1) * dilation
            x = F.pad(x, (padding, 0))
            x = F.conv1d(x, conv1_w, conv1_b, dilation=dilation)

            act2_alpha = weights[f'{res_prefix}.act2.alpha']
            act2_beta = weights[f'{res_prefix}.act2.beta']
            x = snake_beta(x, act2_alpha, act2_beta)

            conv2_w = weights[f'{res_prefix}.conv2.conv.weight']
            conv2_b = weights[f'{res_prefix}.conv2.conv.bias']
            k = conv2_w.shape[2]
            x = F.pad(x, (k - 1, 0))
            x = F.conv1d(x, conv2_w, conv2_b)

            hidden = hidden + x

        print(f"After decoder.{block_idx + 1}: {hidden.shape}, mean={hidden.mean():.6f}, std={hidden.std():.6f}")
        print(f"  (snake={snake_mean:.6f}, trans={trans_mean:.6f})")

    # 6. Final
    print("\n=== 6. Final ===")
    final_alpha = weights['decoder.decoder.5.alpha']
    final_beta = weights['decoder.decoder.5.beta']
    hidden = snake_beta(hidden, final_alpha, final_beta)

    final_conv_w = weights['decoder.decoder.6.conv.weight']
    final_conv_b = weights['decoder.decoder.6.conv.bias']
    k = final_conv_w.shape[2]
    hidden = F.pad(hidden, (k - 1, 0))
    hidden = F.conv1d(hidden, final_conv_w, final_conv_b)

    print(f"Before clamp: {hidden.shape}, range=[{hidden.min():.4f}, {hidden.max():.4f}], mean={hidden.mean():.6f}")

    # Apply clamp
    hidden = hidden.clamp(-1, 1)
    print(f"After clamp: range=[{hidden.min():.4f}, {hidden.max():.4f}], mean={hidden.mean():.6f}")

    # Compare with Rust
    rust_audio_file = rust_dir / 'audio_seed42_frames75.bin'
    if rust_audio_file.exists():
        rust_audio = np.fromfile(rust_audio_file, dtype=np.float32)
        python_audio = hidden.squeeze().detach().numpy()

        print(f"\n=== Comparison ===")
        print(f"Python: {len(python_audio)} samples, range=[{python_audio.min():.4f}, {python_audio.max():.4f}]")
        print(f"Rust: {len(rust_audio)} samples, range=[{rust_audio.min():.4f}, {rust_audio.max():.4f}]")

        min_len = min(len(python_audio), len(rust_audio))
        diff = np.abs(python_audio[:min_len] - rust_audio[:min_len])
        print(f"Max diff: {diff.max():.6f}, Mean diff: {diff.mean():.6f}")


if __name__ == '__main__':
    main()
