#!/usr/bin/env python3
"""Debug decoder by comparing intermediate outputs with Rust."""

import struct
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from safetensors import safe_open


def load_weights(path: str) -> dict:
    """Load weights from safetensors."""
    weights = {}
    with safe_open(path, framework='pt') as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    return weights


def load_codes(path: str, num_frames: int) -> torch.Tensor:
    """Load binary codes."""
    data = Path(path).read_bytes()
    codes = struct.unpack(f'{len(data)//8}q', data)
    codes = np.array(codes, dtype=np.int64).reshape(num_frames, 16)
    # [frames, 16] -> [1, 16, frames]
    return torch.tensor(codes.T).unsqueeze(0).long()


def apply_rope(q, k, cos, sin):
    """Apply rotary position embedding."""
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


def rms_norm(x, weight, eps=1e-5):
    """RMS normalization."""
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return x * weight


def snake_beta(x, alpha, beta):
    """SnakeBeta activation."""
    return x + (1.0 / beta.unsqueeze(0).unsqueeze(2)) * torch.sin(alpha.unsqueeze(0).unsqueeze(2) * x).pow(2)


def main():
    # Load codes (using greedy 10-frame output that matches Python)
    codes = load_codes('../test_data/rust_audio_greedy_10/codes_seed42_frames10.bin', 10)
    print(f"Codes shape: {codes.shape}")  # [1, 16, 10]

    # Load weights
    weights = load_weights('../test_data/speech_tokenizer/model.safetensors')

    batch_size = 1
    seq_len = 10

    # Step 1: Quantizer decode
    print("\n=== Step 1: Quantizer Decode ===")
    first_cb = weights['decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum']
    print(f"First codebook shape: {first_cb.shape}")  # [2048, 256]

    # Lookup first quantizer (semantic)
    first_codes = codes[:, 0, :].flatten()  # [10]
    first_embed = first_cb[first_codes]  # [10, 256]
    quantized = first_embed.unsqueeze(0)  # [1, 10, 256]

    # Add rest quantizers (acoustic)
    for i in range(15):
        cb = weights[f'decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum']
        layer_codes = codes[:, i + 1, :].flatten()
        embed = cb[layer_codes].unsqueeze(0)
        quantized = quantized + embed

    print(f"Quantized shape: {quantized.shape}")
    print(f"Quantized stats: min={quantized.min():.4f}, max={quantized.max():.4f}, mean={quantized.mean():.4f}")

    # Output projection
    output_proj_w = weights['decoder.quantizer.rvq_first.output_proj.weight'].squeeze(2)
    print(f"Output proj weight shape: {output_proj_w.shape}")  # [512, 256]
    quantized = F.linear(quantized, output_proj_w)  # [1, 10, 512]
    print(f"After output proj: {quantized.shape}, mean={quantized.mean():.4f}")

    # Step 2: Pre-conv
    print("\n=== Step 2: Pre-conv ===")
    x = quantized.transpose(1, 2)  # [1, 512, 10]
    pre_conv_w = weights['decoder.pre_conv.conv.weight']
    pre_conv_b = weights['decoder.pre_conv.conv.bias']
    print(f"Pre-conv weight shape: {pre_conv_w.shape}")  # should be [1024, 512, kernel]

    kernel_size = pre_conv_w.shape[2]
    # Causal padding
    x_padded = F.pad(x, (kernel_size - 1, 0))
    x = F.conv1d(x_padded, pre_conv_w, pre_conv_b)
    print(f"After pre-conv: {x.shape}, mean={x.mean():.4f}")

    # Step 3: Pre-transformer
    print("\n=== Step 3: Pre-transformer ===")
    hidden = x.transpose(1, 2)  # [1, 10, 1024]

    input_proj_w = weights['decoder.pre_transformer.input_proj.weight']
    input_proj_b = weights['decoder.pre_transformer.input_proj.bias']
    print(f"Input proj weight shape: {input_proj_w.shape}")  # [512, 1024]
    hidden = F.linear(hidden, input_proj_w, input_proj_b)  # [1, 10, 512]
    print(f"After input proj: {hidden.shape}, mean={hidden.mean():.4f}")

    # Transformer layers
    num_layers = 8
    num_heads = 16
    head_dim = 64

    # Build RoPE
    positions = torch.arange(seq_len).float()
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = positions.unsqueeze(1) @ inv_freq.unsqueeze(0)
    cos = freqs.cos().repeat(1, 2).unsqueeze(0).unsqueeze(0)  # [1, 1, seq, head_dim]
    sin = freqs.sin().repeat(1, 2).unsqueeze(0).unsqueeze(0)

    # Causal mask
    causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)

    for layer_idx in range(num_layers):
        prefix = f'decoder.pre_transformer.layers.{layer_idx}'

        # Input layernorm
        ln_w = weights[f'{prefix}.input_layernorm.weight']
        normed = rms_norm(hidden, ln_w)

        # Self attention
        q_proj_w = weights[f'{prefix}.self_attn.q_proj.weight']
        k_proj_w = weights[f'{prefix}.self_attn.k_proj.weight']
        v_proj_w = weights[f'{prefix}.self_attn.v_proj.weight']
        o_proj_w = weights[f'{prefix}.self_attn.o_proj.weight']

        q = F.linear(normed, q_proj_w)
        k = F.linear(normed, k_proj_w)
        v = F.linear(normed, v_proj_w)

        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        # RoPE
        q, k = apply_rope(q, k, cos, sin)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn = attn + causal_mask
        attn = F.softmax(attn, dim=-1)
        attn_out = attn @ v

        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, num_heads * head_dim)
        attn_out = F.linear(attn_out, o_proj_w)

        # Layer scale
        attn_scale = weights[f'{prefix}.self_attn_layer_scale.scale']
        attn_out = attn_out * attn_scale
        hidden = hidden + attn_out

        # Post-attention layernorm + MLP
        ln2_w = weights[f'{prefix}.post_attention_layernorm.weight']
        mlp_input = rms_norm(hidden, ln2_w)

        gate_w = weights[f'{prefix}.mlp.gate_proj.weight']
        up_w = weights[f'{prefix}.mlp.up_proj.weight']
        down_w = weights[f'{prefix}.mlp.down_proj.weight']

        gate = F.linear(mlp_input, gate_w)
        up = F.linear(mlp_input, up_w)
        mlp_out = F.linear(F.silu(gate) * up, down_w)

        mlp_scale = weights[f'{prefix}.mlp_layer_scale.scale']
        mlp_out = mlp_out * mlp_scale
        hidden = hidden + mlp_out

        if layer_idx == 0:
            print(f"After layer 0: mean={hidden.mean():.4f}")

    # Final norm
    final_ln_w = weights['decoder.pre_transformer.norm.weight']
    hidden = rms_norm(hidden, final_ln_w)

    # Output projection
    output_proj_w = weights['decoder.pre_transformer.output_proj.weight']
    output_proj_b = weights['decoder.pre_transformer.output_proj.bias']
    hidden = F.linear(hidden, output_proj_w, output_proj_b)  # [1, 10, 1024]
    print(f"After transformer output proj: {hidden.shape}, mean={hidden.mean():.4f}")

    # Step 4: Upsampling
    print("\n=== Step 4: Upsampling ===")
    hidden = hidden.transpose(1, 2)  # [1, 1024, 10]
    print(f"Before upsampling: {hidden.shape}")

    # Upsample stages (upsampling_ratios = [2, 2])
    for stage_idx in range(2):
        prefix = f'decoder.upsample.{stage_idx}'
        stride = [2, 2][stage_idx]

        # Trans conv
        conv_w = weights[f'{prefix}.0.conv.weight']
        conv_b = weights[f'{prefix}.0.conv.bias']

        # Calculate output padding for trans conv
        hidden = F.conv_transpose1d(hidden, conv_w, conv_b, stride=stride)
        print(f"After upsample stage {stage_idx} trans_conv: {hidden.shape}")

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

        # Depthwise conv (causal)
        kernel_size = dwconv_w.shape[2]
        x_padded = F.pad(hidden, (kernel_size - 1, 0))
        x = F.conv1d(x_padded, dwconv_w, dwconv_b, groups=dwconv_w.shape[0])

        # Transpose for layer norm
        x = x.transpose(1, 2)  # [B, L, C]
        x = F.layer_norm(x, (x.shape[-1],), norm_w, norm_b)

        # Pointwise convs (as linear)
        x = F.linear(x, pwconv1_w, pwconv1_b)
        x = F.gelu(x)
        x = F.linear(x, pwconv2_w, pwconv2_b)

        # Scale and residual
        x = gamma * x
        x = x.transpose(1, 2)  # [B, C, L]
        hidden = residual + x
        print(f"After upsample stage {stage_idx} convnext: {hidden.shape}")

    # Step 5: Decoder blocks
    print("\n=== Step 5: Decoder Blocks ===")
    # decoder.0 is initial conv
    init_conv_w = weights['decoder.decoder.0.conv.weight']
    init_conv_b = weights['decoder.decoder.0.conv.bias']
    kernel_size = init_conv_w.shape[2]
    hidden = F.pad(hidden, (kernel_size - 1, 0))
    hidden = F.conv1d(hidden, init_conv_w, init_conv_b)
    print(f"After decoder init conv: {hidden.shape}")

    # Decoder blocks 1-4 (indices 1-4 in decoder.decoder)
    upsample_rates = [8, 5, 4, 3]
    for block_idx in range(4):
        prefix = f'decoder.decoder.{block_idx + 1}'
        stride = upsample_rates[block_idx]

        # SnakeBeta activation
        alpha = weights[f'{prefix}.block.0.alpha']
        beta = weights[f'{prefix}.block.0.beta']
        hidden = snake_beta(hidden, alpha, beta)

        # Trans conv
        conv_w = weights[f'{prefix}.block.1.conv.weight']
        conv_b = weights[f'{prefix}.block.1.conv.bias']
        hidden = F.conv_transpose1d(hidden, conv_w, conv_b, stride=stride)
        print(f"After decoder block {block_idx} trans_conv (stride={stride}): {hidden.shape}")

        # Residual units (2-4 are the residual blocks in each decoder block)
        for res_idx in range(2, 5):
            res_prefix = f'{prefix}.block.{res_idx}'

            # First snake activation
            act1_alpha = weights[f'{res_prefix}.act1.alpha']
            act1_beta = weights[f'{res_prefix}.act1.beta']
            x = snake_beta(hidden, act1_alpha, act1_beta)

            # First conv
            conv1_w = weights[f'{res_prefix}.conv1.conv.weight']
            conv1_b = weights[f'{res_prefix}.conv1.conv.bias']
            k = conv1_w.shape[2]
            x = F.pad(x, (k - 1, 0))
            x = F.conv1d(x, conv1_w, conv1_b)

            # Second snake activation
            act2_alpha = weights[f'{res_prefix}.act2.alpha']
            act2_beta = weights[f'{res_prefix}.act2.beta']
            x = snake_beta(x, act2_alpha, act2_beta)

            # Second conv
            conv2_w = weights[f'{res_prefix}.conv2.conv.weight']
            conv2_b = weights[f'{res_prefix}.conv2.conv.bias']
            k = conv2_w.shape[2]
            x = F.pad(x, (k - 1, 0))
            x = F.conv1d(x, conv2_w, conv2_b)

            # Residual
            hidden = hidden + x

    print(f"After all decoder blocks: {hidden.shape}")

    # Step 6: Final layers
    print("\n=== Step 6: Final Layers ===")
    # decoder.5 is final SnakeBeta
    final_alpha = weights['decoder.decoder.5.alpha']
    final_beta = weights['decoder.decoder.5.beta']
    hidden = snake_beta(hidden, final_alpha, final_beta)

    # decoder.6 is final conv
    final_conv_w = weights['decoder.decoder.6.conv.weight']
    final_conv_b = weights['decoder.decoder.6.conv.bias']
    k = final_conv_w.shape[2]
    hidden = F.pad(hidden, (k - 1, 0))
    hidden = F.conv1d(hidden, final_conv_w, final_conv_b)

    print(f"Final output shape: {hidden.shape}")
    print(f"Final output stats: min={hidden.min():.4f}, max={hidden.max():.4f}, mean={hidden.mean():.4f}")

    # Compare with Rust output
    rust_audio = np.fromfile('../test_data/rust_audio_greedy_10/audio_seed42_frames10.bin', dtype=np.float32)
    python_audio = hidden.squeeze().detach().numpy()

    print(f"\n=== Comparison ===")
    print(f"Rust audio: {len(rust_audio)} samples, range [{rust_audio.min():.4f}, {rust_audio.max():.4f}]")
    print(f"Python audio: {len(python_audio)} samples, range [{python_audio.min():.4f}, {python_audio.max():.4f}]")

    min_len = min(len(rust_audio), len(python_audio))
    diff = np.abs(rust_audio[:min_len] - python_audio[:min_len])
    print(f"Max difference: {diff.max():.6f}")
    print(f"Mean difference: {diff.mean():.6f}")

    # Save Python output for comparison
    python_audio.astype(np.float32).tofile('../test_data/rust_audio_greedy_10/python_audio.bin')
    print(f"\nSaved Python audio to python_audio.bin")


if __name__ == '__main__':
    main()
