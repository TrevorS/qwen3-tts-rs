#!/usr/bin/env python3
"""Export reference values for the speech tokenizer decoder."""

import json
import numpy as np
import torch
from pathlib import Path
from safetensors.torch import load_file


def export_decoder_reference():
    """Export reference values for speech tokenizer decoder validation."""

    device = torch.device("cpu")
    output_dir = Path("test_data/reference_values")
    output_dir.mkdir(exist_ok=True)

    # Load speech tokenizer weights
    print("Loading speech tokenizer weights...")
    weights = load_file("test_data/speech_tokenizer/model.safetensors")
    weights = {k: v.float() for k, v in weights.items()}

    # Load config
    with open("test_data/speech_tokenizer/config.json") as f:
        config = json.load(f)

    dec_config = config["decoder_config"]
    print(f"Decoder config: hidden={dec_config['hidden_size']}, layers={dec_config['num_hidden_layers']}")

    # Create test codes: [batch, num_quantizers, seq_len]
    # Use all zeros for reproducible test
    batch_size = 1
    num_quantizers = 16
    seq_len = 2  # Short sequence for testing
    codes = torch.zeros((batch_size, num_quantizers, seq_len), dtype=torch.long)

    print(f"Test codes shape: {codes.shape}")

    # ===== 1. Quantizer decode =====
    print("\n=== Quantizer Decode ===")

    # Get codebook embeddings
    # Split RVQ: rvq_first (1 codebook) + rvq_rest (15 codebooks)
    # IMPORTANT: embeddings = embedding_sum / cluster_usage.clamp(min=epsilon) (per official implementation)
    codebook_dim = dec_config["codebook_dim"]  # 512
    epsilon = 1e-7

    # First quantizer (semantic)
    first_embedding_sum = weights["decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"]
    first_cluster_usage = weights["decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage"]
    first_codebook = first_embedding_sum / first_cluster_usage.clamp(min=epsilon).unsqueeze(-1)
    print(f"First codebook shape: {first_codebook.shape}")

    # Rest quantizers (acoustic)
    rest_codebooks = []
    for i in range(15):
        embedding_sum = weights[f"decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum"]
        cluster_usage = weights[f"decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.cluster_usage"]
        cb = embedding_sum / cluster_usage.clamp(min=epsilon).unsqueeze(-1)
        rest_codebooks.append(cb)
        if i == 0:
            print(f"Rest codebook shape: {cb.shape}")

    # Lookup embeddings
    embeddings = []

    # First (semantic) quantizer
    first_embed = first_codebook[codes[:, 0, :]]  # [batch, seq, dim]
    embeddings.append(first_embed)

    # Rest (acoustic) quantizers
    for i in range(15):
        embed = rest_codebooks[i][codes[:, i+1, :]]
        embeddings.append(embed)

    # Sum all embeddings (RVQ decoding)
    quantized = torch.stack(embeddings, dim=0).sum(dim=0)  # [batch, seq, 256]
    print(f"Quantized sum shape: {quantized.shape}")

    # Apply output projection to get to 512
    # There's a split: first quantizer and rest quantizers have separate projections
    # But they're combined, so we use the first output proj
    # Actually looking at the model, the quantizer.decode returns combined output
    # Let's use the output_proj from rvq_first (they're the same shape anyway)
    output_proj_w = weights["decoder.quantizer.rvq_first.output_proj.weight"]  # [512, 256, 1]
    print(f"Output proj weight: {output_proj_w.shape}")

    # This is a 1x1 conv, equivalent to linear with a squeeze
    output_proj_w_2d = output_proj_w.squeeze(-1)  # [512, 256]
    quantized = torch.nn.functional.linear(quantized, output_proj_w_2d)  # [batch, seq, 512]
    print(f"Quantized after output proj: {quantized.shape}")
    save_tensor(output_dir / "decoder_quantized.bin", quantized)

    # ===== 2. Pre-conv (Causal Conv1d) =====
    print("\n=== Causal Conv1d (Pre-conv) ===")
    # pre_conv: [1024, 512, 3] conv1d with causal (left-only) padding
    pre_conv_w = weights["decoder.pre_conv.conv.weight"]
    pre_conv_b = weights["decoder.pre_conv.conv.bias"]
    kernel_size = pre_conv_w.shape[2]  # 3
    dilation = 1  # pre_conv uses dilation=1
    print(f"Pre-conv weight: {pre_conv_w.shape}")
    print(f"Kernel size: {kernel_size}, dilation: {dilation}")

    # Need to transpose for conv1d: [batch, seq, channels] -> [batch, channels, seq]
    x = quantized.transpose(1, 2)  # [1, 512, 2]

    # Save pre-conv input for validation
    save_tensor(output_dir / "causal_conv_input.bin", x)

    # Causal padding (left only): padding = dilation * (kernel_size - 1)
    padding = dilation * (kernel_size - 1)
    print(f"Causal padding (left): {padding}")
    x_padded = torch.nn.functional.pad(x, (padding, 0))
    pre_conv_out = torch.nn.functional.conv1d(x_padded, pre_conv_w, pre_conv_b)
    print(f"Pre-conv output: {pre_conv_out.shape}")

    # Save output (in [batch, channels, seq] format for conv validation)
    save_tensor(output_dir / "causal_conv_output.bin", pre_conv_out)
    # Also save transposed for pre-transformer input
    save_tensor(output_dir / "decoder_pre_conv.bin", pre_conv_out.transpose(1, 2))

    # ===== 3. Pre-transformer =====
    print("\n=== Pre-transformer ===")

    # Input projection: 1024 -> 512
    input_proj_w = weights["decoder.pre_transformer.input_proj.weight"]
    input_proj_b = weights["decoder.pre_transformer.input_proj.bias"]

    x = pre_conv_out.transpose(1, 2)  # [1, 2, 1024]
    x = torch.nn.functional.linear(x, input_proj_w, input_proj_b)  # [1, 2, 512]
    print(f"After input proj: {x.shape}")

    # Run through 8 transformer layers
    num_layers = dec_config["num_hidden_layers"]  # 8
    hidden_size = dec_config["hidden_size"]  # 512
    num_heads = dec_config["num_attention_heads"]  # 16
    head_dim = dec_config["head_dim"]  # 64
    eps = dec_config["rms_norm_eps"]  # 1e-5
    rope_theta = dec_config["rope_theta"]  # 10000
    layer_scale = dec_config["layer_scale_initial_scale"]  # 0.01

    def rms_norm(x, weight, eps):
        variance = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + eps)
        return weight * x_norm

    # Build RoPE
    positions = torch.arange(seq_len).unsqueeze(0)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = torch.outer(positions.float().squeeze(0), inv_freq)
    cos = torch.cos(freqs).repeat(1, 2).unsqueeze(0).unsqueeze(0)  # [1, 1, seq, head_dim]
    sin = torch.sin(freqs).repeat(1, 2).unsqueeze(0).unsqueeze(0)

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    # Causal mask
    causal = torch.full((seq_len, seq_len), float("-inf"))
    causal = torch.triu(causal, diagonal=1).unsqueeze(0).unsqueeze(0)

    hidden = x
    for layer_idx in range(num_layers):
        prefix = f"decoder.pre_transformer.layers.{layer_idx}"

        # Input LayerNorm
        ln_w = weights[f"{prefix}.input_layernorm.weight"]
        normed = rms_norm(hidden, ln_w, eps)

        # Self attention
        q_proj_w = weights[f"{prefix}.self_attn.q_proj.weight"]
        k_proj_w = weights[f"{prefix}.self_attn.k_proj.weight"]
        v_proj_w = weights[f"{prefix}.self_attn.v_proj.weight"]
        o_proj_w = weights[f"{prefix}.self_attn.o_proj.weight"]

        q = torch.nn.functional.linear(normed, q_proj_w)
        k = torch.nn.functional.linear(normed, k_proj_w)
        v = torch.nn.functional.linear(normed, v_proj_w)

        # Reshape for attention
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        # RoPE
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)

        # Attention
        scaling = head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(2, 3)) * scaling
        attn = attn + causal
        attn = torch.nn.functional.softmax(attn, dim=-1)
        attn_out = torch.matmul(attn, v)

        # O projection
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        attn_out = torch.nn.functional.linear(attn_out, o_proj_w)

        # Layer scale for attention
        attn_scale = weights[f"{prefix}.self_attn_layer_scale.scale"]
        attn_out = attn_out * attn_scale

        # Residual
        hidden = hidden + attn_out

        # MLP
        post_ln_w = weights[f"{prefix}.post_attention_layernorm.weight"]
        mlp_input = rms_norm(hidden, post_ln_w, eps)

        gate_w = weights[f"{prefix}.mlp.gate_proj.weight"]
        up_w = weights[f"{prefix}.mlp.up_proj.weight"]
        down_w = weights[f"{prefix}.mlp.down_proj.weight"]

        gate = torch.nn.functional.linear(mlp_input, gate_w)
        up = torch.nn.functional.linear(mlp_input, up_w)
        mlp_out = torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, down_w)

        # Layer scale for MLP
        mlp_scale = weights[f"{prefix}.mlp_layer_scale.scale"]
        mlp_out = mlp_out * mlp_scale

        # Residual
        hidden = hidden + mlp_out

        if layer_idx % 2 == 0 or layer_idx == num_layers - 1:
            print(f"  Layer {layer_idx}: mean={hidden.mean().item():.6f}")

    print(f"Pre-transformer output (before final norm): {hidden.shape}")
    save_tensor(output_dir / "decoder_pre_transformer.bin", hidden)

    # ===== 4. Final norm and output projection =====
    print("\n=== Final Norm and Output Projection ===")

    # Apply final RMS norm before output projection
    final_norm_w = weights["decoder.pre_transformer.norm.weight"]
    hidden = rms_norm(hidden, final_norm_w, eps)
    print(f"After final norm: mean={hidden.mean().item():.6f}")

    # Output projection
    output_proj_w = weights["decoder.pre_transformer.output_proj.weight"]
    output_proj_b = weights["decoder.pre_transformer.output_proj.bias"]
    hidden = torch.nn.functional.linear(hidden, output_proj_w, output_proj_b)
    print(f"After output proj: {hidden.shape}")
    save_tensor(output_dir / "decoder_output_proj.bin", hidden)

    # Now hidden is [batch, seq, latent_dim=1024]
    # Transpose to [batch, latent_dim, seq] for conv operations
    hidden = hidden.transpose(1, 2)
    print(f"Transposed for conv: {hidden.shape}")

    # ===== 5. Upsample stages (2 stages, each 2x) =====
    print("\n=== Upsample Stages ===")

    for stage in range(2):
        print(f"\n  Stage {stage}:")

        # 5.0: CausalTransConvNet (upsampling)
        # For upsampling_ratios=[2,2], kernel_size=factor, stride=factor
        conv_w = weights[f"decoder.upsample.{stage}.0.conv.weight"]
        conv_b = weights[f"decoder.upsample.{stage}.0.conv.bias"]
        kernel_size = conv_w.shape[2]  # 2
        stride = kernel_size  # stride == kernel_size for these upsample stages

        # ConvTranspose1d
        hidden = torch.nn.functional.conv_transpose1d(hidden, conv_w, conv_b, stride=stride)
        # Trim to maintain causality: pad = kernel_size - stride
        # For kernel=stride, pad=0, so no trimming needed
        pad = kernel_size - stride
        if pad > 0:
            left_pad = (pad + 1) // 2
            right_pad = pad - left_pad
            if right_pad > 0:
                hidden = hidden[..., left_pad:-right_pad]
            else:
                hidden = hidden[..., left_pad:]
        print(f"    After trans_conv: {hidden.shape}")
        save_tensor(output_dir / f"decoder_upsample_{stage}_0.bin", hidden)

        # 5.1: ConvNeXtBlock
        # dwconv (depthwise causal conv)
        dw_w = weights[f"decoder.upsample.{stage}.1.dwconv.conv.weight"]
        dw_b = weights[f"decoder.upsample.{stage}.1.dwconv.conv.bias"]
        dw_kernel = dw_w.shape[2]  # 7
        dw_padding = dw_kernel - 1  # Causal padding
        hidden_dw = torch.nn.functional.pad(hidden, (dw_padding, 0))
        hidden_dw = torch.nn.functional.conv1d(hidden_dw, dw_w, dw_b, groups=hidden.shape[1])

        # Transpose for LayerNorm: [B, C, T] -> [B, T, C]
        hidden_dw = hidden_dw.transpose(1, 2)

        # LayerNorm
        norm_w = weights[f"decoder.upsample.{stage}.1.norm.weight"]
        norm_b = weights[f"decoder.upsample.{stage}.1.norm.bias"]
        hidden_norm = torch.nn.functional.layer_norm(hidden_dw, (hidden_dw.shape[-1],), norm_w, norm_b, eps=1e-6)

        # pwconv1 (linear)
        pw1_w = weights[f"decoder.upsample.{stage}.1.pwconv1.weight"]
        pw1_b = weights[f"decoder.upsample.{stage}.1.pwconv1.bias"]
        hidden_pw1 = torch.nn.functional.linear(hidden_norm, pw1_w, pw1_b)

        # GELU
        hidden_gelu = torch.nn.functional.gelu(hidden_pw1)

        # pwconv2 (linear)
        pw2_w = weights[f"decoder.upsample.{stage}.1.pwconv2.weight"]
        pw2_b = weights[f"decoder.upsample.{stage}.1.pwconv2.bias"]
        hidden_pw2 = torch.nn.functional.linear(hidden_gelu, pw2_w, pw2_b)

        # Gamma scaling
        gamma = weights[f"decoder.upsample.{stage}.1.gamma"]
        hidden_scaled = hidden_pw2 * gamma

        # Transpose back: [B, T, C] -> [B, C, T]
        hidden_scaled = hidden_scaled.transpose(1, 2)

        # Residual
        hidden = hidden + hidden_scaled
        print(f"    After ConvNeXt: {hidden.shape}")
        save_tensor(output_dir / f"decoder_upsample_{stage}_1.bin", hidden)

    print(f"\nAfter all upsampling: {hidden.shape}")

    # ===== 6. Decoder blocks (4 stages with BigVGAN-style blocks) =====
    print("\n=== Decoder Blocks ===")

    # decoder.0: Initial causal conv to decoder_dim
    conv0_w = weights["decoder.decoder.0.conv.weight"]
    conv0_b = weights["decoder.decoder.0.conv.bias"]
    kernel_size = conv0_w.shape[2]  # 7
    padding = kernel_size - 1
    hidden = torch.nn.functional.pad(hidden, (padding, 0))
    hidden = torch.nn.functional.conv1d(hidden, conv0_w, conv0_b)
    print(f"After decoder.0 (initial conv): {hidden.shape}")
    save_tensor(output_dir / "decoder_decoder_0.bin", hidden)

    # Define SnakeBeta activation
    def snake_beta(x, alpha, beta, eps=1e-9):
        """SnakeBeta: x + (1/beta) * sin²(alpha * x), with exp(alpha/beta)"""
        alpha = alpha.view(1, -1, 1)
        beta = beta.view(1, -1, 1)
        alpha = torch.exp(alpha)
        beta = torch.exp(beta)
        return x + (1.0 / (beta + eps)) * torch.pow(torch.sin(x * alpha), 2)

    # Export a dedicated SnakeBeta test case
    print("\n=== SnakeBeta Test ===")
    snake_test_input = hidden.clone()  # Use decoder.0 output as input
    snake_alpha_test = weights["decoder.decoder.1.block.0.alpha"]
    snake_beta_test = weights["decoder.decoder.1.block.0.beta"]
    snake_test_output = snake_beta(snake_test_input, snake_alpha_test, snake_beta_test)
    save_tensor(output_dir / "snake_beta_input.bin", snake_test_input)
    save_tensor(output_dir / "snake_beta_alpha.bin", snake_alpha_test)
    save_tensor(output_dir / "snake_beta_beta.bin", snake_beta_test)
    save_tensor(output_dir / "snake_beta_output.bin", snake_test_output)
    print(f"  Input: {snake_test_input.shape}, Output: {snake_test_output.shape}")

    # decoder.1-4: DecoderBlocks
    upsample_rates = dec_config["upsample_rates"]  # [8, 5, 4, 3]
    current_dim = conv0_w.shape[0]  # 1536

    for block_idx in range(1, 5):
        print(f"\n  Decoder block {block_idx}:")
        prefix = f"decoder.decoder.{block_idx}.block"

        # block.0: SnakeBeta
        snake_alpha = weights[f"{prefix}.0.alpha"]
        snake_beta_param = weights[f"{prefix}.0.beta"]
        hidden = snake_beta(hidden, snake_alpha, snake_beta_param)
        print(f"    After SnakeBeta: mean={hidden.mean().item():.6f}")

        # block.1: CausalTransConvNet (upsampling)
        rate = upsample_rates[block_idx - 1]
        conv_w = weights[f"{prefix}.1.conv.weight"]
        conv_b = weights[f"{prefix}.1.conv.bias"]
        kernel_size = conv_w.shape[2]  # 2 * rate
        stride = rate

        hidden = torch.nn.functional.conv_transpose1d(hidden, conv_w, conv_b, stride=stride)
        # Trim from both sides to match official Qwen3-TTS model
        # pad = kernel_size - stride, trim ceil(pad) from each side
        trim = kernel_size - stride
        if trim > 0:
            hidden = hidden[..., trim:-trim]
        print(f"    After TransConv (rate={rate}): {hidden.shape}")

        # block.2-4: 3 ResidualUnits (dilations 1, 3, 9)
        for unit_idx, dilation in enumerate([1, 3, 9], start=2):
            unit_prefix = f"{prefix}.{unit_idx}"
            residual = hidden

            # act1: SnakeBeta
            a1_alpha = weights[f"{unit_prefix}.act1.alpha"]
            a1_beta = weights[f"{unit_prefix}.act1.beta"]
            hidden = snake_beta(hidden, a1_alpha, a1_beta)

            # conv1: dilated causal conv (kernel=7)
            c1_w = weights[f"{unit_prefix}.conv1.conv.weight"]
            c1_b = weights[f"{unit_prefix}.conv1.conv.bias"]
            k1 = c1_w.shape[2]
            pad1 = dilation * (k1 - 1)
            h_padded = torch.nn.functional.pad(hidden, (pad1, 0))
            hidden = torch.nn.functional.conv1d(h_padded, c1_w, c1_b, dilation=dilation)

            # act2: SnakeBeta
            a2_alpha = weights[f"{unit_prefix}.act2.alpha"]
            a2_beta = weights[f"{unit_prefix}.act2.beta"]
            hidden = snake_beta(hidden, a2_alpha, a2_beta)

            # conv2: 1x1 conv
            c2_w = weights[f"{unit_prefix}.conv2.conv.weight"]
            c2_b = weights[f"{unit_prefix}.conv2.conv.bias"]
            hidden = torch.nn.functional.conv1d(hidden, c2_w, c2_b)

            # Residual
            hidden = hidden + residual

        print(f"    After block: {hidden.shape}, mean={hidden.mean().item():.6f}")
        save_tensor(output_dir / f"decoder_decoder_{block_idx}.bin", hidden)

    # decoder.5: Final SnakeBeta
    snake5_alpha = weights["decoder.decoder.5.alpha"]
    snake5_beta = weights["decoder.decoder.5.beta"]
    hidden = snake_beta(hidden, snake5_alpha, snake5_beta)
    print(f"\nAfter decoder.5 (final SnakeBeta): {hidden.shape}")
    save_tensor(output_dir / "decoder_decoder_5.bin", hidden)

    # decoder.6: Final causal conv to 1 channel
    conv6_w = weights["decoder.decoder.6.conv.weight"]
    conv6_b = weights["decoder.decoder.6.conv.bias"]
    kernel_size = conv6_w.shape[2]  # 7
    padding = kernel_size - 1
    hidden = torch.nn.functional.pad(hidden, (padding, 0))
    hidden = torch.nn.functional.conv1d(hidden, conv6_w, conv6_b)
    print(f"After decoder.6 (final conv): {hidden.shape}")
    save_tensor(output_dir / "decoder_output.bin", hidden)

    print(f"\nFinal audio shape: {hidden.shape}")
    print(f"Total samples: {hidden.shape[2]}")

    # ===== Save metadata =====
    metadata = {
        "codes": codes.tolist(),
        "batch_size": batch_size,
        "num_quantizers": num_quantizers,
        "seq_len": seq_len,
        "codebook_dim": codebook_dim,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
    }

    with open(output_dir / "decoder_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n=== Saved decoder reference values to {output_dir} ===")


def save_tensor(path, tensor):
    """Save tensor as raw float32 binary."""
    np.array(tensor.detach().cpu().numpy(), dtype=np.float32).tofile(path)
    print(f"  Saved {path.name}: {tensor.shape}")


if __name__ == "__main__":
    export_decoder_reference()
