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
    codebook_dim = dec_config["codebook_dim"]  # 512

    # First quantizer (semantic)
    first_codebook = weights["decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"]
    print(f"First codebook shape: {first_codebook.shape}")

    # Rest quantizers (acoustic)
    rest_codebooks = []
    for i in range(15):
        cb = weights[f"decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum"]
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

    # ===== 2. Pre-conv =====
    print("\n=== Pre-conv ===")
    # pre_conv: [1024, 512, 3] conv1d
    pre_conv_w = weights["decoder.pre_conv.conv.weight"]
    pre_conv_b = weights["decoder.pre_conv.conv.bias"]
    print(f"Pre-conv weight: {pre_conv_w.shape}")

    # Need to transpose for conv1d: [batch, seq, channels] -> [batch, channels, seq]
    x = quantized.transpose(1, 2)  # [1, 512, 2]
    # Causal padding (left only)
    padding = pre_conv_w.shape[2] - 1  # kernel_size - 1
    x_padded = torch.nn.functional.pad(x, (padding, 0))
    pre_conv_out = torch.nn.functional.conv1d(x_padded, pre_conv_w, pre_conv_b)
    print(f"Pre-conv output: {pre_conv_out.shape}")
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

    print(f"Pre-transformer output: {hidden.shape}")
    save_tensor(output_dir / "decoder_pre_transformer.bin", hidden)

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
