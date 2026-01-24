#!/usr/bin/env python3
"""Generate TTS audio using the CustomVoice model with built-in speakers."""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
import scipy.io.wavfile as wavfile


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply RMS normalization."""
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return x * weight


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU activation function."""
    return x * torch.sigmoid(x)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """Apply rotary position embeddings."""
    cos = cos[position_ids].unsqueeze(1)  # [batch, 1, seq_len, head_dim]
    sin = sin[position_ids].unsqueeze(1)

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def build_rope_cache(max_seq_len: int, head_dim: int, theta: float = 1000000.0):
    """Build rotary position embedding cache."""
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
):
    """Single transformer layer forward pass."""
    batch, seq_len, hidden_size = hidden_states.shape

    # Input layernorm
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, weights[f"{prefix}.input_layernorm.weight"], eps)

    # Self attention
    q = torch.nn.functional.linear(hidden_states, weights[f"{prefix}.self_attn.q_proj.weight"])
    k = torch.nn.functional.linear(hidden_states, weights[f"{prefix}.self_attn.k_proj.weight"])
    v = torch.nn.functional.linear(hidden_states, weights[f"{prefix}.self_attn.v_proj.weight"])

    q = q.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)
    k = k.view(batch, seq_len, n_kv_heads, head_dim).transpose(1, 2)
    v = v.view(batch, seq_len, n_kv_heads, head_dim).transpose(1, 2)

    # Apply RoPE
    q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin, position_ids)

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

    return residual + hidden_states


def generate_semantic_tokens(
    input_ids: torch.Tensor,
    weights: dict,
    config: dict,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    top_k: int = 50,
):
    """Generate semantic tokens using the talker model."""
    talker_config = config["talker_config"]
    hidden_size = talker_config["hidden_size"]
    n_layers = talker_config["num_hidden_layers"]
    n_heads = talker_config["num_attention_heads"]
    n_kv_heads = talker_config["num_key_value_heads"]
    head_dim = talker_config["head_dim"]
    intermediate_size = talker_config["intermediate_size"]
    eps = talker_config["rms_norm_eps"]
    rope_theta = talker_config["rope_theta"]
    vocab_size = talker_config["vocab_size"]

    # Special tokens
    codec_bos_id = talker_config["codec_bos_id"]  # 2149
    codec_eos_id = talker_config["codec_eos_token_id"]  # 2150
    codec_nothink_id = talker_config["codec_nothink_id"]  # 2155

    print(f"Talker config: hidden_size={hidden_size}, n_layers={n_layers}, n_heads={n_heads}")
    print(f"Special tokens: bos={codec_bos_id}, eos={codec_eos_id}, nothink={codec_nothink_id}")

    # Build RoPE cache
    rope_cos, rope_sin = build_rope_cache(8192, head_dim, rope_theta)

    # Get embeddings - note the weight names have "talker.model." prefix
    text_embed_weight = weights["talker.model.text_embedding.weight"]
    codec_embed_weight = weights["talker.model.codec_embedding.weight"]

    print(f"Text embedding shape: {text_embed_weight.shape}")
    print(f"Codec embedding shape: {codec_embed_weight.shape}")

    # Start with input tokens (speaker + text)
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]

    # Embed input (these are text tokens)
    hidden_states = text_embed_weight[input_ids]
    print(f"Initial hidden shape: {hidden_states.shape}")

    # We'll generate tokens autoregressively
    # Start by appending the BOS token for codec
    generated = [codec_bos_id]

    # Create position ids
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    # Run through talker layers for the initial context
    for layer_idx in range(n_layers):
        prefix = f"talker.model.layers.{layer_idx}"
        hidden_states = transformer_layer(
            hidden_states,
            weights,
            prefix,
            n_heads,
            n_kv_heads,
            head_dim,
            intermediate_size,
            rope_cos,
            rope_sin,
            position_ids,
            eps,
        )

    # Final norm
    hidden_states = rms_norm(hidden_states, weights["talker.model.norm.weight"], eps)

    # LM head projection to get logits - uses codec_head for codec token prediction
    lm_head_weight = weights["talker.codec_head.weight"]
    logits = torch.nn.functional.linear(hidden_states[:, -1:, :], lm_head_weight)
    print(f"Logits shape: {logits.shape}")

    # Sample first token
    logits = logits[:, -1, :] / temperature
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).item()
    generated.append(next_token)
    print(f"First generated token: {next_token}")

    # Continue generating
    for step in range(max_new_tokens - 1):
        # Embed the new token (it's a codec token)
        if next_token < codec_embed_weight.shape[0]:
            new_embed = codec_embed_weight[next_token].unsqueeze(0).unsqueeze(0)
        else:
            print(f"Warning: token {next_token} out of codec embedding range")
            break

        # Update position
        current_pos = seq_len + step + 1
        position_ids = torch.tensor([[current_pos]], dtype=torch.long)

        # Simple forward pass for single token (no KV cache for simplicity)
        # This is inefficient but works for debugging
        all_tokens = input_ids.tolist()[0] + generated[:-1]
        all_embeds = []
        for i, tok in enumerate(all_tokens):
            if i < input_ids.shape[1]:
                all_embeds.append(text_embed_weight[tok])
            else:
                all_embeds.append(codec_embed_weight[tok])
        all_embeds.append(codec_embed_weight[next_token])
        hidden_states = torch.stack(all_embeds).unsqueeze(0)

        position_ids = torch.arange(hidden_states.shape[1], dtype=torch.long).unsqueeze(0)

        # Create causal mask
        seq_len_full = hidden_states.shape[1]
        causal_mask = torch.triu(torch.full((seq_len_full, seq_len_full), float('-inf')), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        for layer_idx in range(n_layers):
            prefix = f"talker.model.layers.{layer_idx}"
            hidden_states = transformer_layer(
                hidden_states,
                weights,
                prefix,
                n_heads,
                n_kv_heads,
                head_dim,
                intermediate_size,
                rope_cos,
                rope_sin,
                position_ids,
                eps,
                attention_mask=causal_mask,
            )

        hidden_states = rms_norm(hidden_states, weights["talker.model.norm.weight"], eps)
        logits = torch.nn.functional.linear(hidden_states[:, -1:, :], lm_head_weight)

        logits = logits[:, -1, :] / temperature
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_token)

        if next_token == codec_eos_id:
            print(f"Hit EOS token at step {step + 1}")
            break

        if step % 10 == 0:
            print(f"Step {step + 1}, token: {next_token}")

    print(f"Generated {len(generated)} semantic tokens")

    # Return both tokens and the final hidden states (before codec_head projection)
    # The hidden states are used by the code predictor
    return generated, hidden_states


def generate_acoustic_codes(
    semantic_tokens: list,
    talker_hidden_states: torch.Tensor,
    weights: dict,
    config: dict,
    num_code_groups: int = 15,
):
    """Generate acoustic codes from semantic tokens using code predictor.

    The code predictor takes the hidden states from the talker model and
    predicts acoustic codes for each code group.

    Args:
        semantic_tokens: The semantic tokens generated by the talker
        talker_hidden_states: Hidden states from the talker model [batch, seq, 2048]
        weights: Model weights dict
        config: Model config dict
        num_code_groups: Number of code groups (15 for this model)

    Returns:
        codes: [seq_len, num_code_groups] acoustic codes
    """
    cp_config = config["talker_config"]["code_predictor_config"]
    hidden_size = cp_config["hidden_size"]  # 1024
    n_layers = cp_config["num_hidden_layers"]  # 5
    n_heads = cp_config["num_attention_heads"]  # 16
    n_kv_heads = cp_config["num_key_value_heads"]  # 8
    head_dim = cp_config["head_dim"]  # 128
    intermediate_size = cp_config["intermediate_size"]  # 3072
    eps = cp_config["rms_norm_eps"]  # 1e-6
    rope_theta = cp_config["rope_theta"]  # 1000000
    codebook_size = 2048

    print(f"\nCode Predictor config: hidden_size={hidden_size}, n_layers={n_layers}")
    print(f"Talker hidden states shape: {talker_hidden_states.shape}")

    # Project from talker hidden size (2048) to code predictor hidden size (1024)
    small_to_mtp = weights["talker.code_predictor.small_to_mtp_projection.weight"]
    hidden_states = torch.nn.functional.linear(talker_hidden_states, small_to_mtp)
    print(f"After projection: {hidden_states.shape}")

    # Build RoPE cache
    rope_cos, rope_sin = build_rope_cache(8192, head_dim, rope_theta)

    batch_size, seq_len, _ = hidden_states.shape
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    # Create causal mask
    causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    # Run through code predictor layers
    for layer_idx in range(n_layers):
        prefix = f"talker.code_predictor.model.layers.{layer_idx}"
        hidden_states = transformer_layer(
            hidden_states,
            weights,
            prefix,
            n_heads,
            n_kv_heads,
            head_dim,
            intermediate_size,
            rope_cos,
            rope_sin,
            position_ids,
            eps,
            attention_mask=causal_mask,
        )

    # Final norm
    hidden_states = rms_norm(hidden_states, weights["talker.code_predictor.model.norm.weight"], eps)

    # Project to code groups
    # Each code group has its own head projecting to codebook_size (2048)
    all_codes = []
    for g in range(num_code_groups):
        head_key = f"talker.code_predictor.lm_head.{g}.weight"
        if head_key in weights:
            logits = torch.nn.functional.linear(hidden_states, weights[head_key])
            codes = torch.argmax(logits, dim=-1)
            all_codes.append(codes)
        else:
            print(f"Warning: missing head {head_key}")
            all_codes.append(torch.zeros(batch_size, seq_len, dtype=torch.long))

    # Stack: [batch, seq_len, num_groups]
    codes = torch.stack(all_codes, dim=-1)
    print(f"Generated codes shape: {codes.shape}")
    return codes[0].numpy()  # Return [seq_len, num_groups]


def decode_to_audio(
    codes: np.ndarray,
    decoder_weights: dict,
    decoder_config: dict,
):
    """Decode acoustic codes to audio waveform."""
    # This uses the 12Hz decoder
    dec_config = decoder_config["decoder_config"]
    hidden_size = dec_config["hidden_size"]  # 512
    n_layers = dec_config["num_hidden_layers"]  # 8
    n_heads = dec_config["num_attention_heads"]  # 16
    n_kv_heads = dec_config["num_key_value_heads"]  # 16
    head_dim = dec_config["head_dim"]  # 64
    intermediate_size = dec_config["intermediate_size"]  # 1024
    eps = dec_config["rms_norm_eps"]  # 1e-5
    latent_dim = dec_config["latent_dim"]  # 1024
    decoder_dim = dec_config["decoder_dim"]  # 1536
    codebook_size = dec_config["codebook_size"]  # 2048
    codebook_dim = dec_config["codebook_dim"]  # 512
    num_quantizers = dec_config["num_quantizers"]  # 16
    upsample_rates = dec_config["upsample_rates"]  # [8, 5, 4, 3]

    print(f"\nDecoder config: hidden_size={hidden_size}, n_layers={n_layers}")
    print(f"Codes shape: {codes.shape}")

    # Get codebook embeddings
    codebook = decoder_weights["decoder.code_layer.codebook.weight"]
    print(f"Codebook shape: {codebook.shape}")

    # Embed codes: codes is [seq_len, num_quantizers]
    # Codebook is [num_quantizers * codebook_size, codebook_dim]
    seq_len, n_quant = codes.shape
    embedded = []
    for q in range(n_quant):
        offset = q * codebook_size
        indices = codes[:, q] + offset
        embedded.append(codebook[indices])
    embedded = torch.stack(embedded, dim=1)  # [seq_len, n_quant, codebook_dim]
    embedded = embedded.sum(dim=1)  # Sum over quantizers -> [seq_len, codebook_dim]

    # Project to latent
    latent_proj = decoder_weights["decoder.code_layer.latent_proj.weight"]
    hidden = torch.nn.functional.linear(embedded, latent_proj)
    hidden = hidden.unsqueeze(0)  # [1, seq_len, latent_dim]

    print(f"After latent proj: {hidden.shape}")

    # Pre-transformer: project to hidden size
    pre_proj = decoder_weights["decoder.pre_transformer.input_proj.weight"]
    pre_proj_b = decoder_weights["decoder.pre_transformer.input_proj.bias"]
    hidden = torch.nn.functional.linear(hidden, pre_proj, pre_proj_b)

    print(f"After pre_transformer input_proj: {hidden.shape}")

    # Build RoPE cache for decoder
    rope_theta = dec_config["rope_theta"]
    rope_cos, rope_sin = build_rope_cache(8192, head_dim, rope_theta)
    position_ids = torch.arange(hidden.shape[1], dtype=torch.long).unsqueeze(0)

    # Run through transformer layers
    for layer_idx in range(n_layers):
        prefix = f"decoder.pre_transformer.layers.{layer_idx}"
        hidden = transformer_layer(
            hidden,
            decoder_weights,
            prefix,
            n_heads,
            n_kv_heads,
            head_dim,
            intermediate_size,
            rope_cos,
            rope_sin,
            position_ids,
            eps,
        )

    # Final norm and output projection
    hidden = rms_norm(hidden, decoder_weights["decoder.pre_transformer.norm.weight"], eps)
    out_proj = decoder_weights["decoder.pre_transformer.output_proj.weight"]
    out_proj_b = decoder_weights["decoder.pre_transformer.output_proj.bias"]
    hidden = torch.nn.functional.linear(hidden, out_proj, out_proj_b)

    print(f"After pre_transformer output: {hidden.shape}")

    # Now run through the upsampling decoder
    # This is the convolutional decoder part
    hidden = hidden.transpose(1, 2)  # [1, decoder_dim, seq_len]

    # Upsampling stages
    for i, rate in enumerate(upsample_rates):
        # ConvTranspose1d upsample
        conv_w = decoder_weights[f"decoder.decoder.model.{i}.conv.conv.weight"]
        conv_b = decoder_weights.get(f"decoder.decoder.model.{i}.conv.conv.bias")
        hidden = torch.nn.functional.conv_transpose1d(
            hidden, conv_w.transpose(0, 1), conv_b, stride=rate, padding=rate // 2
        )
        hidden = silu(hidden)

        # Residual block
        res_prefix = f"decoder.decoder.model.{i}.block.0"
        if f"{res_prefix}.conv.conv.weight" in decoder_weights:
            res_w = decoder_weights[f"{res_prefix}.conv.conv.weight"]
            res_b = decoder_weights.get(f"{res_prefix}.conv.conv.bias")
            res = torch.nn.functional.conv1d(hidden, res_w, res_b, padding=res_w.shape[-1] // 2)
            res = silu(res)
            res_w2 = decoder_weights[f"{res_prefix}.conv2.conv.weight"]
            res_b2 = decoder_weights.get(f"{res_prefix}.conv2.conv.bias")
            res = torch.nn.functional.conv1d(res, res_w2, res_b2, padding=res_w2.shape[-1] // 2)
            hidden = hidden + res

    # Final conv to audio
    final_conv_w = decoder_weights["decoder.decoder.model.4.conv.weight"]
    final_conv_b = decoder_weights.get("decoder.decoder.model.4.conv.bias")
    audio = torch.nn.functional.conv1d(hidden, final_conv_w, final_conv_b, padding=final_conv_w.shape[-1] // 2)

    audio = torch.tanh(audio)
    return audio[0, 0].numpy()  # [samples]


def main():
    parser = argparse.ArgumentParser(description="Generate TTS with CustomVoice model")
    parser.add_argument("--text", type=str, default="Hello", help="Text to synthesize")
    parser.add_argument("--speaker", type=str, default="ryan", help="Speaker name")
    parser.add_argument("--model-dir", type=str, default="test_data/model_customvoice", help="Model directory")
    parser.add_argument("--output", type=str, default="test_data/customvoice_output.wav", help="Output WAV file")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max semantic tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
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

    # Load speech tokenizer config
    with open(model_dir / "speech_tokenizer" / "config.json") as f:
        decoder_config = json.load(f)

    # Get speaker ID
    speaker_ids = config["talker_config"]["spk_id"]
    if args.speaker not in speaker_ids:
        print(f"Unknown speaker: {args.speaker}")
        print(f"Available speakers: {list(speaker_ids.keys())}")
        return
    speaker_id = speaker_ids[args.speaker]
    print(f"Speaker: {args.speaker} (ID: {speaker_id})")

    # Simple text to token mapping (hardcoded for now)
    text_tokens = {
        "Hello": [9707],
        "Hello world": [9707, 1917],
        "Hello, this is a test": [9707, 11, 419, 374, 264, 1273],
    }
    if args.text not in text_tokens:
        print(f"Text not in hardcoded mapping. Using 'Hello'")
        tokens = text_tokens["Hello"]
    else:
        tokens = text_tokens[args.text]

    # Build input: [speaker_id, text_tokens...]
    input_ids = torch.tensor([[speaker_id] + tokens], dtype=torch.long)
    print(f"Input IDs: {input_ids.tolist()}")

    # Load model weights and convert to float32
    print("Loading model weights...")
    weights = load_file(model_dir / "model.safetensors")
    weights = {k: v.float() for k, v in weights.items()}
    print(f"Loaded {len(weights)} weight tensors")

    # Load decoder weights and convert to float32
    print("Loading decoder weights...")
    decoder_weights = load_file(model_dir / "speech_tokenizer" / "model.safetensors")
    decoder_weights = {k: v.float() for k, v in decoder_weights.items()}
    print(f"Loaded {len(decoder_weights)} decoder weight tensors")

    # Generate semantic tokens
    print("\n=== Generating semantic tokens ===")
    semantic_tokens, talker_hidden = generate_semantic_tokens(
        input_ids,
        weights,
        config,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print(f"Semantic tokens: {semantic_tokens[:20]}...")
    print(f"Talker hidden states shape: {talker_hidden.shape}")

    # Generate acoustic codes
    print("\n=== Generating acoustic codes ===")
    codes = generate_acoustic_codes(semantic_tokens, talker_hidden, weights, config)
    print(f"Acoustic codes shape: {codes.shape}")
    print(f"First few codes:\n{codes[:5]}")

    # Save codes for decoding with official decoder
    codes_path = Path(args.output).with_suffix('.npy')
    codes_path.parent.mkdir(parents=True, exist_ok=True)

    # The decoder expects [batch, seq_len, 16] where:
    # - Column 0: semantic token
    # - Columns 1-15: acoustic codes from code predictor
    # We need to align the semantic tokens with the acoustic codes
    # The code predictor produces one set of acoustic codes per semantic token position

    # Only use semantic tokens (excluding BOS token at position 0)
    # The codes array has one row per position in the talker hidden states
    # We need to match the generated semantic tokens to the acoustic codes
    num_generated = len(semantic_tokens) - 1  # Exclude BOS
    num_frames = min(num_generated, codes.shape[0])

    # Build full codes array: [semantic, acoustic_1, ..., acoustic_15]
    full_codes = np.zeros((num_frames, 16), dtype=np.int64)
    # Put semantic tokens (excluding BOS) in first column
    full_codes[:, 0] = semantic_tokens[1:num_frames+1]
    # Put acoustic codes in remaining columns (slice if needed)
    full_codes[:, 1:] = codes[:num_frames]

    np.save(str(codes_path), full_codes)
    print(f"Saved codes to {codes_path}: shape {full_codes.shape}")
    print(f"First few full codes:\n{full_codes[:5]}")

    print("\nTo decode to audio, run:")
    print(f"  uv run python decode_with_official.py --codes-file {codes_path} --output {args.output}")


if __name__ == "__main__":
    main()
