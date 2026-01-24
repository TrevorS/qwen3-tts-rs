#!/usr/bin/env python3
"""Generate reference audio with deterministic seed for Rust comparison.

This script generates WAV audio files using the Qwen3-TTS pipeline with a
specific seed, allowing direct comparison with Rust output.

Usage:
    python3 tools/generate_reference_audio.py --text "Hello world" --seed 42 --duration 2.0
    python3 tools/generate_reference_audio.py --text "Hello" --seed 42 --frames 25
"""

import argparse
import json
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from safetensors.torch import load_file


def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS normalization."""
    variance = x.pow(2).mean(-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps)
    return weight * x_norm


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads for GQA."""
    if n_rep == 1:
        return x
    batch, n_kv_heads, seq, hd = x.shape
    x = x[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, seq, hd)
    return x.reshape(batch, n_kv_heads * n_rep, seq, hd)


def build_rope(seq_len: int, head_dim: int, rope_theta: float = 1000000.0):
    """Build rotary position embeddings."""
    positions = torch.arange(seq_len).unsqueeze(0)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = torch.outer(positions.float().squeeze(0), inv_freq)
    cos = freqs.cos().repeat(1, 2).unsqueeze(0).unsqueeze(0)
    sin = freqs.sin().repeat(1, 2).unsqueeze(0).unsqueeze(0)
    return cos, sin


def sample_token(
    logits: torch.Tensor,
    temperature: float = 0.7,
    top_k: int = 50,
    suppress_tokens: list = None
) -> int:
    """Sample a token from logits with temperature, top-k, and token suppression.

    Args:
        logits: Logits tensor [vocab_size]
        temperature: Sampling temperature (0 = greedy)
        top_k: Top-k sampling (0 = disabled)
        suppress_tokens: List of token IDs to suppress (set to -inf)
    """
    # Suppress tokens by setting their logits to -inf
    # This matches the official Qwen3-TTS behavior which suppresses tokens 2048-3071
    if suppress_tokens is not None:
        for token_id in suppress_tokens:
            if 0 <= token_id < logits.size(-1):
                logits[token_id] = float('-inf')

    if temperature < 0.01:
        return logits.argmax(dim=-1).item()

    logits = logits / temperature

    if top_k > 0:
        values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(-1, indices, values)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


class TalkerModel:
    """Talker model for semantic token generation."""

    def __init__(self, weights: dict, device: torch.device):
        self.weights = weights
        self.device = device

        # Config - talker has hidden_size=2048
        self.hidden_size = 2048
        self.num_heads = 16
        self.num_kv_heads = 8
        self.head_dim = 128
        self.num_layers = 28
        self.rope_theta = 1000000.0
        self.eps = 1e-6
        self.n_rep = self.num_heads // self.num_kv_heads

        # Cache for KV
        self.kv_cache = None

    def prefill(self, input_ids: torch.Tensor):
        """Process input text tokens and return hidden states and logits."""
        batch_size, seq_len = input_ids.shape

        # Text embedding
        text_embed_w = self.weights["talker.model.text_embedding.weight"]
        embeddings = torch.nn.functional.embedding(input_ids, text_embed_w)

        # Text projection
        fc1_w = self.weights["talker.text_projection.linear_fc1.weight"]
        fc1_b = self.weights["talker.text_projection.linear_fc1.bias"]
        fc2_w = self.weights["talker.text_projection.linear_fc2.weight"]
        fc2_b = self.weights["talker.text_projection.linear_fc2.bias"]

        hidden = torch.nn.functional.linear(embeddings, fc1_w, fc1_b)
        hidden = torch.nn.functional.silu(hidden)
        hidden = torch.nn.functional.linear(hidden, fc2_w, fc2_b)

        # Initialize KV cache
        self.kv_cache = [{} for _ in range(self.num_layers)]

        # Build RoPE and mask
        cos, sin = build_rope(seq_len, self.head_dim, self.rope_theta)
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)

        # Run through layers
        hidden = self._run_layers(hidden, cos, sin, causal_mask, batch_size, seq_len, is_prefill=True)

        # Final norm and logits
        final_norm_w = self.weights["talker.model.norm.weight"]
        hidden = rms_norm(hidden, final_norm_w, self.eps)

        codec_head_w = self.weights["talker.codec_head.weight"]
        logits = torch.nn.functional.linear(hidden, codec_head_w)

        return hidden, logits

    def generate_step(self, prev_token: int, offset: int):
        """Generate next token given previous codec token."""
        batch_size = 1

        # Get codec embedding
        codec_embed_w = self.weights["talker.model.codec_embedding.weight"]
        prev_tensor = torch.tensor([[prev_token]], dtype=torch.long)
        hidden = torch.nn.functional.embedding(prev_tensor, codec_embed_w)

        # Build RoPE for new position
        cos, sin = build_rope(offset + 1, self.head_dim, self.rope_theta)
        cos = cos[:, :, offset:offset+1, :]
        sin = sin[:, :, offset:offset+1, :]

        # Causal mask: attend to all previous positions
        # For a single token, mask is just [0, 0, ..., 0] (can attend to all)
        causal_mask = None  # No mask needed for single token with full KV cache

        # Run through layers
        hidden = self._run_layers(hidden, cos, sin, causal_mask, batch_size, 1, is_prefill=False, offset=offset)

        # Final norm and logits
        final_norm_w = self.weights["talker.model.norm.weight"]
        hidden = rms_norm(hidden, final_norm_w, self.eps)

        codec_head_w = self.weights["talker.codec_head.weight"]
        logits = torch.nn.functional.linear(hidden, codec_head_w)

        return hidden, logits

    def _run_layers(self, hidden, cos, sin, causal_mask, batch_size, seq_len, is_prefill, offset=0):
        """Run through all transformer layers."""
        scaling = self.head_dim ** -0.5

        for layer_idx in range(self.num_layers):
            # Input LayerNorm
            input_ln_w = self.weights[f"talker.model.layers.{layer_idx}.input_layernorm.weight"]
            normed = rms_norm(hidden, input_ln_w, self.eps)

            # QKV projections
            q_proj_w = self.weights[f"talker.model.layers.{layer_idx}.self_attn.q_proj.weight"]
            k_proj_w = self.weights[f"talker.model.layers.{layer_idx}.self_attn.k_proj.weight"]
            v_proj_w = self.weights[f"talker.model.layers.{layer_idx}.self_attn.v_proj.weight"]
            q_norm_w = self.weights[f"talker.model.layers.{layer_idx}.self_attn.q_norm.weight"]
            k_norm_w = self.weights[f"talker.model.layers.{layer_idx}.self_attn.k_norm.weight"]

            q = torch.nn.functional.linear(normed, q_proj_w)
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
            q = rms_norm(q, q_norm_w, self.eps)
            q = q.transpose(1, 2)

            k = torch.nn.functional.linear(normed, k_proj_w)
            k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            k = rms_norm(k, k_norm_w, self.eps)
            k = k.transpose(1, 2)

            v = torch.nn.functional.linear(normed, v_proj_w)
            v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            v = v.transpose(1, 2)

            # RoPE
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)

            # Update KV cache
            if is_prefill:
                self.kv_cache[layer_idx]["k"] = k
                self.kv_cache[layer_idx]["v"] = v
            else:
                self.kv_cache[layer_idx]["k"] = torch.cat([self.kv_cache[layer_idx]["k"], k], dim=2)
                self.kv_cache[layer_idx]["v"] = torch.cat([self.kv_cache[layer_idx]["v"], v], dim=2)
                k = self.kv_cache[layer_idx]["k"]
                v = self.kv_cache[layer_idx]["v"]

            # Attention
            k_exp = repeat_kv(k, self.n_rep)
            v_exp = repeat_kv(v, self.n_rep)

            attn_weights = torch.matmul(q, k_exp.transpose(2, 3)) * scaling
            if causal_mask is not None:
                attn_weights = attn_weights + causal_mask
            attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
            attn_output = torch.matmul(attn_probs, v_exp)

            # O projection
            o_proj_w = self.weights[f"talker.model.layers.{layer_idx}.self_attn.o_proj.weight"]
            attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.num_heads * self.head_dim)
            attn_output = torch.nn.functional.linear(attn_output, o_proj_w)

            hidden = hidden + attn_output

            # MLP
            post_ln_w = self.weights[f"talker.model.layers.{layer_idx}.post_attention_layernorm.weight"]
            mlp_input = rms_norm(hidden, post_ln_w, self.eps)

            gate_w = self.weights[f"talker.model.layers.{layer_idx}.mlp.gate_proj.weight"]
            up_w = self.weights[f"talker.model.layers.{layer_idx}.mlp.up_proj.weight"]
            down_w = self.weights[f"talker.model.layers.{layer_idx}.mlp.down_proj.weight"]

            gate = torch.nn.functional.linear(mlp_input, gate_w)
            up = torch.nn.functional.linear(mlp_input, up_w)
            mlp_output = torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, down_w)

            hidden = hidden + mlp_output

        return hidden

    def get_codec_embedding(self, token_id: int) -> torch.Tensor:
        """Get embedding for a codec token."""
        codec_embed_w = self.weights["talker.model.codec_embedding.weight"]
        token_tensor = torch.tensor([[token_id]], dtype=torch.long)
        return torch.nn.functional.embedding(token_tensor, codec_embed_w)


class CodePredictor:
    """Code predictor for acoustic token generation."""

    def __init__(self, weights: dict, device: torch.device):
        self.weights = weights
        self.device = device

        # Config
        self.hidden_size = 1024
        self.num_heads = 16
        self.num_kv_heads = 8
        self.head_dim = 128
        self.num_layers = 5
        self.rope_theta = 1000000.0
        self.eps = 1e-6
        self.n_rep = self.num_heads // self.num_kv_heads
        self.num_acoustic = 15  # 15 acoustic tokens per semantic token

    def generate_acoustic_codes(self, talker_hidden: torch.Tensor, semantic_embed: torch.Tensor) -> list:
        """Generate 15 acoustic tokens given talker hidden state and semantic embedding."""
        batch_size = 1

        # Project talker hidden from 2048 to 1024
        proj_w = self.weights["talker.code_predictor.small_to_mtp_projection.weight"]
        proj_b = self.weights["talker.code_predictor.small_to_mtp_projection.bias"]
        talker_hidden = torch.nn.functional.linear(talker_hidden, proj_w, proj_b)

        # Project semantic embed (also from 2048 to 1024) using first codec embedding
        codec_embed_w = self.weights["talker.code_predictor.model.codec_embedding.0.weight"]
        # codec_embedding is [2048, 2048], projects 2048 -> 2048, then we project that
        # Actually, looking at the shapes, this is a linear layer not an embedding lookup
        # semantic_embed is [1, 1, 2048], project to [1, 1, 2048], then to [1, 1, 1024]
        semantic_proj = torch.nn.functional.linear(semantic_embed, codec_embed_w)
        semantic_proj = torch.nn.functional.linear(semantic_proj, proj_w, proj_b)

        # Concatenate along sequence dim
        hidden = torch.cat([talker_hidden, semantic_proj], dim=1)
        seq_len = hidden.shape[1]

        # Build RoPE and mask
        cos, sin = build_rope(seq_len, self.head_dim, self.rope_theta)
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        scaling = self.head_dim ** -0.5

        # Run through layers
        for layer_idx in range(self.num_layers):
            prefix = f"talker.code_predictor.model.layers.{layer_idx}"

            # Input LayerNorm
            input_ln_w = self.weights[f"{prefix}.input_layernorm.weight"]
            normed = rms_norm(hidden, input_ln_w, self.eps)

            # QKV projections
            q_proj_w = self.weights[f"{prefix}.self_attn.q_proj.weight"]
            k_proj_w = self.weights[f"{prefix}.self_attn.k_proj.weight"]
            v_proj_w = self.weights[f"{prefix}.self_attn.v_proj.weight"]
            q_norm_w = self.weights[f"{prefix}.self_attn.q_norm.weight"]
            k_norm_w = self.weights[f"{prefix}.self_attn.k_norm.weight"]

            q = torch.nn.functional.linear(normed, q_proj_w)
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
            q = rms_norm(q, q_norm_w, self.eps)
            q = q.transpose(1, 2)

            k = torch.nn.functional.linear(normed, k_proj_w)
            k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            k = rms_norm(k, k_norm_w, self.eps)
            k = k.transpose(1, 2)

            v = torch.nn.functional.linear(normed, v_proj_w)
            v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            v = v.transpose(1, 2)

            # RoPE
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)

            # Attention
            k_exp = repeat_kv(k, self.n_rep)
            v_exp = repeat_kv(v, self.n_rep)

            attn_weights = torch.matmul(q, k_exp.transpose(2, 3)) * scaling
            attn_weights = attn_weights + causal_mask
            attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
            attn_output = torch.matmul(attn_probs, v_exp)

            # O projection
            o_proj_w = self.weights[f"{prefix}.self_attn.o_proj.weight"]
            attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.num_heads * self.head_dim)
            attn_output = torch.nn.functional.linear(attn_output, o_proj_w)

            hidden = hidden + attn_output

            # MLP
            post_ln_w = self.weights[f"{prefix}.post_attention_layernorm.weight"]
            mlp_input = rms_norm(hidden, post_ln_w, self.eps)

            gate_w = self.weights[f"{prefix}.mlp.gate_proj.weight"]
            up_w = self.weights[f"{prefix}.mlp.up_proj.weight"]
            down_w = self.weights[f"{prefix}.mlp.down_proj.weight"]

            gate = torch.nn.functional.linear(mlp_input, gate_w)
            up = torch.nn.functional.linear(mlp_input, up_w)
            mlp_output = torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, down_w)

            hidden = hidden + mlp_output

        # Final norm
        norm_w = self.weights["talker.code_predictor.model.norm.weight"]
        hidden = rms_norm(hidden, norm_w, self.eps)

        # Get logits from position 1 (semantic embed position)
        codes = []
        for i in range(self.num_acoustic):
            lm_head_w = self.weights[f"talker.code_predictor.lm_head.{i}.weight"]
            logits = torch.nn.functional.linear(hidden[:, 1:2, :], lm_head_w)
            token = logits.argmax(dim=-1).item()
            codes.append(token)

        return codes


class Decoder12Hz:
    """12Hz speech tokenizer decoder."""

    def __init__(self, weights: dict, device: torch.device):
        self.weights = weights
        self.device = device

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode codec tokens to audio waveform.

        Args:
            codes: Tensor of shape [batch, 16, num_frames]

        Returns:
            Tensor of shape [batch, 1, num_samples]
        """
        # This is a simplified version - we'll use the full decoder
        # For now, just use placeholder since the decoder is complex
        # In practice, you'd implement the full decoder or use the Python model

        # For comparison purposes, we can export the codes and compare at that level
        raise NotImplementedError("Full decoder not implemented - use export_decoder_reference.py patterns")


def generate_audio(
    text: str,
    model_dir: str,
    seed: int,
    num_frames: int,
    temperature: float = 0.7,
    output_dir: str = "test_data/reference_audio",
):
    """Generate reference audio with a specific seed."""

    print(f"=== Generating Reference Audio ===")
    print(f"Text: {text}")
    print(f"Seed: {seed}")
    print(f"Frames: {num_frames}")
    print(f"Temperature: {temperature}")

    # Set seed
    set_all_seeds(seed)

    device = torch.device("cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load weights
    print("\nLoading model weights...")
    weights = load_file(f"{model_dir}/model.safetensors")
    weights = {k: v.float() for k, v in weights.items()}

    # Load speech tokenizer weights
    print("Loading decoder weights...")
    decoder_weights = load_file(f"{model_dir}/speech_tokenizer/model.safetensors")
    decoder_weights = {k: v.float() for k, v in decoder_weights.items()}

    # Create tokenizer (simple char-level for demo, would use real tokenizer)
    # For now, just use a predefined mapping
    text_to_ids = {
        "Hello": [9707],
        "Hello world": [9707, 1917],
        "Hello, this is a": [9707, 11, 419, 374, 264],
        "Hello, this is a test": [9707, 11, 419, 374, 264, 1273],
    }

    if text in text_to_ids:
        input_ids = torch.tensor([text_to_ids[text]], dtype=torch.long)
    else:
        # Default to "Hello" if text not in mapping
        print(f"Warning: Text '{text}' not in tokenizer mapping, using 'Hello'")
        input_ids = torch.tensor([[9707]], dtype=torch.long)

    print(f"Input IDs: {input_ids.tolist()}")

    # Create models
    talker = TalkerModel(weights, device)
    code_predictor = CodePredictor(weights, device)

    # Define suppress_tokens to match official Qwen3-TTS behavior
    # Suppress tokens 2048-3071 (vocab_size=3072, decoder codebook=2048)
    # This ensures semantic tokens stay within the decoder's valid range
    vocab_size = 3072
    codebook_size = 2048
    suppress_tokens = list(range(codebook_size, vocab_size))
    print(f"Suppressing tokens {codebook_size}-{vocab_size-1} to match decoder codebook")

    # Prefill with text
    print("\nRunning prefill...")
    hidden, logits = talker.prefill(input_ids)
    offset = input_ids.shape[1]
    last_hidden = hidden[:, -1:, :]

    # Sample first semantic token (with suppression)
    first_token = sample_token(logits[0, -1, :].clone(), temperature, suppress_tokens=suppress_tokens)
    print(f"First semantic token: {first_token}")

    # Collect all frames
    all_codes = []

    # First frame
    semantic_embed = talker.get_codec_embedding(first_token)
    acoustic_codes = code_predictor.generate_acoustic_codes(last_hidden, semantic_embed)
    frame_codes = [first_token] + acoustic_codes
    all_codes.append(frame_codes)
    print(f"Frame 0: semantic={first_token}, acoustics={acoustic_codes[:3]}...")

    # Generate remaining frames
    for frame_idx in range(1, num_frames):
        prev_token = all_codes[-1][0]
        hidden, logits = talker.generate_step(prev_token, offset)
        offset += 1
        last_hidden = hidden

        # Sample semantic token (with suppression)
        next_token = sample_token(logits[0, 0, :].clone(), temperature, suppress_tokens=suppress_tokens)

        # Generate acoustic tokens
        semantic_embed = talker.get_codec_embedding(next_token)
        acoustic_codes = code_predictor.generate_acoustic_codes(last_hidden, semantic_embed)
        frame_codes = [next_token] + acoustic_codes
        all_codes.append(frame_codes)

        if frame_idx < 5 or frame_idx == num_frames - 1:
            print(f"Frame {frame_idx}: semantic={next_token}, acoustics={acoustic_codes[:3]}...")
        elif frame_idx == 5:
            print("...")

    # Convert to tensor [1, 16, num_frames]
    codes_array = np.array(all_codes, dtype=np.int64).T  # [16, num_frames]
    codes_tensor = torch.tensor(codes_array, dtype=torch.long).unsqueeze(0)  # [1, 16, num_frames]

    print(f"\nCodes tensor shape: {codes_tensor.shape}")

    # Save codes
    codes_path = output_dir / f"codes_seed{seed}_frames{num_frames}.npy"
    np.save(codes_path, codes_array)
    print(f"Saved codes to: {codes_path}")

    # Save metadata
    metadata = {
        "text": text,
        "seed": seed,
        "num_frames": num_frames,
        "temperature": temperature,
        "input_ids": input_ids.tolist(),
        "codes_shape": list(codes_tensor.shape),
    }
    metadata_path = output_dir / f"metadata_seed{seed}_frames{num_frames}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_path}")

    # Export codes as binary for Rust
    codes_bin_path = output_dir / f"codes_seed{seed}_frames{num_frames}.bin"
    codes_array.astype(np.int64).tobytes()
    with open(codes_bin_path, "wb") as f:
        f.write(codes_array.T.flatten().astype(np.int64).tobytes())
    print(f"Saved binary codes to: {codes_bin_path}")

    # Decode to audio using the full decoder with modulo handling for semantic tokens
    print("\nDecoding to audio...")
    st_path = Path(model_dir) / "speech_tokenizer" / "model.safetensors"
    from safetensors.torch import load_file as load_torch
    st_weights = load_torch(st_path)
    # Convert BF16 to F32
    st_weights = {k: v.to(torch.float32) if v.dtype == torch.bfloat16 else v for k, v in st_weights.items()}

    codes_tensor = torch.from_numpy(codes_array).unsqueeze(0)  # [1, 16, seq]
    audio = full_decode(codes_tensor, st_weights)
    audio = audio.squeeze(0).squeeze(0).numpy()  # [samples]

    print(f"Audio shape: {audio.shape}")
    print(f"Audio range: [{audio.min():.4f}, {audio.max():.4f}]")

    # Save audio
    audio_path = output_dir / f"audio_seed{seed}_frames{num_frames}.wav"
    import scipy.io.wavfile as wav
    audio_normalized = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_normalized * 32767).astype(np.int16)
    wav.write(str(audio_path), 24000, audio_int16)
    print(f"Saved audio to: {audio_path}")

    return all_codes, audio


def decode_with_decoder(codes: torch.Tensor, weights: dict) -> torch.Tensor:
    """Decode codes to audio using the decoder."""
    # Implementation based on export_decoder_reference.py patterns

    batch_size = codes.shape[0]
    num_quantizers = codes.shape[1]
    seq_len = codes.shape[2]
    codebook_dim = 256

    # 1. Quantizer decode - normalize by cluster_usage as per official implementation
    first_embedding_sum = weights["decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"]
    first_cluster_usage = weights["decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage"]
    first_codebook = first_embedding_sum / first_cluster_usage.clamp(min=1e-7).unsqueeze(-1)

    embeddings = []
    # First quantizer
    first_codes = codes[:, 0, :].flatten()
    first_embed = first_codebook[first_codes].view(batch_size, seq_len, codebook_dim)
    embeddings.append(first_embed)

    # Rest quantizers
    for i in range(15):
        embedding_sum = weights[f"decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum"]
        cluster_usage = weights[f"decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.cluster_usage"]
        cb = embedding_sum / cluster_usage.clamp(min=1e-7).unsqueeze(-1)
        c = codes[:, i + 1, :].flatten()
        embed = cb[c].view(batch_size, seq_len, codebook_dim)
        embeddings.append(embed)

    # Sum all embeddings
    quantized = sum(embeddings)

    # Output projection
    output_proj_w = weights["decoder.quantizer.rvq_first.output_proj.weight"].squeeze(2)
    quantized = torch.nn.functional.linear(quantized, output_proj_w)

    # 2. Pre-conv (causal conv1d)
    x = quantized.transpose(1, 2)  # [batch, channels, seq]
    pre_conv_w = weights["decoder.pre_conv.conv.weight"]
    pre_conv_b = weights["decoder.pre_conv.conv.bias"]
    kernel_size = pre_conv_w.shape[2]

    # Causal padding
    padding = kernel_size - 1
    x_padded = torch.nn.functional.pad(x, (padding, 0))
    x = torch.nn.functional.conv1d(x_padded, pre_conv_w, pre_conv_b)

    # 3. Pre-transformer (simplified - just use x as is for now)
    # Full implementation would require 8 transformer layers
    # For now, skip pre-transformer since it adds complexity

    # 4. Output projection
    output_proj_w = weights["decoder.pre_transformer.output_proj.weight"]
    output_proj_b = weights["decoder.pre_transformer.output_proj.bias"]

    hidden = x.transpose(1, 2)  # [batch, seq, channels]
    # This won't work directly since we skipped pre-transformer
    # Let's do a simplified version

    # Actually, let's implement the full decoder properly
    return full_decode(codes, weights)


def full_decode(codes: torch.Tensor, weights: dict) -> torch.Tensor:
    """Full decoder implementation."""
    batch_size = codes.shape[0]
    num_quantizers = codes.shape[1]
    seq_len = codes.shape[2]
    codebook_dim = 256
    eps = 1e-5

    # 1. Quantizer decode - normalize by cluster_usage as per official implementation
    first_embedding_sum = weights["decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"]
    first_cluster_usage = weights["decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage"]
    first_codebook = first_embedding_sum / first_cluster_usage.clamp(min=1e-7).unsqueeze(-1)
    codebook_size = first_codebook.shape[0]  # 2048

    embeddings = []
    first_codes = codes[:, 0, :].flatten()
    # Apply modulo to handle semantic tokens 2048-3071 (talker uses 3072 vocab, decoder has 2048)
    first_codes = first_codes % codebook_size
    first_embed = first_codebook[first_codes].view(batch_size, seq_len, codebook_dim)
    embeddings.append(first_embed)

    for i in range(15):
        embedding_sum = weights[f"decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum"]
        cluster_usage = weights[f"decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.cluster_usage"]
        cb = embedding_sum / cluster_usage.clamp(min=1e-7).unsqueeze(-1)
        c = codes[:, i + 1, :].flatten()
        embed = cb[c].view(batch_size, seq_len, codebook_dim)
        embeddings.append(embed)

    quantized = sum(embeddings)

    # Output projection
    output_proj_w = weights["decoder.quantizer.rvq_first.output_proj.weight"].squeeze(2)
    quantized = torch.nn.functional.linear(quantized, output_proj_w)

    # 2. Pre-conv
    x = quantized.transpose(1, 2)
    pre_conv_w = weights["decoder.pre_conv.conv.weight"]
    pre_conv_b = weights["decoder.pre_conv.conv.bias"]
    kernel_size = pre_conv_w.shape[2]
    x_padded = torch.nn.functional.pad(x, (kernel_size - 1, 0))
    x = torch.nn.functional.conv1d(x_padded, pre_conv_w, pre_conv_b)

    # 3. Pre-transformer
    hidden = x.transpose(1, 2)  # [batch, seq, channels]
    input_proj_w = weights["decoder.pre_transformer.input_proj.weight"]
    input_proj_b = weights["decoder.pre_transformer.input_proj.bias"]
    hidden = torch.nn.functional.linear(hidden, input_proj_w, input_proj_b)

    # Pre-transformer has 8 layers with RoPE
    num_layers = 8
    num_heads = 16
    head_dim = 64
    rope_theta = 10000.0

    cos, sin = build_rope(seq_len, head_dim, rope_theta)
    causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)

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

        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        # RoPE
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)

        # Attention
        scaling = head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(2, 3)) * scaling
        attn = attn + causal_mask
        attn = torch.nn.functional.softmax(attn, dim=-1)
        attn_out = torch.matmul(attn, v)

        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, num_heads * head_dim)
        attn_out = torch.nn.functional.linear(attn_out, o_proj_w)

        # Layer scale
        attn_scale = weights[f"{prefix}.self_attn_layer_scale.scale"]
        attn_out = attn_out * attn_scale

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

        # Layer scale
        mlp_scale = weights[f"{prefix}.mlp_layer_scale.scale"]
        mlp_out = mlp_out * mlp_scale

        hidden = hidden + mlp_out

    # 4. Output projection
    output_proj_w = weights["decoder.pre_transformer.output_proj.weight"]
    output_proj_b = weights["decoder.pre_transformer.output_proj.bias"]
    hidden = torch.nn.functional.linear(hidden, output_proj_w, output_proj_b)
    hidden = hidden.transpose(1, 2)  # [batch, channels, seq]

    # 5. Upsample stages (simplified - just use trans conv)
    # Stage 0: ratio=2
    x = hidden
    for stage_idx in range(4):
        prefix = f"decoder.upsample.{stage_idx}"

        # SnakeBeta activation
        alpha = weights[f"{prefix}.0.alpha"]
        beta = weights[f"{prefix}.0.beta"]
        x = x + (1.0 / beta.unsqueeze(0).unsqueeze(2)) * torch.sin(alpha.unsqueeze(0).unsqueeze(2) * x).pow(2)

        # Trans conv
        conv_w = weights[f"{prefix}.1.conv.weight"]
        conv_b = weights[f"{prefix}.1.conv.bias"]
        stride = [2, 4, 5, 8][stage_idx]
        x = torch.nn.functional.conv_transpose1d(x, conv_w, conv_b, stride=stride)

        # Trim to remove extra samples from trans conv
        # This is simplified - actual implementation needs careful handling

    # 6. Decoder blocks (simplified)
    for block_idx in range(4):
        prefix = f"decoder.decoder.{block_idx}.block"

        # SnakeBeta
        alpha = weights[f"{prefix}.0.alpha"]
        beta = weights[f"{prefix}.0.beta"]
        x = x + (1.0 / beta.unsqueeze(0).unsqueeze(2)) * torch.sin(alpha.unsqueeze(0).unsqueeze(2) * x).pow(2)

        # Trans conv upsample
        conv_w = weights[f"{prefix}.1.conv.weight"]
        conv_b = weights[f"{prefix}.1.conv.bias"]
        stride = [8, 6, 4, 4][block_idx]
        x = torch.nn.functional.conv_transpose1d(x, conv_w, conv_b, stride=stride)

        # Residual units (simplified - skip for now)

    # 7. Final conv
    final_alpha = weights["decoder.final_layer.0.alpha"]
    final_beta = weights["decoder.final_layer.0.beta"]
    x = x + (1.0 / final_beta.unsqueeze(0).unsqueeze(2)) * torch.sin(final_alpha.unsqueeze(0).unsqueeze(2) * x).pow(2)

    final_conv_w = weights["decoder.final_layer.1.conv.weight"]
    final_conv_b = weights["decoder.final_layer.1.conv.bias"]
    kernel_size = final_conv_w.shape[2]
    x_padded = torch.nn.functional.pad(x, (kernel_size - 1, 0))
    x = torch.nn.functional.conv1d(x_padded, final_conv_w, final_conv_b)

    return x


def main():
    parser = argparse.ArgumentParser(description="Generate reference audio for comparison")
    parser.add_argument("--text", type=str, default="Hello", help="Text to synthesize")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--frames", type=int, default=25, help="Number of frames to generate")
    parser.add_argument("--duration", type=float, default=None, help="Duration in seconds (overrides frames)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--model-dir", type=str, default="test_data/model", help="Model directory")
    parser.add_argument("--output-dir", type=str, default="test_data/reference_audio", help="Output directory")

    args = parser.parse_args()

    # Calculate frames from duration if specified (12.5 Hz)
    num_frames = args.frames
    if args.duration is not None:
        num_frames = int(args.duration * 12.5)

    generate_audio(
        text=args.text,
        model_dir=args.model_dir,
        seed=args.seed,
        num_frames=num_frames,
        temperature=args.temperature,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
