#!/usr/bin/env python3
"""Compare our embedding construction with official."""

import json
import torch
from pathlib import Path
from safetensors.torch import load_file
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def resize_mlp(x: torch.Tensor, weights: dict, prefix: str) -> torch.Tensor:
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


def main():
    model_dir = Path("/home/trevor/Projects/Qwen3-TTS/qwen3-tts-rs/test_data/model_customvoice")

    # Load official model
    print("Loading official model...")
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        str(model_dir),
        dtype=torch.float32,
    )
    model.eval()

    # Load our weights
    print("Loading our weights...")
    weights = load_file(model_dir / "model.safetensors")
    weights = {k: v.float() for k, v in weights.items()}

    with open(model_dir / "config.json") as f:
        config_dict = json.load(f)

    config = model.config
    talker_config = config.talker_config
    talker = model.talker

    # Token IDs
    im_start = config.im_start_token_id
    assistant = config.assistant_token_id
    newline = 198
    hello_token = 9707

    # === Official construction ===
    with torch.no_grad():
        text_embed_official = talker.model.text_embedding.weight
        codec_embed_official = talker.model.codec_embedding.weight

        # Role prefix (official)
        role_prefix_official = talker.text_projection(
            text_embed_official[[im_start, assistant, newline]].unsqueeze(0)
        )
        print(f"Official role_prefix[-1,:5]: {role_prefix_official[0, -1, :5].tolist()}")

        # TTS special (official)
        tts_special_official = talker.text_projection(
            text_embed_official[[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]].unsqueeze(0)
        )
        tts_bos_official, tts_eos_official, tts_pad_official = tts_special_official.chunk(3, dim=1)
        print(f"Official tts_pad[:,:5]: {tts_pad_official[0, 0, :5].tolist()}")

    # === Our construction ===
    text_embed_ours = weights["talker.model.text_embedding.weight"]
    codec_embed_ours = weights["talker.model.codec_embedding.weight"]

    # Role prefix (ours)
    role_prefix_ours = resize_mlp(
        text_embed_ours[[im_start, assistant, newline]].unsqueeze(0),
        weights, "talker.text_projection"
    )
    print(f"Our role_prefix[-1,:5]: {role_prefix_ours[0, -1, :5].tolist()}")

    # TTS special (ours)
    tts_special_ours = resize_mlp(
        text_embed_ours[[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]].unsqueeze(0),
        weights, "talker.text_projection"
    )
    tts_bos_ours = tts_special_ours[:, 0:1, :]
    tts_eos_ours = tts_special_ours[:, 1:2, :]
    tts_pad_ours = tts_special_ours[:, 2:3, :]
    print(f"Our tts_pad[:,:5]: {tts_pad_ours[0, 0, :5].tolist()}")

    # Compare embeddings
    print("\n=== Checking if embeddings match ===")
    role_diff = (role_prefix_official - role_prefix_ours).abs().max().item()
    print(f"Role prefix max diff: {role_diff}")

    tts_pad_diff = (tts_pad_official - tts_pad_ours).abs().max().item()
    print(f"TTS pad max diff: {tts_pad_diff}")

    # === Compare codec embeddings ===
    print("\n=== Codec embeddings ===")
    language_id = talker_config.codec_language_id["english"]
    speaker_id = talker_config.spk_id["ryan"]

    codec_prefix_ids = [
        talker_config.codec_think_id,
        talker_config.codec_think_bos_id,
        language_id,
        talker_config.codec_think_eos_id,
    ]

    with torch.no_grad():
        codec_prefix_official = codec_embed_official[codec_prefix_ids].unsqueeze(0)
        speaker_official = codec_embed_official[speaker_id].unsqueeze(0).unsqueeze(0)
        codec_pad_bos_official = codec_embed_official[[talker_config.codec_pad_id, talker_config.codec_bos_id]].unsqueeze(0)

    codec_prefix_ours = codec_embed_ours[codec_prefix_ids].unsqueeze(0)
    speaker_ours = codec_embed_ours[speaker_id].unsqueeze(0).unsqueeze(0)
    codec_pad_bos_ours = codec_embed_ours[[talker_config.codec_pad_id, talker_config.codec_bos_id]].unsqueeze(0)

    codec_prefix_diff = (codec_prefix_official - codec_prefix_ours).abs().max().item()
    speaker_diff = (speaker_official - speaker_ours).abs().max().item()
    print(f"Codec prefix max diff: {codec_prefix_diff}")
    print(f"Speaker max diff: {speaker_diff}")

    # === Full construction ===
    print("\n=== Full construction comparison ===")

    with torch.no_grad():
        # Official
        codec_input_official = torch.cat([
            codec_prefix_official,
            speaker_official,
            codec_pad_bos_official
        ], dim=1)

        _talker_input_official = torch.cat((
            tts_pad_official.expand(-1, codec_input_official.shape[1] - 2, -1),
            tts_bos_official,
        ), dim=1) + codec_input_official[:, :-1]

        talker_input_official = torch.cat([role_prefix_official, _talker_input_official], dim=1)

        first_text_official = talker.text_projection(
            text_embed_official[hello_token].unsqueeze(0).unsqueeze(0)
        )
        talker_input_official = torch.cat([
            talker_input_official,
            first_text_official + codec_input_official[:, -1:]
        ], dim=1)

        print(f"Official final shape: {talker_input_official.shape}")
        print(f"Official final[-1,:5]: {talker_input_official[0, -1, :5].tolist()}")

    # Ours
    codec_input_ours = torch.cat([
        codec_prefix_ours,
        speaker_ours,
        codec_pad_bos_ours
    ], dim=1)

    _talker_input_ours = torch.cat((
        tts_pad_ours.expand(-1, codec_input_ours.shape[1] - 2, -1),
        tts_bos_ours,
    ), dim=1) + codec_input_ours[:, :-1]

    talker_input_ours = torch.cat([role_prefix_ours, _talker_input_ours], dim=1)

    first_text_ours = resize_mlp(
        text_embed_ours[hello_token].unsqueeze(0).unsqueeze(0),
        weights, "talker.text_projection"
    )
    talker_input_ours = torch.cat([
        talker_input_ours,
        first_text_ours + codec_input_ours[:, -1:]
    ], dim=1)

    print(f"Our final shape: {talker_input_ours.shape}")
    print(f"Our final[-1,:5]: {talker_input_ours[0, -1, :5].tolist()}")

    # Final comparison
    final_diff = (talker_input_official - talker_input_ours).abs().max().item()
    print(f"\nFinal embedding max diff: {final_diff}")

    if final_diff < 1e-5:
        print("✓ Embeddings match!")
    else:
        print("✗ Embeddings differ!")
        # Find where they differ
        for i in range(talker_input_official.shape[1]):
            diff = (talker_input_official[0, i] - talker_input_ours[0, i]).abs().max().item()
            if diff > 1e-5:
                print(f"  Position {i} max diff: {diff}")
                print(f"    Official: {talker_input_official[0, i, :5].tolist()}")
                print(f"    Ours: {talker_input_ours[0, i, :5].tolist()}")


if __name__ == "__main__":
    main()
