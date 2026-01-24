#!/usr/bin/env python3
"""Trace what the official CustomVoice model produces."""

import torch
from pathlib import Path
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig

def main():
    model_dir = Path("/home/trevor/Projects/Qwen3-TTS/qwen3-tts-rs/test_data/model_customvoice")

    print("Loading official CustomVoice model...")
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        str(model_dir),
        dtype=torch.float32,
    )
    model.eval()
    print("Official model loaded!")
    print(f"TTS model type: {model.tts_model_type}")
    print(f"Talker hidden size: {model.config.talker_config.hidden_size}")

    config = model.config
    talker_config = config.talker_config

    print(f"\nSpecial tokens:")
    print(f"  tts_bos_token_id: {config.tts_bos_token_id}")
    print(f"  tts_eos_token_id: {config.tts_eos_token_id}")
    print(f"  tts_pad_token_id: {config.tts_pad_token_id}")
    print(f"  codec_eos_token_id: {talker_config.codec_eos_token_id}")
    print(f"  codec_bos_id: {talker_config.codec_bos_id}")
    print(f"  codec_pad_id: {talker_config.codec_pad_id}")
    print(f"  codec_think_id: {talker_config.codec_think_id}")
    print(f"  vocab_size: {talker_config.vocab_size}")

    print(f"\nSpeaker IDs:")
    for spk, sid in talker_config.spk_id.items():
        print(f"  {spk}: {sid}")

    print(f"\nLanguage IDs:")
    for lang, lid in talker_config.codec_language_id.items():
        print(f"  {lang}: {lid}")

    # Build input
    im_start = config.im_start_token_id
    im_end = config.im_end_token_id
    assistant = config.assistant_token_id
    newline = 198
    hello_token = 9707

    # Build ChatML input
    input_ids = torch.tensor([[
        im_start, assistant, newline,  # role prefix
        hello_token,                    # text
        im_end, newline, im_start, assistant, newline  # generation prompt
    ]], dtype=torch.long)

    print(f"\nInput IDs: {input_ids.tolist()}")

    # Trace through talker prefill
    with torch.no_grad():
        talker = model.talker

        text_embed_layer = talker.get_text_embeddings()
        codec_embed_layer = talker.get_input_embeddings()  # This is the codec embedding

        # Get special text tokens
        tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(
            text_embed_layer(
                torch.tensor([[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]])
            )
        ).chunk(3, dim=1)

        print(f"\ntts_pad_embed shape: {tts_pad_embed.shape}")
        print(f"tts_pad_embed[:5]: {tts_pad_embed[0, 0, :5].tolist()}")

        # Build codec embeddings for CustomVoice
        language_id = talker_config.codec_language_id["english"]
        speaker_id = talker_config.spk_id["ryan"]

        print(f"\nLanguage ID (english): {language_id}")
        print(f"Speaker ID (ryan): {speaker_id}")

        codec_prefill = [
            talker_config.codec_think_id,
            talker_config.codec_think_bos_id,
            language_id,
            talker_config.codec_think_eos_id,
        ]

        print(f"Codec prefill: {codec_prefill}")

        codec_input_0 = codec_embed_layer(torch.tensor([codec_prefill]))  # [1, 4, dim]
        codec_input_1 = codec_embed_layer(torch.tensor([[talker_config.codec_pad_id, talker_config.codec_bos_id]]))  # [1, 2, dim]
        speaker_embed = codec_embed_layer(torch.tensor([speaker_id])).unsqueeze(0)  # [1, 1, dim]

        codec_input = torch.cat([codec_input_0, speaker_embed, codec_input_1], dim=1)  # [1, 7, dim]
        print(f"codec_input shape: {codec_input.shape}")

        # Role prefix
        role_prefix_embed = talker.text_projection(
            text_embed_layer(input_ids[:, :3])
        )
        print(f"role_prefix_embed shape: {role_prefix_embed.shape}")

        # Codec positions with tts_pad/bos
        num_codec = codec_input.shape[1] - 1  # 6 positions
        tts_text = torch.cat([
            tts_pad_embed.expand(-1, num_codec - 1, -1),  # 5 tts_pad
            tts_bos_embed
        ], dim=1)

        codec_hidden = tts_text + codec_input[:, :-1]  # Add with first 6 codec tokens
        print(f"codec_hidden shape: {codec_hidden.shape}")

        # Combine role prefix and codec
        talker_input_embed = torch.cat([role_prefix_embed, codec_hidden], dim=1)

        # Add first text token with codec_bos
        first_text_embed = talker.text_projection(
            text_embed_layer(input_ids[:, 3:4])
        )
        talker_input_embed = torch.cat([
            talker_input_embed,
            first_text_embed + codec_input[:, -1:]  # Add codec_bos
        ], dim=1)

        print(f"talker_input_embed shape: {talker_input_embed.shape}")
        print(f"talker_input_embed[-1,:5]: {talker_input_embed[0, -1, :5].tolist()}")

        # Run through model
        batch, seq_len = talker_input_embed.shape[:2]
        pos_1d = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        position_ids = pos_1d.unsqueeze(0).expand(3, -1, -1)

        outputs = talker.model(
            inputs_embeds=talker_input_embed,
            position_ids=position_ids,
            use_cache=False,
        )
        hidden_states = outputs.last_hidden_state
        hidden_states = talker.model.norm(hidden_states)

        print(f"hidden_states shape: {hidden_states.shape}")

        # Codec head
        logits = talker.codec_head(hidden_states[:, -1:])
        print(f"logits shape: {logits.shape}")

        # Get top tokens before suppression
        top_vals, top_indices = logits[0, 0].topk(10)
        print(f"\nTop 10 tokens before suppression:")
        for val, idx in zip(top_vals.tolist(), top_indices.tolist()):
            print(f"  token {idx}: {val:.4f}")

        # Suppress tokens
        suppress_start = talker_config.vocab_size - 1024
        suppress_tokens = [i for i in range(suppress_start, talker_config.vocab_size)
                         if i != talker_config.codec_eos_token_id]
        print(f"\nSuppressing tokens {suppress_start} to {talker_config.vocab_size - 1} (except {talker_config.codec_eos_token_id})")

        logits_suppressed = logits.clone()
        logits_suppressed[:, :, suppress_tokens] = float('-inf')

        # Get top tokens after suppression
        top_vals, top_indices = logits_suppressed[0, 0].topk(10)
        print(f"\nTop 10 tokens after suppression:")
        for val, idx in zip(top_vals.tolist(), top_indices.tolist()):
            print(f"  token {idx}: {val:.4f}")

        # Sample with temperature
        temperature = 0.9
        probs = torch.softmax(logits_suppressed / temperature, dim=-1)

        torch.manual_seed(42)
        first_token = torch.multinomial(probs.view(-1), 1)
        print(f"\nSampled first token (seed=42): {first_token.item()}")


if __name__ == "__main__":
    main()
