#!/usr/bin/env python3
"""Trace what the official generate produces."""

import torch
from pathlib import Path
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration

def main():
    model_dir = Path("/home/trevor/Projects/Qwen3-TTS/qwen3-tts-rs/test_data/model_customvoice")

    print("Loading official model...")
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        str(model_dir),
        dtype=torch.float32,
    )
    model.eval()
    print("Official model loaded!")

    # Get special tokens
    config = model.config
    talker_config = config.talker_config

    # Construct input_ids as generate expects
    # Format: [im_start, assistant, newline, first_text, ...remaining_text..., im_end, newline, im_start, assistant, newline]
    # The [-5:] is for the generation prompt

    im_start = config.im_start_token_id  # 151644
    im_end = config.im_end_token_id  # 151645
    assistant = config.assistant_token_id  # 77091
    newline = 198

    # Text: "Hello"
    hello_token = 9707

    # The format from generate is: input_id[:, :3] = role prefix, input_id[:, 3:4] = first text
    # We need at least 8+ tokens to have input_id[:, 4:-5] work correctly
    # Let's construct: [im_start, assistant, newline, hello, im_end, newline, im_start, assistant, newline]
    # That's 9 tokens, so [:3] = prefix, [3:4] = hello, [4:-5] = empty, [-5:] = end markers

    # Actually looking at the code more, it seems like input_id is processed differently
    # Let me construct based on what the code expects:
    # - input_id[:, :3] = [im_start, assistant, newline] (role prefix)
    # - input_id[:, 3:4] = [first_text_token]
    # - input_id[:, 4:-5] = [remaining text tokens]
    # - input_id[:, -5:] = some end markers

    # For just "Hello", input might be:
    # [im_start, assistant, newline, hello, im_end, newline, im_start, assistant, newline]
    # = [151644, 77091, 198, 9707, 151645, 198, 151644, 77091, 198]

    input_ids = torch.tensor([[
        im_start, assistant, newline,  # [:3] role prefix
        hello_token,                    # [3:4] first text token
        # Nothing for [4:-5]
        im_end, newline, im_start, assistant, newline  # [-5:] generation prompt
    ]], dtype=torch.long)

    print(f"Input IDs: {input_ids.tolist()}")
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"[:3] = {input_ids[0, :3].tolist()}")
    print(f"[3:4] = {input_ids[0, 3:4].tolist()}")
    print(f"[4:-5] = {input_ids[0, 4:-5].tolist()}")
    print(f"[-5:] = {input_ids[0, -5:].tolist()}")

    # Now let's trace what generate produces
    # We can do this by calling generate with minimal output and checking the inputs

    with torch.no_grad():
        # Construct talker_input_embeds like the generate function does
        talker = model.talker

        text_embed = talker.model.text_embedding.weight
        codec_embed = talker.model.codec_embedding.weight

        # Role prefix
        role_prefix_embed = talker.text_projection(
            text_embed[input_ids[0, :3]].unsqueeze(0)
        )
        print(f"\nRole prefix embed shape: {role_prefix_embed.shape}")
        print(f"Role prefix embed[-1,:5]: {role_prefix_embed[0, -1, :5].tolist()}")

        # TTS special embeds
        tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(
            text_embed[[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]].unsqueeze(0)
        ).chunk(3, dim=1)

        # Language and speaker
        language_id = talker_config.codec_language_id["english"]
        speaker_id = talker_config.spk_id["ryan"]

        # Codec prefix
        codec_prefill_list = [[
            talker_config.codec_think_id,
            talker_config.codec_think_bos_id,
            language_id,
            talker_config.codec_think_eos_id,
        ]]
        codec_input_emebdding_0 = codec_embed[codec_prefill_list[0]].unsqueeze(0)

        codec_input_emebdding_1 = codec_embed[[talker_config.codec_pad_id, talker_config.codec_bos_id]].unsqueeze(0)

        speaker_embed = codec_embed[speaker_id].unsqueeze(0).unsqueeze(0)

        codec_input_emebdding = torch.cat([
            codec_input_emebdding_0,
            speaker_embed,
            codec_input_emebdding_1
        ], dim=1)
        print(f"Codec input embedding shape: {codec_input_emebdding.shape}")

        # _talker_input_embed
        _talker_input_embed = torch.cat((
            tts_pad_embed.expand(-1, codec_input_emebdding.shape[1] - 2, -1),
            tts_bos_embed,
        ), dim=1) + codec_input_emebdding[:, :-1]
        print(f"_talker_input_embed shape: {_talker_input_embed.shape}")

        # Combine with role prefix
        talker_input_embed = torch.cat([role_prefix_embed, _talker_input_embed], dim=1)
        print(f"talker_input_embed (before first text) shape: {talker_input_embed.shape}")

        # Add first text token with codec_bos
        first_text_embed = talker.text_projection(
            text_embed[input_ids[0, 3:4]].unsqueeze(0)
        )
        talker_input_embed = torch.cat([
            talker_input_embed,
            first_text_embed + codec_input_emebdding[:, -1:]
        ], dim=1)
        print(f"talker_input_embed (final) shape: {talker_input_embed.shape}")
        print(f"talker_input_embed[-1,:5]: {talker_input_embed[0, -1, :5].tolist()}")

        # Run through model
        batch, seq_len = talker_input_embed.shape[:2]
        pos_1d = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        position_ids = pos_1d.unsqueeze(0).expand(3, -1, -1)

        # Use the model's forward to get logits
        outputs = talker.model(
            inputs_embeds=talker_input_embed,
            position_ids=position_ids,
            use_cache=False,
        )
        hidden_states = outputs.last_hidden_state
        hidden_states = talker.model.norm(hidden_states)

        # Codec head
        logits = talker.codec_head(hidden_states[:, -1:])
        print(f"\nLogits max: {logits[0, -1].max().item():.4f} at {logits[0, -1].argmax().item()}")

        # After suppression
        vocab_size = talker_config.vocab_size
        codec_eos_id = talker_config.codec_eos_token_id
        suppress_start = vocab_size - 1024
        suppress_tokens = [i for i in range(suppress_start, vocab_size) if i != codec_eos_id]
        logits_suppressed = logits.clone()
        logits_suppressed[:, :, suppress_tokens] = float('-inf')
        print(f"After suppression max: {logits_suppressed[0, -1].max().item():.4f} at {logits_suppressed[0, -1].argmax().item()}")


if __name__ == "__main__":
    main()
