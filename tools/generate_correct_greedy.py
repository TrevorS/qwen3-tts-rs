#!/usr/bin/env python3
"""Generate audio using greedy decoding (no sampling) for exact comparison."""

import torch
import scipy.io.wavfile as wav
import numpy as np

from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration


def main():
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        "../test_data/model_customvoice", dtype=torch.float32
    )
    model.eval()

    config = model.config
    talker_config = config.talker_config
    talker = model.talker

    text_tokens = [9707]  # "Hello"
    speaker_id = talker_config.spk_id["ryan"]
    language_id = talker_config.codec_language_id["english"]

    im_start = config.im_start_token_id
    assistant = config.assistant_token_id
    newline = 198
    eos_token_id = talker_config.codec_eos_token_id

    with torch.no_grad():
        text_embed_layer = talker.get_text_embeddings()
        codec_embed_layer = talker.get_input_embeddings()
        code_predictor = talker.code_predictor

        # Get TTS special embeddings
        tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(
            text_embed_layer(torch.tensor([[
                config.tts_bos_token_id,
                config.tts_eos_token_id,
                config.tts_pad_token_id
            ]]))
        ).chunk(3, dim=1)

        # Build codec prefill
        codec_prefill = [
            talker_config.codec_think_id,
            talker_config.codec_think_bos_id,
            language_id,
            talker_config.codec_think_eos_id,
        ]
        codec_input_0 = codec_embed_layer(torch.tensor([codec_prefill]))
        codec_input_1 = codec_embed_layer(torch.tensor([[talker_config.codec_pad_id, talker_config.codec_bos_id]]))
        speaker_embed = codec_embed_layer(torch.tensor([speaker_id])).unsqueeze(0)
        codec_input = torch.cat([codec_input_0, speaker_embed, codec_input_1], dim=1)

        # Role prefix
        role_prefix_embed = talker.text_projection(
            text_embed_layer(torch.tensor([[im_start, assistant, newline]]))
        )

        # Codec positions with tts_pad/tts_bos
        num_codec = codec_input.shape[1] - 1
        tts_text = torch.cat([
            tts_pad_embed.expand(-1, num_codec - 1, -1),
            tts_bos_embed
        ], dim=1)
        codec_hidden = tts_text + codec_input[:, :-1]

        # Combine role prefix and codec
        talker_input_embed = torch.cat([role_prefix_embed, codec_hidden], dim=1)

        # Add first text token with codec_bos
        first_text_embed = talker.text_projection(
            text_embed_layer(torch.tensor([text_tokens[:1]]))
        )
        talker_input_embed = torch.cat([
            talker_input_embed,
            first_text_embed + codec_input[:, -1:]
        ], dim=1)

        # Trailing text = tts_eos only for single-token text
        trailing_text_hidden = tts_eos_embed

        print(f"Input embed shape: {talker_input_embed.shape}")
        print(f"Trailing text hidden shape: {trailing_text_hidden.shape}")

        # Position IDs
        batch, seq_len = talker_input_embed.shape[:2]
        pos_1d = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        position_ids = pos_1d.unsqueeze(0).expand(3, -1, -1)

        # Token suppression
        suppress_start = talker_config.vocab_size - 1024
        suppress_tokens = [i for i in range(suppress_start, talker_config.vocab_size)
                         if i != eos_token_id]

        # === PREFILL ===
        outputs = talker.model(
            inputs_embeds=talker_input_embed,
            position_ids=position_ids,
            use_cache=True,
        )
        hidden_states = outputs.last_hidden_state
        past_kv = outputs.past_key_values
        past_hidden = hidden_states[:, -1:]

        # Get logits and suppress, then GREEDY
        logits = talker.codec_head(hidden_states[:, -1:])
        logits[:, :, suppress_tokens] = float('-inf')
        first_token = logits[0, 0].argmax().item()

        print(f"First semantic token (greedy): {first_token}")

        all_codes = []
        generation_step = 0
        prev_semantic = first_token
        max_frames = 50

        for frame_idx in range(max_frames):
            if prev_semantic == eos_token_id:
                print(f"EOS at frame {frame_idx}")
                break

            # Get semantic embedding
            semantic_embed = codec_embed_layer(torch.tensor([[prev_semantic]]))

            # Generate acoustic codes GREEDY
            cp_input = torch.cat([past_hidden, semantic_embed], dim=1)
            cp_outputs = code_predictor(inputs_embeds=cp_input, use_cache=True)
            acoustic_codes = [cp_outputs.logits[0, -1].argmax().item()]
            cp_past = cp_outputs.past_key_values
            cp_gen_steps = cp_outputs.generation_steps

            prev_code = acoustic_codes[0]
            for _ in range(14):
                cp_outputs = code_predictor(
                    input_ids=torch.tensor([[prev_code]]),
                    past_key_values=cp_past,
                    generation_steps=cp_gen_steps,
                    use_cache=True,
                )
                code = cp_outputs.logits[0, -1].argmax().item()
                acoustic_codes.append(code)
                cp_past = cp_outputs.past_key_values
                prev_code = code
                cp_gen_steps = cp_outputs.generation_steps

            # Save
            frame_codes = [prev_semantic] + acoustic_codes
            all_codes.append(frame_codes)
            print(f"Frame {frame_idx}: semantic={prev_semantic}, acoustics={acoustic_codes[:3]}...")

            if frame_idx == max_frames - 1:
                break

            # Build input: sum(semantic + acoustic) + trailing text
            codec_hiddens = [semantic_embed]
            for i in range(15):
                ac_embed = code_predictor.get_input_embeddings()[i](
                    torch.tensor([[acoustic_codes[i]]])
                )
                codec_hiddens.append(ac_embed)
            codec_hiddens = torch.cat(codec_hiddens, dim=1)
            inputs_embeds = codec_hiddens.sum(1, keepdim=True)

            if generation_step < trailing_text_hidden.shape[1]:
                inputs_embeds = inputs_embeds + trailing_text_hidden[:, generation_step:generation_step+1]
            else:
                inputs_embeds = inputs_embeds + tts_pad_embed

            generation_step += 1

            new_pos = seq_len + frame_idx
            pos_ids = torch.tensor([[[new_pos]]]).expand(3, -1, -1)

            outputs = talker.model(
                inputs_embeds=inputs_embeds,
                position_ids=pos_ids,
                past_key_values=past_kv,
                use_cache=True,
            )
            hidden_states = outputs.last_hidden_state
            past_kv = outputs.past_key_values
            past_hidden = hidden_states[:, -1:]

            # GREEDY
            logits = talker.codec_head(hidden_states[:, -1:])
            logits[:, :, suppress_tokens] = float('-inf')
            next_token = logits[0, 0].argmax().item()

            prev_semantic = next_token

        # Decode
        if not all_codes:
            print("No codes")
            return

        num_frames = len(all_codes)
        codes_tensor = torch.zeros((1, num_frames, 16), dtype=torch.long)
        for f, frame in enumerate(all_codes):
            for q, code in enumerate(frame):
                codes_tensor[0, f, q] = code

        print(f"\nCodes tensor shape: {codes_tensor.shape}")

        output = model.speech_tokenizer.model.decode(codes_tensor)
        audio = output.audio_values[0].squeeze().numpy()

        print(f"Audio samples: {len(audio)} ({len(audio)/24000:.2f}s at 24kHz)")

        audio_int16 = (audio * 32767).astype(np.int16)
        wav.write("output_python_greedy.wav", 24000, audio_int16)
        print("Saved to: output_python_greedy.wav")


if __name__ == "__main__":
    main()
