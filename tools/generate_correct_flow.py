#!/usr/bin/env python3
"""Generate audio using the correct official flow.

Key insight: During generation, the input to talker model is:
   sum(semantic_embed + all_acoustic_embeds) + trailing_text_hidden

This is different from just using semantic_embed.
"""

import torch
import argparse
import scipy.io.wavfile as wav
import numpy as np

from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="Hello", help="Text to synthesize")
    parser.add_argument("--speaker", default="ryan")
    parser.add_argument("--language", default="english")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--model-dir", default="../test_data/model_customvoice")
    parser.add_argument("--output", default="output_correct_flow.wav")
    args = parser.parse_args()

    print(f"=== Python TTS - Correct Official Flow ===")
    print(f"Text: {args.text}")
    print(f"Speaker: {args.speaker}")
    print(f"Frames: {args.frames}")
    print(f"Temperature: {args.temperature}")
    print(f"Seed: {args.seed}")

    torch.manual_seed(args.seed)

    print(f"\nLoading model from {args.model_dir}...")
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        args.model_dir, dtype=torch.float32
    )
    model.eval()

    config = model.config
    talker_config = config.talker_config
    talker = model.talker

    text_map = {
        "Hello": [9707],
        "Hello world": [9707, 1879],
        "Hello, this is a test": [9707, 11, 419, 374, 264, 1273],
    }
    text_tokens = text_map.get(args.text, [9707])
    print(f"Text tokens: {text_tokens}")

    speaker_id = talker_config.spk_id[args.speaker]
    language_id = talker_config.codec_language_id[args.language]

    im_start = config.im_start_token_id
    assistant = config.assistant_token_id
    newline = 198

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

        print(f"Input embed shape: {talker_input_embed.shape}")

        # Build trailing_text_hidden (remaining tokens + tts_eos)
        if len(text_tokens) > 1:
            remaining_tokens = torch.tensor([text_tokens[1:]])
            remaining_embeds = talker.text_projection(text_embed_layer(remaining_tokens))
            trailing_text_hidden = torch.cat([remaining_embeds, tts_eos_embed], dim=1)
        else:
            trailing_text_hidden = tts_eos_embed

        print(f"Trailing text hidden shape: {trailing_text_hidden.shape}")

        # Position IDs
        batch, seq_len = talker_input_embed.shape[:2]
        pos_1d = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        position_ids = pos_1d.unsqueeze(0).expand(3, -1, -1)

        # Token suppression
        suppress_start = talker_config.vocab_size - 1024
        suppress_tokens = [i for i in range(suppress_start, talker_config.vocab_size)
                         if i != talker_config.codec_eos_token_id]
        eos_token_id = talker_config.codec_eos_token_id
        print(f"EOS token ID: {eos_token_id}")

        # === PREFILL ===
        outputs = talker.model(
            inputs_embeds=talker_input_embed,
            position_ids=position_ids,
            use_cache=True,
        )
        hidden_states = outputs.last_hidden_state
        past_kv = outputs.past_key_values
        past_hidden = hidden_states[:, -1:]  # Save for code_predictor

        # Get logits and suppress
        logits = talker.codec_head(hidden_states[:, -1:])
        logits[:, :, suppress_tokens] = float('-inf')

        # Sample first semantic token
        probs = torch.softmax(logits / args.temperature, dim=-1)
        first_token = torch.multinomial(probs.view(-1), 1).item()
        print(f"First semantic token: {first_token}")

        all_codes = []
        generation_step = 0  # Starts at 0 after prefill

        # === GENERATION LOOP ===
        # Each iteration: use prev semantic token to predict next semantic token
        # Also generates acoustic codes for prev frame

        prev_semantic = first_token

        for frame_idx in range(args.frames):
            # Check for EOS
            if prev_semantic == eos_token_id:
                print(f"EOS detected at frame {frame_idx}, stopping generation")
                break

            # Get embedding of previous semantic token
            last_id_hidden = codec_embed_layer(torch.tensor([[prev_semantic]]))

            # Generate acoustic codes using code_predictor
            # Input: [past_hidden, semantic_embed]
            cp_input = torch.cat([past_hidden, last_id_hidden], dim=1)
            cp_outputs = code_predictor.generate(
                inputs_embeds=cp_input,
                max_new_tokens=talker_config.num_code_groups - 1,
                do_sample=True,
                top_k=50,
                top_p=1.0,
                temperature=0.9,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            acoustic_codes = cp_outputs.sequences[0].tolist()  # 15 acoustic codes

            # Save this frame's codes
            frame_codes = [prev_semantic] + acoustic_codes
            all_codes.append(frame_codes)

            # Debug: check for out-of-range codes
            max_code = max(frame_codes)
            if max_code >= 2048:
                print(f"WARNING Frame {frame_idx}: max code {max_code} >= 2048!")
                print(f"  semantic={prev_semantic}, acoustics={acoustic_codes}")

            if frame_idx < 5 or frame_idx == args.frames - 1:
                print(f"Frame {frame_idx}: semantic={prev_semantic}, acoustics={acoustic_codes[:3]}...")
            elif frame_idx == 5:
                print("...")

            # If this is the last frame, no need to predict next
            if frame_idx == args.frames - 1:
                break

            # Build codec_hiddens: sum of semantic + all acoustic embeddings
            codec_hiddens = [last_id_hidden]
            for i in range(talker_config.num_code_groups - 1):
                ac_embed = code_predictor.get_input_embeddings()[i](
                    torch.tensor([[acoustic_codes[i]]])
                )
                codec_hiddens.append(ac_embed)
            codec_hiddens = torch.cat(codec_hiddens, dim=1)  # [1, 16, hidden_size]
            inputs_embeds = codec_hiddens.sum(1, keepdim=True)  # [1, 1, hidden_size]

            # Add trailing text
            if generation_step < trailing_text_hidden.shape[1]:
                inputs_embeds = inputs_embeds + trailing_text_hidden[:, generation_step:generation_step+1]
            else:
                inputs_embeds = inputs_embeds + tts_pad_embed

            generation_step += 1

            # Position
            new_pos = seq_len + frame_idx
            pos_ids = torch.tensor([[[new_pos]]]).expand(3, -1, -1)

            # Forward through talker model
            outputs = talker.model(
                inputs_embeds=inputs_embeds,
                position_ids=pos_ids,
                past_key_values=past_kv,
                use_cache=True,
            )
            hidden_states = outputs.last_hidden_state
            past_kv = outputs.past_key_values
            past_hidden = hidden_states[:, -1:]  # Update for next iteration

            # Get logits and sample next semantic token
            logits = talker.codec_head(hidden_states[:, -1:])
            logits[:, :, suppress_tokens] = float('-inf')

            probs = torch.softmax(logits / args.temperature, dim=-1)
            next_token = torch.multinomial(probs.view(-1), 1).item()

            prev_semantic = next_token

        # Decode
        num_frames = len(all_codes)
        codes_tensor = torch.zeros((1, num_frames, 16), dtype=torch.long)
        for f, frame in enumerate(all_codes):
            for q, code in enumerate(frame):
                codes_tensor[0, f, q] = code

        print(f"\nCodes tensor shape: {codes_tensor.shape}")
        print("Decoding to audio...")

        output = model.speech_tokenizer.model.decode(codes_tensor)
        # audio_values is a list of tensors, one per batch
        audio = output.audio_values[0].squeeze().numpy()

        print(f"Audio samples: {len(audio)} ({len(audio)/24000:.2f}s at 24kHz)")
        print(f"Audio range: [{audio.min():.4f}, {audio.max():.4f}]")

        audio_int16 = (audio * 32767).astype(np.int16)
        wav.write(args.output, 24000, audio_int16)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
