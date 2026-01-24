#!/usr/bin/env python3
"""Generate audio with proper trailing text handling."""

import torch
import argparse
from pathlib import Path
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
    parser.add_argument("--output", default="output_trailing_text.wav")
    args = parser.parse_args()

    print(f"=== Python CustomVoice TTS with Trailing Text ===")
    print(f"Text: {args.text}")
    print(f"Speaker: {args.speaker}")
    print(f"Language: {args.language}")
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

    # Token mapping
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

        # Prefill
        outputs = talker.model(
            inputs_embeds=talker_input_embed,
            position_ids=position_ids,
            use_cache=True,
        )
        hidden_states = outputs.last_hidden_state
        past_kv = outputs.past_key_values

        # Get logits and suppress
        logits = talker.codec_head(hidden_states[:, -1:])
        suppress_start = talker_config.vocab_size - 1024
        suppress_tokens = [i for i in range(suppress_start, talker_config.vocab_size)
                         if i != talker_config.codec_eos_token_id]
        logits[:, :, suppress_tokens] = float('-inf')

        # Sample first semantic token
        probs = torch.softmax(logits / args.temperature, dim=-1)
        first_token = torch.multinomial(probs.view(-1), 1).item()
        print(f"First semantic token: {first_token}")

        all_codes = []

        # Generate acoustic codes for first frame
        semantic_embed = codec_embed_layer(torch.tensor([[first_token]]))
        cp_input = torch.cat([hidden_states[:, -1:], semantic_embed], dim=1)
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

        frame_codes = [first_token] + acoustic_codes
        all_codes.append(frame_codes)
        print(f"Frame 0: semantic={first_token}, acoustics={frame_codes[1:4]}...")

        # Track generation step for trailing text
        generation_step = 0

        # Generate remaining frames
        for frame_idx in range(1, args.frames):
            prev_token = all_codes[-1][0]

            # Embed previous semantic token
            input_embed = codec_embed_layer(torch.tensor([[prev_token]]))

            # Generate acoustic codes FIRST using code_predictor
            semantic_embed = codec_embed_layer(torch.tensor([[prev_token]]))
            cp_input = torch.cat([hidden_states[:, -1:], semantic_embed], dim=1)
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

            # Sum up the codec embeddings (semantic + all acoustic)
            codec_hiddens = input_embed.clone()
            for i, ac in enumerate(acoustic_codes):
                if i == 0:
                    codec_hiddens = codec_hiddens + codec_embed_layer(torch.tensor([[ac]]))
                else:
                    codec_hiddens = codec_hiddens + code_predictor.get_input_embeddings()[i-1](torch.tensor([[ac]]))

            # ADD TRAILING TEXT HIDDEN - this is the key difference!
            if generation_step < trailing_text_hidden.shape[1]:
                codec_hiddens = codec_hiddens + trailing_text_hidden[:, generation_step:generation_step+1]
            else:
                codec_hiddens = codec_hiddens + tts_pad_embed

            generation_step += 1

            # Position
            new_pos = seq_len + frame_idx - 1
            pos_ids = torch.tensor([[[new_pos]]]).expand(3, -1, -1)

            # Forward through talker
            outputs = talker.model(
                inputs_embeds=codec_hiddens,
                position_ids=pos_ids,
                past_key_values=past_kv,
                use_cache=True,
            )
            hidden_states = outputs.last_hidden_state
            past_kv = outputs.past_key_values

            # Get logits and sample
            logits = talker.codec_head(hidden_states[:, -1:])
            logits[:, :, suppress_tokens] = float('-inf')

            probs = torch.softmax(logits / args.temperature, dim=-1)
            next_token = torch.multinomial(probs.view(-1), 1).item()

            frame_codes = [next_token] + acoustic_codes
            all_codes.append(frame_codes)

            if frame_idx < 5 or frame_idx == args.frames - 1:
                print(f"Frame {frame_idx}: semantic={next_token}, acoustics={frame_codes[1:4]}...")
            elif frame_idx == 5:
                print("...")

        # Decode
        num_frames = len(all_codes)
        codes_tensor = torch.zeros((1, num_frames, 16), dtype=torch.long)
        for f, frame in enumerate(all_codes):
            for q, code in enumerate(frame):
                codes_tensor[0, f, q] = code

        print(f"\nCodes tensor shape: {codes_tensor.shape}")
        print("Decoding to audio...")

        output = model.speech_tokenizer.model.decode(codes_tensor)
        if hasattr(output, 'audio_values'):
            waveform = output.audio_values
        elif isinstance(output, (tuple, list)):
            waveform = output[0]
        else:
            waveform = output
        if hasattr(waveform, 'squeeze'):
            audio = waveform.squeeze().numpy()
        else:
            audio = np.array(waveform).squeeze()

        print(f"Audio samples: {len(audio)} ({len(audio)/24000:.2f}s at 24kHz)")

        audio_int16 = (audio * 32767).astype(np.int16)
        wav.write(args.output, 24000, audio_int16)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
