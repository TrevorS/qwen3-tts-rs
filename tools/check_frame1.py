#!/usr/bin/env python3
"""Check frame 1 logits to understand token differences."""

import torch
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration

model = Qwen3TTSForConditionalGeneration.from_pretrained("../test_data/model_customvoice", dtype=torch.float32)
model.eval()

config = model.config
talker_config = config.talker_config

talker = model.talker
text_embed = talker.model.text_embedding.weight
codec_embed = talker.model.codec_embedding.weight

im_start = config.im_start_token_id
assistant = config.assistant_token_id
newline = 198

role_prefix = talker.text_projection(text_embed[[im_start, assistant, newline]].unsqueeze(0))
tts_bos, tts_eos, tts_pad = talker.text_projection(
    text_embed[[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]].unsqueeze(0)
).chunk(3, dim=1)

lang_id = talker_config.codec_language_id["english"]
spk_id = talker_config.spk_id["ryan"]
codec_tokens = [
    talker_config.codec_think_id, talker_config.codec_think_bos_id, lang_id,
    talker_config.codec_think_eos_id, spk_id, talker_config.codec_pad_id, talker_config.codec_bos_id,
]
codec_input = codec_embed[codec_tokens].unsqueeze(0)

tts_text = torch.cat([tts_pad.expand(-1, 5, -1), tts_bos], dim=1)
codec_hidden = tts_text + codec_input[:, :6]
talker_input = torch.cat([role_prefix, codec_hidden], dim=1)
first_text = talker.text_projection(text_embed[[9707]].unsqueeze(0))
talker_input = torch.cat([talker_input, first_text + codec_input[:, 6:7]], dim=1)

batch, seq_len = talker_input.shape[:2]
pos_1d = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
position_ids = pos_1d.unsqueeze(0).expand(3, -1, -1)

with torch.no_grad():
    outputs = talker.model(inputs_embeds=talker_input, position_ids=position_ids, use_cache=True)
    hidden = outputs.last_hidden_state
    past_kv = outputs.past_key_values

    logits = talker.codec_head(hidden[:, -1:])
    suppress_start = talker_config.vocab_size - 1024
    suppress_tokens = [i for i in range(suppress_start, talker_config.vocab_size) if i != talker_config.codec_eos_token_id]
    logits[:, :, suppress_tokens] = float("-inf")

    first_token = logits[0, 0].argmax().item()
    print(f"First token (greedy): {first_token}")

    # Generate frame 1
    input_embed = codec_embed[[first_token]].unsqueeze(0)
    pos_ids = torch.tensor([[[10]]]).expand(3, -1, -1)

    outputs = talker.model(inputs_embeds=input_embed, position_ids=pos_ids, past_key_values=past_kv, use_cache=True)
    hidden = outputs.last_hidden_state

    logits = talker.codec_head(hidden[:, -1:])
    logits[:, :, suppress_tokens] = float("-inf")

    top5 = logits[0, 0].topk(5)
    print(f"Frame 1 top 5: {list(zip(top5.indices.tolist(), [f'{v:.4f}' for v in top5.values.tolist()]))}")

    # Check specific tokens
    print(f"Logit for 210: {logits[0, 0, 210].item():.4f}")
    print(f"Logit for 415: {logits[0, 0, 415].item():.4f}")
