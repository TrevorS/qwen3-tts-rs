#!/usr/bin/env python3
"""Test to understand the trailing_text_hidden mechanism."""

import torch
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration

model = Qwen3TTSForConditionalGeneration.from_pretrained(
    "../test_data/model_customvoice", dtype=torch.float32
)
model.eval()

config = model.config
talker_config = config.talker_config
talker = model.talker

# "Hello" = [9707] is only 1 token
# Let's check what the input_ids structure looks like for a typical call

# According to the code, input_ids contains:
# [im_start, assistant, newline, text_tokens..., im_end, newline, im_start, assistant, newline]
# where position 3 is first text token, and -5 is right before im_end

# For "Hello" = [9707]:
# input_ids[:, 3:4] = first text token (Hello)
# input_ids[:, 4:-5] = remaining text tokens (empty for single word)

print("=== Understanding trailing_text_hidden ===")
print()

# Simulate the input_ids structure
im_start = config.im_start_token_id
im_end = config.im_end_token_id
assistant = config.assistant_token_id
newline = 198

# For "Hello" (single token)
text_tokens_hello = [9707]
input_ids_hello = torch.tensor([[
    im_start, assistant, newline,  # 0,1,2 - role prefix
    *text_tokens_hello,            # 3 - first text
    im_end, newline, im_start, assistant, newline  # -5,-4,-3,-2,-1
]])

print(f"For 'Hello' (1 token):")
print(f"  input_ids shape: {input_ids_hello.shape}")
print(f"  input_ids: {input_ids_hello.tolist()}")
print(f"  First text token ([:, 3:4]): {input_ids_hello[:, 3:4].tolist()}")
print(f"  Remaining text tokens ([:, 4:-5]): {input_ids_hello[:, 4:-5].tolist()}")
print(f"  Remaining count: {input_ids_hello[:, 4:-5].shape[1]}")
print()

# For "Hello, this is a test" = [9707, 11, 419, 374, 264, 1273]
text_tokens_long = [9707, 11, 419, 374, 264, 1273]
input_ids_long = torch.tensor([[
    im_start, assistant, newline,
    *text_tokens_long,
    im_end, newline, im_start, assistant, newline
]])

print(f"For 'Hello, this is a test' (6 tokens):")
print(f"  input_ids shape: {input_ids_long.shape}")
print(f"  First text token ([:, 3:4]): {input_ids_long[:, 3:4].tolist()}")
print(f"  Remaining text tokens ([:, 4:-5]): {input_ids_long[:, 4:-5].tolist()}")
print(f"  Remaining count: {input_ids_long[:, 4:-5].shape[1]}")
print()

# Now test with the actual generation
print("=== Testing actual generation difference ===")
print()

text_embed = talker.get_text_embeddings()
tts_pad_embed = talker.text_projection(
    text_embed(torch.tensor([[config.tts_pad_token_id]]))
)
tts_eos_embed = talker.text_projection(
    text_embed(torch.tensor([[config.tts_eos_token_id]]))
)

# For single token: trailing_text_hidden is just tts_eos_embed
trailing_hello = torch.cat([
    talker.text_projection(text_embed(input_ids_hello[:, 4:-5])),
    tts_eos_embed
], dim=1)
print(f"Trailing text hidden for 'Hello': shape {trailing_hello.shape}")

# For multi token: trailing_text_hidden includes remaining text + tts_eos
trailing_long = torch.cat([
    talker.text_projection(text_embed(input_ids_long[:, 4:-5])),
    tts_eos_embed
], dim=1)
print(f"Trailing text hidden for 'Hello, this is a test': shape {trailing_long.shape}")

print()
print("=== Key insight ===")
print("For 'Hello' (1 token):")
print(f"  - trailing_text_hidden has {trailing_hello.shape[1]} position(s)")
print("  - That position is tts_eos_embed")
print("  - So frame 0 adds tts_eos_embed, frame 1+ adds tts_pad_embed")
print()
print("For 'Hello, this is a test' (6 tokens):")
print(f"  - trailing_text_hidden has {trailing_long.shape[1]} position(s)")
print("  - Positions 0-4 are the remaining 5 text tokens [11, 419, 374, 264, 1273]")
print("  - Position 5 is tts_eos_embed")
print("  - So frames 0-5 each add the corresponding text/eos embed")
print("  - Frames 6+ add tts_pad_embed")
