# Qwen3-TTS Architecture Deep Dive

This document describes the Qwen3-TTS (CustomVoice) architecture in detail, based on analysis of the official Python implementation.

## Table of Contents
1. [Model Overview](#model-overview)
2. [Key Components](#key-components)
3. [Input Text Format](#input-text-format)
4. [Input Embedding Construction](#input-embedding-construction)
5. [Generation Loop](#generation-loop)
6. [Code Predictor](#code-predictor)
7. [Weight Names](#weight-names)

---

## Model Overview

Qwen3-TTS consists of three main components:

1. **Talker Model** (`Qwen3TTSTalkerForConditionalGeneration`)
   - A 28-layer transformer (hidden_size=2048, heads=16, kv_heads=8)
   - Generates semantic tokens autoregressively
   - Contains the code predictor as a submodule

2. **Code Predictor** (`Qwen3TTSTalkerCodePredictorModelForConditionalGeneration`)
   - A 5-layer transformer (hidden_size=1024, heads=16, kv_heads=8)
   - Generates 15 acoustic codes for each semantic token
   - Called DURING talker generation, not after

3. **Decoder** (`Qwen3TTSTokenizerV2Model`)
   - Converts semantic + acoustic codes to audio waveforms
   - 12Hz model: 1 frame = 2000 samples at 24kHz = 83.33ms

---

## Key Components

### Talker Model Structure
```
Qwen3TTSTalkerForConditionalGeneration
├── model (Qwen3TTSTalkerModel)
│   ├── text_embedding: Embedding(151936, 2048)  # For text tokens
│   ├── codec_embedding: Embedding(3072, 2048)   # For codec/speaker tokens
│   ├── layers: 28x DecoderLayer
│   ├── norm: RMSNorm
│   └── rotary_emb: RotaryEmbedding (3D RoPE)
├── text_projection: ResizeMLP(2048 → 2048 → 2048)  # Projects text embeddings
├── codec_head: Linear(2048 → 3072)                  # Predicts semantic tokens
└── code_predictor: CodePredictorForConditionalGeneration
```

### Code Predictor Structure
```
Qwen3TTSTalkerCodePredictorModelForConditionalGeneration
├── model (Qwen3TTSTalkerCodePredictorModel)
│   ├── codec_embedding: ModuleList of 15 Embeddings
│   │   Each: Embedding(2048, 2048)  # embedding_dim = talker_hidden_size
│   ├── layers: 5x DecoderLayer
│   ├── norm: RMSNorm
│   └── rotary_emb: RotaryEmbedding
├── small_to_mtp_projection: Linear(2048 → 1024)  # Projects talker dim → code predictor dim
└── lm_head: ModuleList of 15 Linear layers
    Each: Linear(1024 → 2048)  # Predicts acoustic codes
```

### ResizeMLP Structure (text_projection)
```python
class Qwen3TTSTalkerResizeMLP(nn.Module):
    def __init__(self, input_size, intermediate_size, output_size, act, bias=False):
        self.linear_fc1 = nn.Linear(input_size, intermediate_size, bias=bias)
        self.linear_fc2 = nn.Linear(intermediate_size, output_size, bias=bias)
        self.act_fn = silu  # For hidden_act="silu"

    def forward(self, x):
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))
```

For text_projection: `ResizeMLP(2048, 2048, 2048, "silu", bias=True)`

### Special Token IDs (from config.json)
```
# TTS special tokens (text embedding)
tts_bos_token_id = 151672   # TTS begin of speech
tts_eos_token_id = 151673   # TTS end of speech
tts_pad_token_id = 151671   # TTS padding

# Codec control tokens (codec embedding)
codec_bos_id = 2149         # Codec sequence start
codec_eos_token_id = 2150   # Codec sequence end
codec_pad_id = 2148         # Codec padding
codec_think_id = 2154       # Thinking mode marker
codec_nothink_id = 2155     # No-thinking mode marker
codec_think_bos_id = 2156   # Think block start
codec_think_eos_id = 2157   # Think block end

# Text special tokens
im_start_token_id = 151644  # <|im_start|>
im_end_token_id = 151645    # <|im_end|>
assistant_token_id = 77091  # "assistant"

# Speaker IDs (embedded via codec_embedding)
ryan = 3061
vivian = 3065
serena = 3066
# etc.

# Language IDs
english = 2050
chinese = 2055
# etc.
```

---

## Input Text Format

The input text is formatted as a chat template:
```
<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
```

For example, "Hello" becomes:
```
<|im_start|>assistant\nHello<|im_end|>\n<|im_start|>assistant\n
```

When tokenized (approximate):
```
Position | Token ID | Token
---------|----------|-------
0        | 151644   | <|im_start|>
1        | 77091    | assistant
2        | 198      | \n
3        | 9707     | Hello
4        | 151645   | <|im_end|>
5        | 198      | \n
6        | 151644   | <|im_start|>
7        | 77091    | assistant
8        | 198      | \n
```

The code references these positions:
- `input_id[:, :3]` = role prefix: `[im_start, assistant, \n]`
- `input_id[:, 3:-5]` = text content: `[Hello]`
- `input_id[:, -5:]` = role suffix: `[im_end, \n, im_start, assistant, \n]`

---

## Input Embedding Construction

This is the most complex and critical part. The official code constructs inputs by **ADDING** text embeddings and codec embeddings together.

### Step 1: Prepare TTS Special Embeddings

```python
# Get TTS special token embeddings via text_projection
tts_bos_embed, tts_eos_embed, tts_pad_embed = self.talker.text_projection(
    self.talker.get_text_embeddings()(
        torch.tensor([[tts_bos_token_id, tts_eos_token_id, tts_pad_token_id]])
    )
).chunk(3, dim=1)
# Each is shape [1, 1, 2048]
```

### Step 2: Prepare Role Prefix

```python
# Embed and project the role prefix: "<|im_start|>assistant\n"
_talker_input_embed_role = self.talker.text_projection(
    self.talker.get_text_embeddings()(input_id[:, :3])
)
# Shape: [1, 3, 2048]
```

### Step 3: Build Codec Control Sequence

For specified language (e.g., English):
```python
codec_prefill_list = [[
    codec_think_id,      # 2154: Thinking mode marker
    codec_think_bos_id,  # 2156: Think block start
    language_id,         # e.g., 2050 for English
    codec_think_eos_id,  # 2157: Think block end
]]
# 4 tokens
```

For auto language detection:
```python
codec_prefill_list = [[
    codec_nothink_id,    # 2155: No-thinking mode
    codec_think_bos_id,  # 2156
    codec_think_eos_id,  # 2157
]]
# 3 tokens
```

### Step 4: Build Full Codec Embedding

```python
# Control tokens embedded
codec_input_embedding_0 = codec_embedding(codec_prefill_list)  # [1, 4, 2048]

# Speaker embedding
speaker_embed = codec_embedding([[speaker_id]])  # [1, 1, 2048]

# Pad + BOS tokens
codec_input_embedding_1 = codec_embedding([[codec_pad_id, codec_bos_id]])  # [1, 2, 2048]

# Concatenate: [control_tokens, speaker, pad, bos]
codec_input_embedding = torch.cat([
    codec_input_embedding_0,  # [1, 4, 2048]
    speaker_embed,            # [1, 1, 2048]
    codec_input_embedding_1   # [1, 2, 2048]
], dim=1)
# Total: [1, 7, 2048]
```

### Step 5: Combine Text + Codec (THE KEY INSIGHT!)

```python
# Text side: tts_pad repeated, then tts_bos
_talker_input_embed = torch.cat((
    tts_pad_embed.expand(-1, codec_input_embedding.shape[1] - 2, -1),  # [1, 5, 2048]
    tts_bos_embed,  # [1, 1, 2048]
), dim=1)
# Shape: [1, 6, 2048]

# ADD text + codec (excluding last codec position)
_talker_input_embed = _talker_input_embed + codec_input_embedding[:, :-1]

# Prepend role prefix
talker_input_embed = torch.cat((_talker_input_embed_role, _talker_input_embed), dim=1)
# Shape: [1, 9, 2048]
```

### Step 6: Add First Text Token

```python
# First text token + codec_bos
first_text_embed = self.talker.text_projection(
    self.talker.get_text_embeddings()(input_id[:, 3:4])
) + codec_input_embedding[:, -1:]

talker_input_embed = torch.cat([talker_input_embed, first_text_embed], dim=1)
# Shape: [1, 10, 2048]
```

### Step 7: Prepare Trailing Text

```python
# Remaining text tokens + tts_eos (for streaming mode)
trailing_text_hidden = torch.cat((
    self.talker.text_projection(
        self.talker.get_text_embeddings()(input_id[:, 4:-5])  # Rest of text
    ),
    tts_eos_embed
), dim=1)
# Used during generation to add text to each step
```

### Final Input Sequence Structure (Streaming Mode, English)

```
Position | Text Embedding (text_proj)   | Codec Embedding      | Combined
---------|------------------------------|----------------------|----------
0        | im_start                     | (none)               | text only
1        | assistant                    | (none)               | text only
2        | \n                           | (none)               | text only
3        | tts_pad                      | think_id (2154)      | added
4        | tts_pad                      | think_bos_id (2156)  | added
5        | tts_pad                      | language_id (2050)   | added
6        | tts_pad                      | think_eos_id (2157)  | added
7        | tts_pad                      | speaker_id (3061)    | added
8        | tts_bos                      | pad_id (2148)        | added
9        | first_text ("Hello")         | bos_id (2149)        | added
```

---

## Generation Loop

The generation is NOT just semantic tokens - it interleaves semantic token generation with acoustic code prediction.

### Generation Step (from forward() in TalkerForConditionalGeneration)

```python
# This runs for EACH generated semantic token:

# 1. Get embedding of the just-sampled semantic token
last_id_hidden = self.get_input_embeddings()(input_ids)  # [1, 1, 2048]

# 2. Generate 15 acoustic codes using code predictor
predictor_result = self.code_predictor.generate(
    inputs_embeds=torch.cat((past_hidden, last_id_hidden), dim=1),  # [1, 2, 2048]
    max_new_tokens=15,
    do_sample=subtalker_dosample,
    top_p=subtalker_top_p,
    top_k=subtalker_top_k,
    temperature=subtalker_temperature,
    output_hidden_states=True,
    return_dict_in_generate=True,
)

# 3. Combine semantic token with acoustic codes
codec_ids = torch.cat((input_ids, predictor_result.sequences), dim=-1)
# codec_ids is now [semantic, a1, a2, ..., a15] - 16 total tokens

# 4. Get embeddings for all 16 codes
codec_hiddens = torch.cat(
    [last_id_hidden]  # Semantic token embedding from main codec_embedding
    + [
        self.code_predictor.get_input_embeddings()[i](
            predictor_result.sequences[..., i:i+1]
        ) for i in range(15)
    ],  # Each acoustic code from its respective embedding layer
    dim=1,
)  # [1, 16, 2048]

# 5. SUM all 16 embeddings to get input for next step (Residual VQ pattern)
inputs_embeds = codec_hiddens.sum(1, keepdim=True)  # [1, 1, 2048]

# 6. Add trailing text embedding if available
if generation_step < trailing_text_hidden.shape[1]:
    inputs_embeds = inputs_embeds + trailing_text_hidden[:, generation_step].unsqueeze(1)
else:
    inputs_embeds = inputs_embeds + tts_pad_embed

# 7. Run this combined embedding through the talker to get next hidden state
outputs = self.model(inputs_embeds=inputs_embeds, ...)
hidden_states = outputs.last_hidden_state

# 8. Sample next semantic token
logits = self.codec_head(hidden_states)
next_token = sample(logits)

# 9. Save hidden state for next code predictor call
past_hidden = hidden_states[:, -1:, :]

# Return includes all 16 codes
return Qwen3TTSTalkerOutputWithPast(
    logits=logits,
    hidden_states=(outputs.hidden_states, codec_ids),
    past_hidden=past_hidden,
    ...
)
```

### Key Insight: Residual VQ Pattern

The model uses **Residual Vector Quantization**. For each frame:
1. The semantic token provides the coarse representation
2. Each acoustic code refines the representation
3. All 16 embeddings are **SUMMED** to create the input for the next step

---

## Code Predictor

### Forward Pass

The code predictor generates 15 acoustic codes autoregressively:

```python
def forward(...):
    # Prefill stage (when given [past_hidden, last_id_hidden] = 2 positions)
    if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
        generation_steps = inputs_embeds.shape[1] - 2  # = 0 initially

    # Generation stage (for subsequent tokens)
    else:
        # Get embedding from the appropriate embedding layer
        inputs_embeds = self.model.get_input_embeddings()[generation_steps - 1](input_ids)

    # Project from talker dim (2048) to code predictor dim (1024)
    inputs_embeds = self.small_to_mtp_projection(inputs_embeds)

    # Run through 5 transformer layers
    outputs = self.model(inputs_embeds=inputs_embeds, ...)
    hidden_states = outputs.last_hidden_state

    # Use the appropriate lm_head for this generation step
    logits = self.lm_head[generation_steps](hidden_states)

    return logits
```

### Embedding Layers

The code predictor has **15 separate embedding layers**, one for each acoustic code group:

```python
self.codec_embedding = nn.ModuleList(
    [nn.Embedding(2048, 2048) for _ in range(15)]
)
# vocab_size = 2048 (acoustic codebook size)
# embedding_dim = 2048 (talker hidden size, NOT code predictor hidden size)
```

Important: the embeddings output 2048-dim vectors, which are then projected to 1024 by `small_to_mtp_projection`.

### LM Heads

The code predictor has **15 separate LM heads**, one for each acoustic code group:

```python
self.lm_head = nn.ModuleList(
    [nn.Linear(1024, 2048, bias=False) for _ in range(15)]
)
# 1024 = code predictor hidden size
# 2048 = acoustic codebook size
```

---

## Weight Names

### Main Model Weights
```
talker.model.text_embedding.weight              [151936, 2048]
talker.model.codec_embedding.weight             [3072, 2048]
talker.model.layers.{0-27}.input_layernorm.weight
talker.model.layers.{0-27}.self_attn.{q,k,v,o}_proj.weight
talker.model.layers.{0-27}.post_attention_layernorm.weight
talker.model.layers.{0-27}.mlp.{gate,up,down}_proj.weight
talker.model.norm.weight
talker.text_projection.linear_fc1.{weight,bias} [2048, 2048]
talker.text_projection.linear_fc2.{weight,bias} [2048, 2048]
talker.codec_head.weight                        [3072, 2048]
```

### Code Predictor Weights
```
talker.code_predictor.small_to_mtp_projection.{weight,bias}  [1024, 2048]
talker.code_predictor.model.codec_embedding.{0-14}.weight    [2048, 2048]
talker.code_predictor.model.layers.{0-4}.input_layernorm.weight
talker.code_predictor.model.layers.{0-4}.self_attn.{q,k,v,o}_proj.weight
talker.code_predictor.model.layers.{0-4}.post_attention_layernorm.weight
talker.code_predictor.model.layers.{0-4}.mlp.{gate,up,down}_proj.weight
talker.code_predictor.model.norm.weight
talker.code_predictor.lm_head.{0-14}.weight                  [2048, 1024]
```

---

## Decoder Notes

The decoder expects codes in the shape `[batch, seq_len, 16]` where:
- Column 0: semantic token
- Columns 1-15: acoustic codes from code predictor

**Important**: Semantic tokens >= 2048 are special control tokens and must be filtered out before decoding:
- `codec_eos_token_id = 2150`: End of sequence
- `codec_think_id = 2154`: Thinking mode marker
- `codec_think_bos_id = 2156`: Think block start
- `codec_think_eos_id = 2157`: Think block end

The model may generate thinking tokens (2156, 2157) as part of its internal reasoning. These should be filtered:

```python
semantic_tokens = codes[:, 0]
valid_mask = semantic_tokens < 2048
codes = codes[valid_mask]
```

---

## Summary

1. **Input format**: Text is wrapped in `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`
2. **Text projection**: All text embeddings go through `text_projection` (2-layer MLP with SiLU)
3. **Embedding combination**: Text embeddings and codec embeddings are **ADDED** together
4. **Generation loop**: For each step, generate semantic token → call code predictor → sum all 16 embeddings → add trailing text → continue
5. **Residual VQ**: All 16 code embeddings are summed for the next input
6. **Trailing text**: Added incrementally during generation (streaming mode)
7. **Token filtering**: Semantic tokens >= 2048 are control tokens and must be filtered before decoding
