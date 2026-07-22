"""A tiny, randomly-initialized (no download) Llama-architecture model +
word-level tokenizer, shared by tests that need to actually call
`generate()`/`model()` (batching/steering correctness can't be verified by
mocking those calls by hand -- that would test the mock, not the code).

Llama architecture is used deliberately, not GPT-2: this codebase's real
target models (Qwen3, Gemma-3) are RoPE + attention-mask-aware, like
Llama, not GPT-2's absolute-position embeddings. That distinction is not
cosmetic for these tests -- a hand-verification during development found
that left-padded batching gives WRONG hidden states on GPT-2 (its default
attention path does not correctly exclude padded-key positions without
extra hand-computed position_ids), while Llama's attention handles the
same left-padded batch correctly out of the box via `attention_mask`. A
GPT-2 fixture would either give false confidence (if it happened to match)
or force irrelevant GPT-2-specific workarounds into production code that
Qwen3/Gemma-3 never need. Using the architecture family this project
actually targets is the point, not an arbitrary choice.
"""
from __future__ import annotations

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from steering_factory.model_utils import LoadedModel

VOCAB = {"<pad>": 0, "<unk>": 1, "<eos>": 2, "<bos>": 3}
for _i in range(4, 60):
    VOCAB[f"w{_i}"] = _i


def build_tiny_tokenizer() -> PreTrainedTokenizerFast:
    tok_model = Tokenizer(models.WordLevel(vocab=dict(VOCAB), unk_token="<unk>"))
    tok_model.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok_model,
        pad_token="<pad>", eos_token="<eos>", bos_token="<bos>", unk_token="<unk>",
    )


def build_loaded_model(seed: int = 0) -> LoadedModel:
    """Builds a small LlamaForCausalLM with deterministic random weights
    (fixed seed) wrapped in the same LoadedModel shape load_model() returns,
    so sweep.py/extraction.py code under test doesn't need to know it isn't
    a real downloaded model."""
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=len(VOCAB), hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=64, pad_token_id=VOCAB["<pad>"],
        eos_token_id=VOCAB["<eos>"], bos_token_id=VOCAB["<bos>"],
    )
    model = LlamaForCausalLM(config)
    model.eval()
    tokenizer = build_tiny_tokenizer()
    layers = model.model.layers
    return LoadedModel(
        model=model, tokenizer=tokenizer, layers=layers, layer_path="model.layers",
        num_layers=len(layers), hidden_size=config.hidden_size, device="cpu",
    )
