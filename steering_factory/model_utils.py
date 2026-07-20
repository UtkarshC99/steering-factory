from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import TargetModelConfig

logger = logging.getLogger(__name__)

_TOKENIZER_COMPAT_PATCHED = False


def _patch_tokenizer_special_tokens_compat():
    """transformers v4.x crashes loading some newer models because their
    tokenizer_config.json stores `extra_special_tokens` as a list (a
    format introduced in v5) while v4's
    PreTrainedTokenizerBase._set_model_specific_special_tokens still calls
    .keys() on it, raising AttributeError before it ever gets to a useful
    version-check message. v5 already handles both formats, so this is a
    no-op there. Idempotent -- safe to call on every model load."""
    global _TOKENIZER_COMPAT_PATCHED
    if _TOKENIZER_COMPAT_PATCHED:
        return
    try:
        from transformers import tokenization_utils_base as tub

        if hasattr(tub.PreTrainedTokenizerBase, "_set_model_specific_special_tokens"):
            original = tub.PreTrainedTokenizerBase._set_model_specific_special_tokens

            def _patched(self, special_tokens=None, *args, **kwargs):
                if isinstance(special_tokens, list):
                    special_tokens = {t: t for t in special_tokens}
                return original(self, special_tokens, *args, **kwargs)

            tub.PreTrainedTokenizerBase._set_model_specific_special_tokens = _patched
            logger.debug("Applied v4/v5 tokenizer_config.json compatibility shim.")
    except Exception as e:  # never let the shim itself break loading
        logger.debug("Tokenizer compat shim not applied: %s", e)
    _TOKENIZER_COMPAT_PATCHED = True

# Attribute paths that house the list of transformer blocks across common
# architectures. We walk these in order and use the first one that
# resolves and is indexable.
COMMON_LAYER_PATHS = [
    "model.layers",          # Llama, Mistral, Qwen2, Gemma/Gemma2, Yi, ...
    "transformer.h",          # GPT-2 family
    "gpt_neox.layers",         # GPT-NeoX, Pythia
    "model.decoder.layers",     # OPT, BLOOM-style decoder wrappers
    "transformer.blocks",        # MPT
]

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _resolve_attr_path(obj, path: str):
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def resolve_layers(model, override_path: Optional[str] = None):
    """Returns (layers_module_list, path_used)."""
    paths = [override_path] if override_path else COMMON_LAYER_PATHS
    for path in paths:
        if not path:
            continue
        candidate = _resolve_attr_path(model, path)
        if candidate is not None and hasattr(candidate, "__len__"):
            return candidate, path
    raise RuntimeError(
        "Could not auto-resolve transformer layer list for this model. "
        "Set target_model.layer_path_override in config.yaml (e.g. "
        "'model.layers') to the dotted attribute path of the ModuleList "
        "of decoder blocks."
    )


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: object
    layers: object
    layer_path: str
    num_layers: int
    hidden_size: int
    device: str


def load_model(cfg: TargetModelConfig) -> LoadedModel:
    _patch_tokenizer_special_tokens_compat()
    dtype = _DTYPE_MAP.get(cfg.dtype, torch.bfloat16)

    quant_kwargs = {}
    if cfg.quantization in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig

        if cfg.quantization == "4bit":
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.name_or_path, trust_remote_code=cfg.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.name_or_path,
        torch_dtype=dtype,
        device_map=cfg.device_map,
        trust_remote_code=cfg.trust_remote_code,
        **quant_kwargs,
    )
    model.eval()

    layers, layer_path = resolve_layers(model, cfg.layer_path_override)
    num_layers = len(layers)
    hidden_size = getattr(model.config, "hidden_size", None) or getattr(
        model.config, "n_embd", None
    )
    device = str(next(model.parameters()).device)

    logger.info(
        "Loaded %s | layers=%d @ %s | hidden_size=%s | device=%s",
        cfg.name_or_path,
        num_layers,
        layer_path,
        hidden_size,
        device,
    )

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        layers=layers,
        layer_path=layer_path,
        num_layers=num_layers,
        hidden_size=hidden_size,
        device=device,
    )


def layer_index_from_fraction(num_layers: int, fraction: float) -> int:
    idx = int(round(fraction * (num_layers - 1)))
    return max(0, min(num_layers - 1, idx))


def format_chat(tokenizer, user_message: str, system: Optional[str] = None) -> str:
    """Best-effort chat formatting using the tokenizer's chat template if
    present, else a plain fallback so this still works on base models."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_message})

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    prefix = f"{system}\n\n" if system else ""
    return f"{prefix}{user_message}\n"
