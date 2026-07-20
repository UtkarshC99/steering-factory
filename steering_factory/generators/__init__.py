from __future__ import annotations

from typing import Optional

from ..config import GeneratorConfig
from .base import GeneratorBackend


def build_generator(cfg: GeneratorConfig, loaded_model_for_local_hf=None) -> GeneratorBackend:
    """Factory: builds whichever generator backend config.generator.backend
    selects. `loaded_model_for_local_hf` is only needed (and only loaded)
    when backend == 'local_hf' and you want to reuse the already-loaded
    target model rather than loading a second one."""
    if cfg.backend == "anthropic":
        from .anthropic_gen import AnthropicGenerator

        return AnthropicGenerator(model=cfg.anthropic.model)

    if cfg.backend == "openai_compat":
        from .openai_compat_gen import OpenAICompatGenerator

        return OpenAICompatGenerator(
            base_url=cfg.openai_compat.base_url,
            model=cfg.openai_compat.model,
            api_key_env=cfg.openai_compat.api_key_env,
        )

    if cfg.backend == "local_hf":
        from .local_hf_gen import LocalHFGenerator
        from ..model_utils import load_model
        from ..config import TargetModelConfig

        if cfg.local_hf.name_or_path:
            loaded = load_model(
                TargetModelConfig(
                    name_or_path=cfg.local_hf.name_or_path,
                    dtype=cfg.local_hf.dtype,
                    quantization=cfg.local_hf.quantization,
                )
            )
        elif loaded_model_for_local_hf is not None:
            loaded = loaded_model_for_local_hf
        else:
            raise ValueError(
                "generator.backend == 'local_hf' with no local_hf.name_or_path "
                "set, and no already-loaded target model was passed in to "
                "reuse. Set one or the other."
            )
        return LocalHFGenerator(loaded)

    raise ValueError(f"Unknown generator backend: {cfg.backend}")
