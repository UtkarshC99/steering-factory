from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Optional, List

import yaml


@dataclass
class TargetModelConfig:
    name_or_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    dtype: str = "bfloat16"
    quantization: str = "4bit"  # none | 4bit | 8bit
    device_map: str = "auto"
    layer_path_override: Optional[str] = None
    trust_remote_code: bool = False


@dataclass
class AnthropicGenConfig:
    model: str = "claude-sonnet-5"


@dataclass
class OpenAICompatGenConfig:
    base_url: str = "http://localhost:11434/v1"
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "llama3.1:8b-instruct"


@dataclass
class LocalHFGenConfig:
    name_or_path: Optional[str] = None
    dtype: str = "bfloat16"
    quantization: str = "4bit"


@dataclass
class GeneratorConfig:
    backend: str = "anthropic"  # anthropic | openai_compat | local_hf
    anthropic: AnthropicGenConfig = field(default_factory=AnthropicGenConfig)
    openai_compat: OpenAICompatGenConfig = field(default_factory=OpenAICompatGenConfig)
    local_hf: LocalHFGenConfig = field(default_factory=LocalHFGenConfig)


@dataclass
class StorageConfig:
    vectors_dir: str = "data/vectors"
    examples_dir: str = "data/examples"


@dataclass
class DefaultsConfig:
    num_pairs_per_topic: int = 24
    extraction_method: str = "mean_diff"
    layer_fraction: float = 0.6
    sweep_coefficients: List[float] = field(
        default_factory=lambda: [-4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4]
    )
    fluency_cap_ratio: float = 1.6
    max_new_tokens: int = 120


@dataclass
class AppConfig:
    target_model: TargetModelConfig = field(default_factory=TargetModelConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Optional[str] = None) -> AppConfig:
    """Load config.yaml if present, falling back to defaults. Never raises
    on a missing file -- the app should always be runnable with sane
    defaults and full override from the Streamlit sidebar."""
    cfg = AppConfig()
    candidates = [path] if path else ["config.yaml", "config.example.yaml"]
    raw = {}
    for c in candidates:
        if c and os.path.exists(c):
            with open(c, "r") as f:
                raw = yaml.safe_load(f) or {}
            break

    tm = _merge(asdict(cfg.target_model), raw.get("target_model", {}))
    gen_raw = raw.get("generator", {})
    g = _merge(asdict(cfg.generator), gen_raw)
    st = _merge(asdict(cfg.storage), raw.get("storage", {}))
    df = _merge(asdict(cfg.defaults), raw.get("defaults", {}))

    return AppConfig(
        target_model=TargetModelConfig(**tm),
        generator=GeneratorConfig(
            backend=g.get("backend", "anthropic"),
            anthropic=AnthropicGenConfig(**g.get("anthropic", {})),
            openai_compat=OpenAICompatGenConfig(**g.get("openai_compat", {})),
            local_hf=LocalHFGenConfig(**g.get("local_hf", {})),
        ),
        storage=StorageConfig(**st),
        defaults=DefaultsConfig(**df),
    )
