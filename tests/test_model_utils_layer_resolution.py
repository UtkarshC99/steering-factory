import torch.nn as nn

from steering_factory.model_utils import _resolve_hidden_size, resolve_layers


class _FakeDecoderLayer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden)


def test_resolve_layers_plain_model_layers():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([_FakeDecoderLayer(8) for _ in range(3)])

    layers, path = resolve_layers(Model())
    assert path == "model.layers"
    assert len(layers) == 3


def test_resolve_layers_nested_language_model_gemma3_style():
    """Gemma-3 (and other text+vision *ForConditionalGeneration wrappers)
    nest the text decoder under model.language_model.layers rather than
    model.layers directly."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.language_model = nn.Module()
            self.model.language_model.layers = nn.ModuleList([_FakeDecoderLayer(8) for _ in range(4)])

    layers, path = resolve_layers(Model())
    assert path == "model.language_model.layers"
    assert len(layers) == 4


def test_resolve_layers_prefers_model_layers_over_nested_when_both_present():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([_FakeDecoderLayer(8) for _ in range(2)])
            self.model.language_model = nn.Module()
            self.model.language_model.layers = nn.ModuleList([_FakeDecoderLayer(8) for _ in range(9)])

    layers, path = resolve_layers(Model())
    assert path == "model.layers"
    assert len(layers) == 2


def test_resolve_layers_raises_when_nothing_matches():
    class Model(nn.Module):
        pass

    try:
        resolve_layers(Model())
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "layer_path_override" in str(exc)


def test_resolve_hidden_size_top_level_attribute():
    class Config:
        hidden_size = 4096

    assert _resolve_hidden_size(Config(), layers=[]) == 4096


def test_resolve_hidden_size_gpt2_style_n_embd():
    class Config:
        n_embd = 768

    assert _resolve_hidden_size(Config(), layers=[]) == 768


def test_resolve_hidden_size_nested_text_config_gemma3_style():
    """Gemma3Config has no top-level hidden_size -- only text_config and
    vision_config -- so the resolver must fall through to text_config."""

    class TextConfig:
        hidden_size = 2560

    class Config:
        text_config = TextConfig()

    assert _resolve_hidden_size(Config(), layers=[]) == 2560


def test_resolve_hidden_size_falls_back_to_layer_weight_shape():
    class Config:
        pass

    layers = [_FakeDecoderLayer(1152)]
    assert _resolve_hidden_size(Config(), layers) == 1152


def test_resolve_hidden_size_raises_when_nothing_resolves():
    class Config:
        pass

    try:
        _resolve_hidden_size(Config(), layers=[])
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
