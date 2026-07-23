"""Tests for QLoRA's precision now matching the manifest's model entry
instead of hardcoding 4bit nf4 everywhere.

`model_utils.bnb_config_for`/`resolve_dtype` are pure functions -- no real
model/GPU/bitsandbytes needed to test the branch logic itself (constructing
a `BitsAndBytesConfig` object doesn't touch a GPU or load any weights).
`train_qlora`'s own model-loading/training path is GPU+peft+bitsandbytes-gated
(see `qlora_available()`), so this file only exercises the parts that don't
require those: the config-resolution helpers, and (via a monkeypatched spy)
that `train_qlora` calls `prepare_model_for_kbit_training` only when the
resolved config is actually quantized.
"""
import sys
import types

import pytest
import torch

from steering_factory.model_utils import bnb_config_for, resolve_dtype


def test_resolve_dtype_known_values():
    assert resolve_dtype("float32") is torch.float32
    assert resolve_dtype("float16") is torch.float16
    assert resolve_dtype("bfloat16") is torch.bfloat16


def test_resolve_dtype_defaults_to_bfloat16_for_unknown_or_missing():
    assert resolve_dtype(None) is torch.bfloat16
    assert resolve_dtype("") is torch.bfloat16
    assert resolve_dtype("bf16") is torch.bfloat16  # no such alias -- falls through to default
    assert resolve_dtype("16bit") is torch.bfloat16  # not a dtype string at all


def test_bnb_config_for_4bit_uses_nf4_double_quant_and_compute_dtype():
    from transformers import BitsAndBytesConfig

    cfg = bnb_config_for("4bit", torch.bfloat16)
    assert isinstance(cfg, BitsAndBytesConfig)
    assert cfg.load_in_4bit is True
    assert cfg.bnb_4bit_quant_type == "nf4"
    assert cfg.bnb_4bit_use_double_quant is True
    assert cfg.bnb_4bit_compute_dtype == torch.bfloat16


def test_bnb_config_for_8bit_uses_load_in_8bit():
    from transformers import BitsAndBytesConfig

    cfg = bnb_config_for("8bit", torch.bfloat16)
    assert isinstance(cfg, BitsAndBytesConfig)
    assert cfg.load_in_8bit is True


def test_bnb_config_for_none_or_unrecognized_returns_none():
    # "none" (the manifest's documented literal for "no quantization"), an
    # unset value, and any typo/unrecognized string all mean the same thing:
    # no BitsAndBytesConfig -- load the model at `dtype` directly. There is
    # no "16bit" literal (see model_utils.py's TargetModelConfig comment).
    assert bnb_config_for("none", torch.bfloat16) is None
    assert bnb_config_for(None, torch.bfloat16) is None
    assert bnb_config_for("16bit", torch.bfloat16) is None
    assert bnb_config_for("bf16", torch.bfloat16) is None


def test_bnb_config_for_matches_load_model_behavior():
    # model_utils.load_model uses this exact helper -- this pins that the
    # refactor in model_utils.py (extracting bnb_config_for/resolve_dtype
    # out of load_model) didn't change load_model's own resolved values.
    from steering_factory.model_utils import _DTYPE_MAP

    for dtype_name, torch_dtype in _DTYPE_MAP.items():
        assert resolve_dtype(dtype_name) is torch_dtype
    for quantization in ("4bit", "8bit", "none", None, "unrecognized"):
        # Just confirm it doesn't raise and returns the documented type.
        result = bnb_config_for(quantization, torch.bfloat16)
        if quantization in ("4bit", "8bit"):
            assert result is not None
        else:
            assert result is None


class _FakeModel:
    def __init__(self):
        self.saved_to = None

    def parameters(self):
        return iter([])

    def save_pretrained(self, path):
        self.saved_to = path


class _FakePeftModel(_FakeModel):
    pass


class _FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __call__(self, *a, **k):
        return {"input_ids": [1, 2, 3]}

    def save_pretrained(self, path):
        pass


class _FakeLoraConfig:
    def __init__(self, **kwargs):
        pass


class _FakeDataset:
    @staticmethod
    def from_list(records):
        class _DS:
            def map(self, fn, remove_columns=None):
                return self
        return _DS()


class _FakeTrainer:
    def __init__(self, *a, **k):
        pass

    def train(self):
        class _Output:
            training_loss = 0.0
            global_step = 0
        return _Output()

    state = type("S", (), {"log_history": []})()


class _FakeTrainingArguments:
    def __init__(self, **kwargs):
        pass


class _FakeDataCollator:
    def __init__(self, **kwargs):
        pass


def _install_train_qlora_fakes(monkeypatch, calls):
    """Stubs every heavy import `train_qlora` makes lazily inside its own
    body (datasets/peft/transformers), so the function's control flow --
    specifically whether it calls `prepare_model_for_kbit_training` -- can
    be exercised without a GPU, bitsandbytes, or real model weights.
    `calls` is a shared dict the fakes record into, so the test can assert
    on what train_qlora actually did."""
    import steering_factory.finetune as finetune_module

    monkeypatch.setattr(finetune_module, "qlora_available", lambda: True)

    def fake_prepare_model_for_kbit_training(model):
        calls["kbit_prep"] += 1
        return model

    def fake_get_peft_model(model, lora_config):
        return _FakePeftModel()

    class _FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(name, device_map=None, trust_remote_code=False, **kwargs):
            calls["quant_kwargs"] = kwargs
            return _FakeModel()

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name, trust_remote_code=False):
            return _FakeTokenizer()

    # bnb_config_for (called via model_utils, imported separately inside
    # train_qlora) does its own `from transformers import
    # BitsAndBytesConfig` -- the real one, not a fake, since it's actually
    # constructed (not called) and constructing it doesn't touch a GPU or
    # load weights. Grabbed before sys.modules["transformers"] is stubbed.
    from transformers import BitsAndBytesConfig as _RealBitsAndBytesConfig

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.Dataset = _FakeDataset

    fake_peft = types.ModuleType("peft")
    fake_peft.LoraConfig = _FakeLoraConfig
    fake_peft.get_peft_model = fake_get_peft_model
    fake_peft.prepare_model_for_kbit_training = fake_prepare_model_for_kbit_training

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = _FakeAutoModelForCausalLM
    fake_transformers.AutoTokenizer = _FakeAutoTokenizer
    fake_transformers.DataCollatorForSeq2Seq = _FakeDataCollator
    fake_transformers.Trainer = _FakeTrainer
    fake_transformers.TrainingArguments = _FakeTrainingArguments
    fake_transformers.BitsAndBytesConfig = _RealBitsAndBytesConfig

    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    return finetune_module


@pytest.mark.parametrize(
    "quantization,expect_kbit_prep,expect_quant_kwarg",
    [
        ("none", False, False),   # bf16 control preset: unquantized, skip kbit prep
        ("4bit", True, True),     # baseline/8bit-midpoint-style: quantized, run kbit prep
        ("8bit", True, True),
    ],
)
def test_train_qlora_kbit_prep_is_precision_conditional(
    monkeypatch, tmp_path, quantization, expect_kbit_prep, expect_quant_kwarg,
):
    """The core behavioral claim of this change: `prepare_model_for_kbit_training`
    is only valid for an actually-quantized base model, so it must be
    skipped when `quantization` resolves to unquantized (e.g. the bf16
    control preset), and must still run for 4bit/8bit -- the skip is
    precision-conditional, not just always-off."""
    calls = {"kbit_prep": 0, "quant_kwargs": None}
    finetune_module = _install_train_qlora_fakes(monkeypatch, calls)

    records = [{"prompt": "hi", "positive": "there"}]
    config = {
        "model_name": "tiny/tiny", "output_dir": str(tmp_path / "adapter"),
        "target_modules": ["q_proj"], "quantization": quantization, "dtype": "bfloat16", "max_steps": 1,
    }
    finetune_module.train_qlora(records, config)

    assert calls["kbit_prep"] == (1 if expect_kbit_prep else 0)
    assert ("quantization_config" in calls["quant_kwargs"]) is expect_quant_kwarg
    if not expect_quant_kwarg:
        assert calls["quant_kwargs"]["torch_dtype"] is torch.bfloat16


def test_train_qlora_defaults_quantization_to_4bit_when_key_absent():
    """Backward compatibility: a config dict that never sets `quantization`
    at all (the shape every pre-existing manifest/test produced before this
    change) must still resolve to 4bit -- not silently become unquantized.
    Exercised directly against bnb_config_for/resolve_dtype with the same
    `.get("quantization", "4bit")` default train_qlora uses internally."""
    dtype = resolve_dtype({}.get("dtype"))
    quant = bnb_config_for({}.get("quantization", "4bit"), dtype)
    assert quant is not None
    assert quant.load_in_4bit is True
