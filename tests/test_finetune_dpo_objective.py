"""Tests for finetune.py's `objective` dispatch (Tier 3): the QLoRA arm
previously trained ONLY on `prompt + positive`, discarding `negative`
entirely, even though ContrastiveExample always carries it and the
steering arm's whole mechanism is the pos-minus-neg activation
difference. A real run showed exactly the predicted failure: with a
single canned refusal as `positive`, "always emit it" is the SFT
objective's optimum, and the adapter did that on 358/364 held-out rows
regardless of input.

`objective: dpo` trains on the PREFERENCE between `positive` (chosen) and
`negative` (rejected) via trl.DPOTrainer instead. `objective` defaults to
"sft" -- every manifest written before this existed must behave exactly
as before, unchanged.

Mirrors test_finetune_quantization.py's approach: stubs every heavy
import train_qlora/_train_qlora_dpo makes lazily inside their own bodies
(datasets/peft/transformers/trl), so this needs no real GPU/bitsandbytes/
trl training loop -- only the control-flow and data-shaping claims are
under test, which is exactly what a real run's failure was about (the
WRONG DATA reaching the trainer), not model quality.
"""
import sys
import types

import pytest
import torch

from tests.test_finetune_quantization import (
    _FakeDataCollator,
    _FakeDataset,
    _FakeLoraConfig,
    _FakeModel,
    _FakePeftModel,
    _FakeTokenizer,
    _FakeTrainer,
    _FakeTrainingArguments,
)


def _install_sft_fakes(monkeypatch, calls):
    """Same fakes as test_finetune_quantization's _install_train_qlora_fakes,
    duplicated (not imported) so this file doesn't depend on that file's
    exact fixture signature evolving independently."""
    import steering_factory.finetune as finetune_module

    monkeypatch.setattr(finetune_module, "qlora_available", lambda: True)

    def fake_prepare_model_for_kbit_training(model):
        calls["kbit_prep"] = calls.get("kbit_prep", 0) + 1
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


class _CapturingDataset:
    """Records exactly the list of dicts train_qlora_dpo builds, so the
    test can assert on prompt/chosen/rejected content directly rather than
    trusting an opaque from_list call succeeded."""
    captured = None

    @staticmethod
    def from_list(records):
        _CapturingDataset.captured = list(records)
        return _CapturingDataset()


class _FakeDPOTrainer:
    last_kwargs = None

    def __init__(self, **kwargs):
        _FakeDPOTrainer.last_kwargs = kwargs

    def train(self):
        class _Output:
            training_loss = 0.0
            global_step = 0
        return _Output()

    def save_model(self, path):
        pass

    state = type("S", (), {"log_history": []})()
    tokenizer = _FakeTokenizer()


class _FakeDPOConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


def _install_dpo_fakes(monkeypatch, calls):
    """Adds trl's DPOTrainer/DPOConfig fakes on top of the SFT fakes --
    _train_qlora_dpo still loads the base model the same way train_qlora's
    SFT path does (same bnb_config_for/resolve_dtype), only the trainer and
    dataset shape differ."""
    finetune_module = _install_sft_fakes(monkeypatch, calls)

    _CapturingDataset.captured = None
    _FakeDPOTrainer.last_kwargs = None

    fake_datasets = sys.modules["datasets"]
    fake_datasets.Dataset = _CapturingDataset

    fake_trl = types.ModuleType("trl")
    fake_trl.DPOTrainer = _FakeDPOTrainer
    fake_trl.DPOConfig = _FakeDPOConfig
    monkeypatch.setitem(sys.modules, "trl", fake_trl)

    return finetune_module


def test_objective_defaults_to_sft_and_is_recorded_on_the_result(monkeypatch, tmp_path):
    calls = {}
    finetune_module = _install_sft_fakes(monkeypatch, calls)
    records = [{"prompt": "hi", "positive": "there", "negative": "no"}]
    config = {"model_name": "tiny/tiny", "output_dir": str(tmp_path / "adapter"),
              "target_modules": ["q_proj"], "max_steps": 1}
    result = finetune_module.train_qlora(records, config)
    assert result["objective"] == "sft"


def test_unknown_objective_raises():
    import steering_factory.finetune as finetune_module
    with pytest.raises(ValueError, match="objective"):
        finetune_module.train_qlora(
            [{"prompt": "hi", "positive": "there", "negative": "no"}],
            {"model_name": "tiny/tiny", "output_dir": "x", "target_modules": ["q_proj"], "objective": "nonsense"},
        )


def test_dpo_objective_dispatches_to_dpo_trainer_and_records_objective(monkeypatch, tmp_path):
    calls = {}
    finetune_module = _install_dpo_fakes(monkeypatch, calls)
    records = [{"prompt": "hi", "positive": "there", "negative": "no"}]
    config = {"model_name": "tiny/tiny", "output_dir": str(tmp_path / "adapter"),
              "target_modules": ["q_proj"], "max_steps": 1, "objective": "dpo"}
    result = finetune_module.train_qlora(records, config)
    assert result["objective"] == "dpo"
    assert _FakeDPOTrainer.last_kwargs is not None


def test_dpo_dataset_uses_both_positive_and_negative_as_chosen_and_rejected(monkeypatch, tmp_path):
    """THE core claim of Tier 3: unlike the SFT path (prompt+positive
    only, negative silently unused), the DPO dataset must contain BOTH
    positive (as chosen) and negative (as rejected) for every record --
    trl.DPOTrainer's own _prepare_dataset expects exactly prompt/chosen/
    rejected as plain strings (confirmed against its source, not assumed)."""
    calls = {}
    finetune_module = _install_dpo_fakes(monkeypatch, calls)
    records = [
        {"prompt": "Do you agree?", "positive": "(B) No, I disagree.", "negative": "(A) Yes, I agree!"},
        {"prompt": "Second question?", "positive": "honest answer", "negative": "sycophantic answer"},
    ]
    config = {"model_name": "tiny/tiny", "output_dir": str(tmp_path / "adapter"),
              "target_modules": ["q_proj"], "max_steps": 1, "objective": "dpo"}
    finetune_module.train_qlora(records, config)

    captured = _CapturingDataset.captured
    assert captured is not None
    assert len(captured) == 2
    for row in captured:
        assert set(row.keys()) == {"prompt", "chosen", "rejected"}
    assert captured[0]["chosen"] == "(B) No, I disagree."
    assert captured[0]["rejected"] == "(A) Yes, I agree!"
    assert captured[1]["chosen"] == "honest answer"
    assert captured[1]["rejected"] == "sycophantic answer"


def test_dpo_config_reads_beta_and_max_length_from_manifest(monkeypatch, tmp_path):
    calls = {}
    finetune_module = _install_dpo_fakes(monkeypatch, calls)
    records = [{"prompt": "hi", "positive": "there", "negative": "no"}]
    config = {"model_name": "tiny/tiny", "output_dir": str(tmp_path / "adapter"),
              "target_modules": ["q_proj"], "max_steps": 1, "objective": "dpo",
              "dpo_beta": 0.3, "max_length": 256}
    finetune_module.train_qlora(records, config)
    dpo_args = _FakeDPOTrainer.last_kwargs["args"]
    assert dpo_args.beta == 0.3
    assert dpo_args.max_length == 256


def test_dpo_config_defaults_beta_to_trl_default(monkeypatch, tmp_path):
    calls = {}
    finetune_module = _install_dpo_fakes(monkeypatch, calls)
    records = [{"prompt": "hi", "positive": "there", "negative": "no"}]
    config = {"model_name": "tiny/tiny", "output_dir": str(tmp_path / "adapter"),
              "target_modules": ["q_proj"], "max_steps": 1, "objective": "dpo"}
    finetune_module.train_qlora(records, config)
    dpo_args = _FakeDPOTrainer.last_kwargs["args"]
    assert dpo_args.beta == 0.1


def test_dpo_passes_peft_config_directly_to_dpotrainer_not_get_peft_model(monkeypatch, tmp_path):
    """DPOTrainer applies LoRA itself via its own `peft_config` kwarg
    (confirmed against its __init__ source: it calls get_peft_model
    internally when given peft_config) -- _train_qlora_dpo must NOT also
    call get_peft_model manually first, which would double-wrap the
    model."""
    calls = {"get_peft_model_calls": 0}
    finetune_module = _install_dpo_fakes(monkeypatch, calls)

    import steering_factory.finetune as fm
    original = sys.modules["peft"].get_peft_model

    def counting_get_peft_model(*a, **k):
        calls["get_peft_model_calls"] += 1
        return original(*a, **k)
    sys.modules["peft"].get_peft_model = counting_get_peft_model

    records = [{"prompt": "hi", "positive": "there", "negative": "no"}]
    config = {"model_name": "tiny/tiny", "output_dir": str(tmp_path / "adapter"),
              "target_modules": ["q_proj"], "max_steps": 1, "objective": "dpo"}
    finetune_module.train_qlora(records, config)
    assert calls["get_peft_model_calls"] == 0
    assert "peft_config" in _FakeDPOTrainer.last_kwargs


def test_dpo_prompt_is_formatted_through_the_chat_template(monkeypatch, tmp_path):
    """The prompt going into the DPO dataset must go through format_chat,
    matching the SFT path and the steering arm's own extraction/eval --
    otherwise this arm would train on a differently-formatted prompt than
    the one it's being compared against."""
    calls = {}
    finetune_module = _install_dpo_fakes(monkeypatch, calls)

    sentinel_calls = []

    def fake_format_chat(tokenizer, prompt, system=None):
        sentinel_calls.append(prompt)
        return f"<<TEMPLATED>>{prompt}"

    import steering_factory.model_utils as model_utils_module
    monkeypatch.setattr(model_utils_module, "format_chat", fake_format_chat)

    records = [{"prompt": "raw prompt text", "positive": "there", "negative": "no"}]
    config = {"model_name": "tiny/tiny", "output_dir": str(tmp_path / "adapter"),
              "target_modules": ["q_proj"], "max_steps": 1, "objective": "dpo"}
    finetune_module.train_qlora(records, config)

    assert "raw prompt text" in sentinel_calls
    assert _CapturingDataset.captured[0]["prompt"] == "<<TEMPLATED>>raw prompt text"
