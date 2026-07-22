"""Optional QLoRA backend. Imported lazily so base experiments stay light."""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from .callbacks import RunCallback, _safe


def qlora_available() -> bool:
    try:
        import peft  # noqa: F401
        import trl  # noqa: F401
        return True
    except ImportError:
        return False


def _build_trainer_callback(callback: Optional[RunCallback], context: Dict[str, Any]):
    """Wraps a RunCallback as a transformers.TrainerCallback, forwarding
    HF's own `on_log` events (loss/grad_norm/learning_rate/epoch, emitted
    every `logging_steps`) as `"train_log"` events. Returns None if no
    callback was given, so `Trainer(callbacks=[x] if x else [])` stays a
    no-op path identical to before this existed."""
    if callback is None:
        return None
    from transformers import TrainerCallback

    class _Forwarder(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            _safe(callback, {"arm": "qlora", "event": "train_log", "step": state.global_step,
                              "max_steps": state.max_steps if state.max_steps and state.max_steps > 0 else None,
                              "epoch": logs.get("epoch"), "loss": logs.get("loss"),
                              "grad_norm": logs.get("grad_norm"), "learning_rate": logs.get("learning_rate"),
                              **context})

    return _Forwarder()


def train_qlora(
    records: List[Dict[str, Any]],
    config: Dict[str, Any],
    callback: Optional[RunCallback] = None,
    callback_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train a matched QLoRA adapter when optional dependencies are present.

    The caller supplies already-normalized records, preserving identical split
    and prompt formatting to the steering arm. Full trainer configuration is
    recorded by the runner manifest instead of hidden in this function.

    `callback`, if given, receives one `"train_log"` event per
    `logging_steps` (HF's own `Trainer` instrumentation, just forwarded
    instead of discarded) via a small `TrainerCallback` adapter.
    `callback_context` (e.g. `{"model_id", "recipe_id"}`) is merged into
    every emitted event so a multi-model/multi-recipe run's events stay
    attributable. Always `None` from the CLI; see `callbacks.py`.
    """
    if not qlora_available():
        raise RuntimeError("QLoRA requires optional dependencies: pip install peft trl datasets")
    required = {"model_name", "output_dir", "target_modules"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"QLoRA config missing {sorted(missing)}")
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              DataCollatorForSeq2Seq, Trainer, TrainingArguments)
    from .model_utils import format_chat

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=config.get("trust_remote_code", False))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(config["model_name"], quantization_config=quant, device_map="auto",
                                                  trust_remote_code=config.get("trust_remote_code", False))
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=int(config.get("rank", 16)), lora_alpha=int(config.get("alpha", 32)),
        lora_dropout=float(config.get("dropout", 0.05)), bias="none", task_type="CAUSAL_LM", target_modules=config["target_modules"]))

    def tokenize(row: Dict[str, Any]) -> Dict[str, Any]:
        prompt, target = format_chat(tokenizer, row["prompt"]), row["positive"]
        full = tokenizer(prompt + target, truncation=True, max_length=int(config.get("max_length", 512)))
        prompt_ids = tokenizer(prompt, truncation=True, max_length=int(config.get("max_length", 512))).input_ids
        labels = list(full["input_ids"])
        labels[:len(prompt_ids)] = [-100] * min(len(prompt_ids), len(labels))
        full["labels"] = labels
        return full

    dataset = Dataset.from_list(records).map(tokenize, remove_columns=list(records[0].keys()))
    args = TrainingArguments(output_dir=config["output_dir"], num_train_epochs=float(config.get("epochs", 1)),
        max_steps=int(config.get("max_steps", -1)), learning_rate=float(config.get("learning_rate", 2e-4)),
        # Defaults 4/2 (was 1/8): effective batch size stays 4*2=8, unchanged
        # from 1*8 -- training dynamics and the resulting adapter are
        # identical, only micro-batch throughput improves. See
        # GPU-batching plan for the L4/22.5GB VRAM sizing this is based on.
        per_device_train_batch_size=int(config.get("batch_size", 4)), gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 2)),
        logging_steps=int(config.get("logging_steps", 5)), save_strategy="no", report_to=[])
    trainer_callback = _build_trainer_callback(callback, callback_context or {})
    trainer = Trainer(model=model, args=args, train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100, padding=True),
        callbacks=[trainer_callback] if trainer_callback is not None else None)
    output = trainer.train()
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    size = sum(os.path.getsize(os.path.join(root, file)) for root, _, files in os.walk(config["output_dir"]) for file in files)
    log_history = list(trainer.state.log_history)
    return {"records": len(records), "wall_time_s": time.perf_counter() - started, "train_loss": output.training_loss,
            "global_step": output.global_step, "adapter_dir": config["output_dir"], "adapter_size_bytes": size,
            "log_history": log_history}


def _generate_batched_rows(
    model, tokenizer, examples: List[Dict[str, Any]], max_new_tokens: int, batch_size: int,
) -> List[Dict[str, Any]]:
    """The actual batched-generation loop, factored out of
    `evaluate_qlora_adapter` so it's testable against a plain (unquantized,
    non-PEFT) tiny model -- quantization/PEFT loading can't run without a
    real CUDA GPU, but the batching/padding/score-reduction logic here has
    nothing to do with either and is exactly what needs a correctness
    guarantee (see tests/test_finetune_batching.py).

    Same pattern as sweep.generate_with_steering_batch: left-padded batch,
    scores reduced to chosen-token logprob per step (no steering hook or
    JS-divergence baseline needed here, so first_step_probs is discarded).
    `latency_s` is amortized (`batch_wall_time_s / batch_size`);
    `batch_size`/`batch_wall_time_s` are recorded alongside it per row so
    cost analysis can tell measured-per-row apart from amortized-per-row,
    matching the steering arm's row schema."""
    import torch

    from .model_utils import format_chat
    from .sweep import _reduce_scores

    device = next(model.parameters()).device
    rows: List[Dict[str, Any]] = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        prompt_texts = [format_chat(tokenizer, example["prompt"]) for example in chunk]

        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            enc = tokenizer(prompt_texts, return_tensors="pt", padding=True).to(device)
        finally:
            tokenizer.padding_side = original_padding_side
        prompt_len = enc["input_ids"].shape[1]

        before = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                  output_scores=True, return_dict_in_generate=True,
                                  pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        batch_wall_time_s = time.perf_counter() - before

        for row, example in enumerate(chunk):
            gen_ids_row = out.sequences[row][prompt_len:]
            text = tokenizer.decode(gen_ids_row, skip_special_tokens=True)
            row_scores = [step_logits[row : row + 1] for step_logits in out.scores]
            token_logprobs, _ = _reduce_scores(row_scores, gen_ids_row)
            rows.append({
                "example_id": example["id"], "behavior_id": example["behavior_id"], "split": example["split"],
                "category": example.get("category"), "prompt": example["prompt"], "output": text,
                "latency_s": batch_wall_time_s / len(chunk), "batch_size": len(chunk),
                "batch_wall_time_s": batch_wall_time_s, "tokens_generated": len(token_logprobs),
            })
    return rows


def evaluate_qlora_adapter(
    examples: List[Dict[str, Any]],
    model_name: str,
    adapter_dir: str,
    max_new_tokens: int = 96,
    trust_remote_code: bool = False,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """Generate on `examples` (validation/test split) with the base model +
    trained QLoRA adapter, returning per-example rows in the same shape as
    the steering arm's `results/generations.jsonl` (minus steering-only
    fields like layer_idx/coefficient/method), so `comparison.py` can score
    both arms with the identical evaluator and join on quality.

    Loaded lazily and released per call -- this is meant to run once per
    (model, recipe) after training, not kept resident.
    """
    if not qlora_available():
        raise RuntimeError("QLoRA requires optional dependencies: pip install peft trl datasets")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant, device_map="auto",
                                                 trust_remote_code=trust_remote_code)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    rows = _generate_batched_rows(model, tokenizer, examples, max_new_tokens, batch_size)

    del model, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"rows": rows, "wall_time_s": time.perf_counter() - started}
