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

    `config["objective"]` (added 2026-07-26, default "sft" -- unchanged
    behavior for every manifest written before this existed) selects the
    training objective:
      - "sft": the original path below, trains on `prompt + positive` only.
        `negative` is loaded onto every record (ContrastiveExample always
        carries it) but never read here -- this arm discards half the
        supervision the steering arm's whole mechanism is built on (the
        pos-minus-neg activation difference). A real run showed exactly
        the failure mode this predicts: with a single canned refusal as
        `positive`, "always emit it" IS the objective's optimum, and the
        adapter did that on 358/364 held-out rows regardless of input.
      - "dpo": trains on (prompt, chosen=positive, rejected=negative)
        via trl.DPOTrainer -- the data is already in exactly this shape,
        `trl` is already a dependency. This is the closer analogue to
        steering: the loss is a function of the PREFERENCE between the two
        responses, not a memorized target string.
    See finetune.py's _train_qlora_dpo for the dpo path.
    """
    if not qlora_available():
        raise RuntimeError("QLoRA requires optional dependencies: pip install peft trl datasets")
    required = {"model_name", "output_dir", "target_modules"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"QLoRA config missing {sorted(missing)}")

    objective = config.get("objective", "sft")
    if objective == "dpo":
        return _train_qlora_dpo(records, config, callback, callback_context)
    if objective != "sft":
        raise ValueError(f"Unknown finetune.objective: {objective!r} (expected 'sft' or 'dpo')")

    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments
    from .model_utils import bnb_config_for, format_chat, resolve_dtype

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=config.get("trust_remote_code", False))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Reads the model entry's own quantization/dtype instead of hardcoding
    # 4bit nf4 -- matches the steering arm's precision (model_utils.load_model
    # uses the identical bnb_config_for/resolve_dtype pair) so a quantization
    # sweep varies both arms together rather than only the steering side.
    # "qlora" as a name stays literal ("Q" = 4-bit) only when quantization is
    # actually 4bit/8bit; an unquantized entry runs plain LoRA on a full-
    # precision base -- see the quantization/dtype fields recorded on this
    # adapter's result row (runner.py) for the honest label per run.
    # Default "4bit" (not None) when the key is simply absent from config --
    # preserves the pre-existing hardcoded-4bit behavior for any caller that
    # doesn't set quantization explicitly, exactly like
    # evaluate_qlora_adapter/evaluate_base_model's own "4bit" default below.
    dtype = resolve_dtype(config.get("dtype"))
    quant = bnb_config_for(config.get("quantization", "4bit"), dtype)
    quant_kwargs = {"quantization_config": quant} if quant is not None else {"torch_dtype": dtype}
    model = AutoModelForCausalLM.from_pretrained(config["model_name"], device_map="auto",
                                                  trust_remote_code=config.get("trust_remote_code", False), **quant_kwargs)
    if quant is not None:
        # prepare_model_for_kbit_training casts norms to fp32 and enables
        # gradient checkpointing hooks specific to a quantized base -- it is
        # only valid (and only needed) when the base is actually loaded
        # k-bit; calling it on a full-precision model is unnecessary and, for
        # some versions, mishandles a base that has no quantized layers.
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=int(config.get("rank", 16)), lora_alpha=int(config.get("alpha", 32)),
        lora_dropout=float(config.get("dropout", 0.05)), bias="none", task_type="CAUSAL_LM", target_modules=config["target_modules"]))

    def tokenize(row: Dict[str, Any]) -> Dict[str, Any]:
        prompt, target = format_chat(tokenizer, row["prompt"], system=config.get("system_prompt")), row["positive"]
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
            "log_history": log_history, "objective": "sft"}


def _train_qlora_dpo(
    records: List[Dict[str, Any]],
    config: Dict[str, Any],
    callback: Optional[RunCallback] = None,
    callback_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """DPO objective: trains on the PREFERENCE between `positive` (chosen)
    and `negative` (rejected) for the same prompt, rather than memorizing
    `positive` as a fixed target. trl.DPOTrainer's expected dataset schema
    (confirmed directly against its own _prepare_dataset -- not assumed)
    is exactly `prompt`/`chosen`/`rejected` as plain strings, which is a
    direct rename of this project's own `prompt`/`positive`/`negative`
    fields -- no reshaping needed.

    Model loading mirrors train_qlora's SFT path (same bnb_config_for/
    resolve_dtype precision handling, so quantization stays matched to the
    steering arm), except LoRA is applied via DPOTrainer's own
    `peft_config` argument rather than a manual get_peft_model call --
    DPOTrainer calls get_peft_model internally when given peft_config, but
    (confirmed by reading DPOTrainer.__init__) does NOT call
    prepare_model_for_kbit_training itself, so that step is still done
    here explicitly for a quantized base, exactly like the SFT path."""
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer
    from .model_utils import bnb_config_for, format_chat, resolve_dtype

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=config.get("trust_remote_code", False))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = resolve_dtype(config.get("dtype"))
    quant = bnb_config_for(config.get("quantization", "4bit"), dtype)
    quant_kwargs = {"quantization_config": quant} if quant is not None else {"torch_dtype": dtype}
    model = AutoModelForCausalLM.from_pretrained(config["model_name"], device_map="auto",
                                                  trust_remote_code=config.get("trust_remote_code", False), **quant_kwargs)
    if quant is not None:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(r=int(config.get("rank", 16)), lora_alpha=int(config.get("alpha", 32)),
        lora_dropout=float(config.get("dropout", 0.05)), bias="none", task_type="CAUSAL_LM", target_modules=config["target_modules"])

    dataset = Dataset.from_list([
        {"prompt": format_chat(tokenizer, r["prompt"], system=config.get("system_prompt")),
         "chosen": r["positive"], "rejected": r["negative"]}
        for r in records
    ])

    args = DPOConfig(
        output_dir=config["output_dir"], num_train_epochs=float(config.get("epochs", 1)),
        max_steps=int(config.get("max_steps", -1)), learning_rate=float(config.get("learning_rate", 2e-4)),
        per_device_train_batch_size=int(config.get("batch_size", 4)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 2)),
        logging_steps=int(config.get("logging_steps", 5)), save_strategy="no", report_to=[],
        # beta: how strongly the loss penalizes moving away from the
        # reference (pre-training) policy -- trl's own default (0.1) unless
        # the manifest overrides it. Deliberately NOT reusing "temperature"
        # or another borrowed name; this is DPO's own hyperparameter.
        beta=float(config.get("dpo_beta", 0.1)),
        max_length=int(config.get("max_length", 512)),
    )
    trainer_callback = _build_trainer_callback(callback, callback_context or {})
    trainer = DPOTrainer(
        model=model, args=args, train_dataset=dataset, processing_class=tokenizer, peft_config=peft_config,
        callbacks=[trainer_callback] if trainer_callback is not None else None,
    )
    output = trainer.train()
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    size = sum(os.path.getsize(os.path.join(root, file)) for root, _, files in os.walk(config["output_dir"]) for file in files)
    log_history = list(trainer.state.log_history)
    return {"records": len(records), "wall_time_s": time.perf_counter() - started, "train_loss": output.training_loss,
            "global_step": output.global_step, "adapter_dir": config["output_dir"], "adapter_size_bytes": size,
            "log_history": log_history, "objective": "dpo"}


def _generate_batched_rows(
    model, tokenizer, examples: List[Dict[str, Any]], max_new_tokens: int, batch_size: int,
    max_length: int = 1024, system_prompt: Optional[str] = None,
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
    matching the steering arm's row schema.

    `max_length` MUST track the steering arm's own per-recipe
    `decoding.max_length`: it was hardcoded to 1024 here while the steering
    arm honored a per-recipe override (2048 for structured_output_real),
    which silently truncated the two arms' prompts at DIFFERENT lengths for
    the same recipe -- the arms would then be compared on inputs they did
    not both see in full.

    Length bucketing + OOM backoff mirror the steering arm's eval loop
    (see sweep.length_bucketed_chunks / sweep.run_with_oom_backoff) so both
    arms stay stable on recipes whose prompt lengths vary by an order of
    magnitude. Row order always follows `examples`, never batch order."""
    import torch

    from .model_utils import format_chat
    from .sweep import _reduce_scores, length_bucketed_chunks, run_with_oom_backoff

    device = next(model.parameters()).device
    if not examples:
        return []
    prompt_texts_all = [format_chat(tokenizer, example["prompt"], system=system_prompt) for example in examples]
    rows_by_index: List[Optional[Dict[str, Any]]] = [None] * len(examples)

    def _run_batch(indices: List[int]) -> List[Dict[str, Any]]:
        chunk = [examples[i] for i in indices]
        prompt_texts = [prompt_texts_all[i] for i in indices]

        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            # truncation=True + an explicit cap: without a length bound, a
            # single oversized prompt in the batch pads every other row out
            # to match it, which caused a real CUDA OOM (~26 GiB requested
            # in one torch.embedding call) during a steering sweep.
            enc = tokenizer(prompt_texts, return_tensors="pt", truncation=True,
                            max_length=max_length, padding=True).to(device)
        finally:
            tokenizer.padding_side = original_padding_side
        prompt_len = enc["input_ids"].shape[1]

        before = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                  output_scores=True, return_dict_in_generate=True,
                                  pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        batch_wall_time_s = time.perf_counter() - before

        built = []
        for row, example in enumerate(chunk):
            gen_ids_row = out.sequences[row][prompt_len:]
            text = tokenizer.decode(gen_ids_row, skip_special_tokens=True)
            row_scores = [step_logits[row : row + 1] for step_logits in out.scores]
            token_logprobs, _ = _reduce_scores(row_scores, gen_ids_row)
            built.append({
                "example_id": example["id"], "behavior_id": example["behavior_id"], "split": example["split"],
                "category": example.get("category"), "prompt": example["prompt"], "output": text,
                "latency_s": batch_wall_time_s / len(chunk), "batch_size": len(chunk),
                "batch_wall_time_s": batch_wall_time_s, "tokens_generated": len(token_logprobs),
            })
        return built

    for indices in length_bucketed_chunks(tokenizer, prompt_texts_all, batch_size, max_length):
        # run_with_oom_backoff halves on OOM; it operates on the index list
        # so results scatter back to their original positions either way.
        built = run_with_oom_backoff(_run_batch, indices)
        for index, row in zip(indices, built):
            rows_by_index[index] = row
    return [row for row in rows_by_index if row is not None]


def evaluate_qlora_adapter(
    examples: List[Dict[str, Any]],
    model_name: str,
    adapter_dir: str,
    max_new_tokens: int = 96,
    trust_remote_code: bool = False,
    batch_size: int = 16,
    quantization: Optional[str] = "4bit",
    dtype: Optional[str] = None,
    max_length: int = 1024,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate on `examples` (validation/test split) with the base model +
    trained QLoRA adapter, returning per-example rows in the same shape as
    the steering arm's `results/generations.jsonl` (minus steering-only
    fields like layer_idx/coefficient/method), so `comparison.py` can score
    both arms with the identical evaluator and join on quality.

    `quantization`/`dtype` must match whatever `train_qlora` loaded the
    adapter's base model at (default "4bit" preserves the pre-existing
    hardcoded behavior for callers that don't pass these) -- reloading the
    base at a DIFFERENT precision than it was trained on would silently
    evaluate a mismatched model.

    Loaded lazily and released per call -- this is meant to run once per
    (model, recipe) after training, not kept resident.
    """
    if not qlora_available():
        raise RuntimeError("QLoRA requires optional dependencies: pip install peft trl datasets")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .model_utils import bnb_config_for, resolve_dtype

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    resolved_dtype = resolve_dtype(dtype)
    quant = bnb_config_for(quantization, resolved_dtype)
    quant_kwargs = {"quantization_config": quant} if quant is not None else {"torch_dtype": resolved_dtype}
    base = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",
                                                 trust_remote_code=trust_remote_code, **quant_kwargs)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    rows = _generate_batched_rows(model, tokenizer, examples, max_new_tokens, batch_size, max_length, system_prompt)

    del model, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"rows": rows, "wall_time_s": time.perf_counter() - started}


def evaluate_base_model(
    examples: List[Dict[str, Any]],
    model_name: str,
    max_new_tokens: int = 96,
    trust_remote_code: bool = False,
    batch_size: int = 16,
    quantization: Optional[str] = "4bit",
    dtype: Optional[str] = None,
    max_length: int = 1024,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Same as `evaluate_qlora_adapter` but loads the plain base model with
    no PEFT adapter -- used for the "before" half of the QLoRA arm's
    capability-regression measurement (capability_probe.py), so the
    before/after comparison is exactly base-model vs base-model+adapter,
    not base-model vs some other loading path. `quantization`/`dtype`
    default to the pre-existing hardcoded 4bit behavior for callers that
    don't pass these, and should match whatever the adapter itself was
    trained/evaluated at."""
    if not qlora_available():
        raise RuntimeError("QLoRA requires optional dependencies: pip install peft trl datasets")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .model_utils import bnb_config_for, resolve_dtype

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    resolved_dtype = resolve_dtype(dtype)
    quant = bnb_config_for(quantization, resolved_dtype)
    quant_kwargs = {"quantization_config": quant} if quant is not None else {"torch_dtype": resolved_dtype}
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",
                                                  trust_remote_code=trust_remote_code, **quant_kwargs)
    model.eval()

    rows = _generate_batched_rows(model, tokenizer, examples, max_new_tokens, batch_size, max_length, system_prompt)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"rows": rows, "wall_time_s": time.perf_counter() - started}
