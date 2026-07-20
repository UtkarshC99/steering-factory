"""Optional QLoRA backend. Imported lazily so base experiments stay light."""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List


def qlora_available() -> bool:
    try:
        import peft  # noqa: F401
        import trl  # noqa: F401
        return True
    except ImportError:
        return False


def train_qlora(records: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """Train a matched QLoRA adapter when optional dependencies are present.

    The caller supplies already-normalized records, preserving identical split
    and prompt formatting to the steering arm. Full trainer configuration is
    recorded by the runner manifest instead of hidden in this function.
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
        per_device_train_batch_size=int(config.get("batch_size", 1)), gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 8)),
        logging_steps=int(config.get("logging_steps", 5)), save_strategy="no", report_to=[])
    trainer = Trainer(model=model, args=args, train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100, padding=True))
    output = trainer.train()
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    size = sum(os.path.getsize(os.path.join(root, file)) for root, _, files in os.walk(config["output_dir"]) for file in files)
    return {"records": len(records), "wall_time_s": time.perf_counter() - started, "train_loss": output.training_loss,
            "global_step": output.global_step, "adapter_dir": config["output_dir"], "adapter_size_bytes": size}
