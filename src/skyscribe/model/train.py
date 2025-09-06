from __future__ import annotations
import os, json, math, random
from dataclasses import dataclass
from typing import List, Dict

from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
                          DataCollatorForLanguageModeling, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

from skyscribe.agents.ingestion import resolve_location, fetch_forecast
from skyscribe.agents.features import facts_for_day
from skyscribe.model.templater import render_brief

import torch

SYSTEM = ("You write short, precise local weather briefs using only provided facts. "
          "Write 4–6 short sentences, include numeric ranges and time windows. "
          "End with a one-line advice. No emojis.")

def build_pairs(cities: List[str], *, units="metric", days=2) -> List[Dict]:
    """
    Build small (prompt,target) pairs by fetching 1–2 day forecasts for a handful of cities,
    computing facts, and templating target texts.
    """
    examples = []
    for city in cities:
        loc = resolve_location(city)
        if not loc: 
            continue
        fx = fetch_forecast(loc["lat"], loc["lon"], units=units, days=days)
        for day in (["today","tomorrow"] if days >= 2 else ["today"]):
            facts = {"location": loc["name"], **facts_for_day(fx, day=day, units=units)}
            if not facts or not facts.get("date"):
                continue
            prompt = f"{SYSTEM}\n\nFacts (JSON):\n{json.dumps(facts, ensure_ascii=False)}\n\nWrite the brief now:\n"
            target = render_brief(facts)
            examples.append({"prompt": prompt, "target": target})
    return examples

def tokenize_with_labels(tokenizer, examples, max_len=512):
    """Pad/truncate to a uniform length and mask prompt tokens in labels."""
    tokenized = []
    for ex in examples:
        prompt = ex["prompt"]
        target = ex["target"]

        # full sequence = prompt + target
        full_text = prompt + target

        enc_full = tokenizer(
            full_text,
            truncation=True,
            padding="max_length",   # <-- key change
            max_length=max_len,
        )

        # get prompt length in tokens (also truncated to same max_len)
        enc_prompt = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )

        input_ids = enc_full["input_ids"]
        attn_mask = enc_full["attention_mask"]
        labels = input_ids[:]  # copy

        # mask the prompt portion so loss only applies to target
        prompt_len = sum(1 for x in enc_prompt["attention_mask"] if x == 1)
        labels[:prompt_len] = [-100] * prompt_len

        tokenized.append({
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
        })
    return tokenized

def run_training(
    cities: List[str],
    out_dir: str = "adapters/tinyllama-lora",
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    units: str = "metric",
    days: int = 2,
    quantize_4bit: bool = False,
    epochs: float = 1.0,
    lr: float = 2e-4,
    batch_size: int = 1,
    grad_accum: int = 8,
):
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    has_cuda = torch.cuda.is_available()
    use_4bit = bool(quantize_4bit and has_cuda)

    bnb_cfg = None
    if quantize_4bit:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16"
        )

    if use_4bit:
        # CUDA + 4-bit path
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",                 # let accelerate place on GPU
            quantization_config=bnb_cfg,
            torch_dtype=torch.float16,
        )
        # prepare for k-bit training
        model = prepare_model_for_kbit_training(model)
    else:
        # CPU (or no 4-bit) path
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="cpu",                  # <-- explicit
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,           # <-- critical: avoid meta tensors
        )
        model.to("cpu")

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    bnb_cfg = None
    if quantize_4bit:
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        quantization_config=bnb_cfg
    )

    if quantize_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    pairs = build_pairs(cities, units=units, days=days)
    assert pairs, "No training pairs were generated (check cities/Internet)."

    tokenized = tokenize_with_labels(tokenizer, pairs)
    ds = Dataset.from_list(tokenized)

    args = TrainingArguments(
        output_dir=os.path.join(out_dir, "checkpoints"),
        num_train_epochs=epochs,
        learning_rate=lr,
        fp16=False,
        bf16=False,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
    trainer.train()

    model.save_pretrained(out_dir)  # saves LoRA adapter only
    tokenizer.save_pretrained(out_dir)
    print(f"Saved LoRA adapter to: {out_dir}")
