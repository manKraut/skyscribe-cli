from __future__ import annotations
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_model_and_tokenizer(
    base_model: str = None,
    adapter_path: str | None = None,
    quantize_4bit: bool = False,
):
    base = base_model or os.environ.get("SKYSCRIBE_BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    has_cuda = torch.cuda.is_available()
    use_4bit = bool(quantize_4bit and has_cuda)

    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        mdl = AutoModelForCausalLM.from_pretrained(
            base,
            device_map="auto",
            quantization_config=bnb_cfg,
            torch_dtype=torch.float16,
        )
    else:
        mdl = AutoModelForCausalLM.from_pretrained(
            base,
            device_map="cpu",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,  # avoid meta tensors
        )
        mdl.to("cpu")

    if getattr(mdl.config, "pad_token_id", None) is None:
        mdl.config.pad_token_id = tok.pad_token_id

    adapter = adapter_path or os.environ.get("SKYSCRIBE_LORA_ADAPTER")
    if adapter and os.path.isdir(adapter):
        mdl = PeftModel.from_pretrained(mdl, adapter)
        print(f"[SkyScribe] Loaded LoRA adapter: {adapter}")
    else:
        print("[SkyScribe] No LoRA adapter provided; using base model.")

    return tok, mdl
