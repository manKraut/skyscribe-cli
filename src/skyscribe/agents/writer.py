from __future__ import annotations
import os, json, torch
from skyscribe.model.load import load_model_and_tokenizer

_tok = None
_mdl = None

SYSTEM = (
    "You write short, precise local weather briefs using only provided facts. "
    "4–6 short sentences, include numbers and time windows. End with a one-line advice. No emojis."
)
PROMPT = """{system}

Facts (JSON):
{facts}

Write the brief now:
"""

def _lazy():
    global _tok, _mdl
    if _mdl is None:
        quant = bool(int(os.environ.get("SKYSCRIBE_QUANT4BIT", "0")))
        adapter = os.environ.get("SKYSCRIBE_LORA_ADAPTER")
        base = os.environ.get("SKYSCRIBE_BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        _tok, _mdl = load_model_and_tokenizer(base_model=base, adapter_path=adapter, quantize_4bit=quant)
    return _tok, _mdl

def write_brief_from_facts(facts: dict, max_new_tokens: int = 180) -> str:
    tok, mdl = _lazy()
    text = PROMPT.format(system=SYSTEM, facts=json.dumps(facts, ensure_ascii=False))
    ids = tok(text, return_tensors="pt").to(mdl.device)
    out = mdl.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0)
    decoded = tok.decode(out[0], skip_special_tokens=True)
    return decoded.split("Write the brief now:")[-1].strip()
