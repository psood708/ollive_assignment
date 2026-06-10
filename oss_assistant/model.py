import os
from typing import Dict, List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
_tokenizer = None
_model = None

def _load() -> None:
    global _tokenizer, _model
    if _tokenizer is not None:
        return
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
    )

def generate(message: str, history: List[Dict[str, str]]) -> Tuple[str, Dict[str, int]]:
    _load()
    messages = [{"role": "system", "content": "You are a helpful personal assistant."}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    text = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        output_ids = _model.generate(
            input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            pad_token_id=_tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_ids.shape[1]:]
    reply = _tokenizer.decode(new_tokens, skip_special_tokens=True)
    return reply, {"input": int(input_ids.shape[1]), "output": int(len(new_tokens))}
