"""HF transformers generation backend for the 3 new 2026 models
(GLM-4.7-Flash / Qwen3.6-27B / Gemma-4-31B-it).

Why this exists: those architectures (glm4_moe_lite / qwen3_5 / gemma4) need
transformers>=5.x, which is incompatible with the vLLM 0.9 used by the existing
pipeline, and no vLLM that supports them runs on this host's CUDA 12.4 driver.
So new-model generation goes through plain HF `.generate()` in the isolated
`Recur2` env (transformers 5.12.1, torch 2.7 cu126). The old vLLM scripts are
left untouched for the old models.

Keep generation knobs aligned with the config-priority rule: only `temperature`
is taken from config; top-p / top-k are never passed (see memory
feedback_sampling_params_only_temperature). Greedy decoding is used when
temperature == 0.

This module is import-safe without a GPU (torch import only).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# --- model family detection --------------------------------------------------

def model_type_of(model_path: str) -> str:
    """Read config.json model_type without importing the model."""
    import json
    cfg = json.load(open(Path(model_path) / "config.json"))
    return cfg.get("model_type", "")


# model_types whose chat template understands `enable_thinking`. DeepSeek/QwQ
# hardcode <think> in the template and ignore the kwarg (harmless to pass).
_ENABLE_THINKING_TYPES = {"qwen3_5", "glm4_moe_lite", "gemma4", "qwen3"}


# --- loading -----------------------------------------------------------------

def load_hf(model_path: str, dtype=torch.bfloat16):
    """Load tokenizer + model onto the visible CUDA device.

    Caller is responsible for setting CUDA_VISIBLE_DEVICES before import-time
    CUDA init (i.e. before calling this).
    """
    tok = AutoTokenizer.from_pretrained(model_path)
    # decoder-only batched generation requires LEFT padding
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, device_map="cuda:0"
    )
    model.eval()
    return model, tok


# --- prompt construction -----------------------------------------------------

def build_prompt_text(tokenizer, user_content: str, model_type: str) -> str:
    """Render the chat template as a string (tokenize=False).

    enable_thinking=True is passed for templates that understand it so the
    model actually produces a reasoning trace; it is silently ignored by
    templates that don't reference it (DeepSeek/QwQ).
    """
    kwargs = dict(add_generation_prompt=True, tokenize=False)
    if model_type in _ENABLE_THINKING_TYPES:
        kwargs["enable_thinking"] = True
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}], **kwargs
    )


# --- batched generation ------------------------------------------------------

@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    prompt_texts: list[str],
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    """Generate completions for a batch of pre-rendered prompt strings.

    Returns the newly generated text for each prompt (prompt stripped),
    decoded WITHOUT skipping special tokens so </think> / channel markers
    survive for the thinking split.
    """
    enc = tokenizer(
        prompt_texts, return_tensors="pt", padding=True, add_special_tokens=False
    ).to("cuda:0")
    do_sample = temperature is not None and temperature > 0
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
    out = model.generate(**enc, **gen_kwargs)
    gen_only = out[:, enc["input_ids"].shape[1]:]
    return [
        tokenizer.decode(row, skip_special_tokens=False) for row in gen_only
    ]


# --- model-aware thinking split ---------------------------------------------

_THINK_CLOSE = "</think>"
_GEMMA_CHAN_OPEN = "<|channel>"
_GEMMA_CHAN_CLOSE = "<channel|>"
_EOT_RE = re.compile(r"<\|?(?:eot_id|end_of_text|im_end|turn\|)>|</s>|<\|endoftext\|>")


def _clean_tail(s: str) -> str:
    """Drop trailing end-of-turn / eos markers a non-skip decode leaves behind."""
    return _EOT_RE.split(s)[0].strip()


def split_thinking(text: str, model_type: str) -> tuple[str, str]:
    """Return (thinking, answer) for a raw generated completion.

    - <think> models (deepseek/qwq/qwen3_5/glm4_moe_lite): split on </think>.
    - gemma4: reasoning lives in <|channel>...<channel|> segments; the answer is
      whatever follows the last closed channel (strip_thinking semantics).
    """
    if model_type == "gemma4":
        if _GEMMA_CHAN_CLOSE in text:
            head, tail = text.rsplit(_GEMMA_CHAN_CLOSE, 1)
            # thinking = the channel-thought content (strip the open marker)
            thinking = head.split(_GEMMA_CHAN_OPEN, 1)[-1]
            thinking = thinking.replace("thought", "", 1) if thinking.lstrip().startswith("thought") else thinking
            return _clean_tail(thinking), _clean_tail(tail)
        # no closed channel -> treat all as thinking
        return _clean_tail(text), ""
    # default <think> family
    if _THINK_CLOSE in text:
        head, tail = text.split(_THINK_CLOSE, 1)
        return _clean_tail(head), _clean_tail(tail)
    return _clean_tail(text), ""
