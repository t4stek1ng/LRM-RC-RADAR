"""Extract a single, layer-averaged head-aggregated ``(seq, seq)`` attention
matrix per sample.

Companion to ``extract_hf_attention`` — same eager-attention + forward-hook
mechanism, but instead of persisting one ``attn_hf_L<i>.npy`` per layer we
accumulate every layer's head-aggregated matrix into a single fp32 buffer on
the fly and divide by ``n_layers`` once the forward pass finishes. Output is
one ``attn_layer_avg.npy`` (fp16) per sample. This collapses storage from
~``n_layers ×`` per-sample down to one matrix, which is what the cross-model
trajectory experiment actually consumes — per-layer matrices were thrown
away after averaging anyway.

Output layout::
    <cache-dir>/<category>_<id>/
        meta.json
        attn_layer_avg.npy        # (seq_len, seq_len) fp16
        step_token_text.json

Usage::
    PYTHONPATH=. python -m recur_code.src.analysis.extract_layer_avg_attention \\
        --model $RC_MODELS_ROOT/DeepSeek-R1-Distill-Qwen-14B \\
        --records reasoning_trajectory/.../concise_reasoning.jsonl ... \\
        --ids 0 1 ... \\
        --output-dir exp/hf_attention_cache_total/<model> \\
        --max-seq-len 4096
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .extract_hf_attention import HEAD_AGG_CHOICES, _aggregate_heads
from .extract_step_attention import (
    find_prompt_step_positions,
    find_step_end_positions,
    get_decoder_layers,
    load_record,
)


def _make_accumulate_hook(
    state: dict[str, Any],
    layer_idx: int,
    head_agg: str,
):
    """Forward hook on ``self_attn``: head-aggregate, fold into a single
    running fp32 sum on CPU, and null out the GPU attention tensor so HF
    doesn't pin one per layer."""

    def hook(_module, _inputs, output):
        if not isinstance(output, tuple) or len(output) < 2:
            return output
        attn_w = output[1]
        if attn_w is None:
            return output
        agg = _aggregate_heads(attn_w.detach(), head_agg)  # (seq, seq) on GPU
        agg_cpu = agg.to("cpu", dtype=torch.float32).numpy()
        if state["sum"] is None:
            state["sum"] = agg_cpu
        else:
            if state["sum"].shape != agg_cpu.shape:
                raise RuntimeError(
                    f"layer {layer_idx}: shape {agg_cpu.shape} "
                    f"!= prior {state['sum'].shape}"
                )
            state["sum"] += agg_cpu
        state["seen"].add(layer_idx)
        return (output[0], None) + tuple(output[2:])

    return hook


@torch.no_grad()
def extract_one_sample(
    model: Any,
    tokenizer: Any,
    record: dict[str, Any],
    output_dir: Path,
    max_seq_len: int,
    head_agg: str,
) -> dict[str, Any]:
    category = record.get("category", "sample")
    sample_id = record["id"]
    user_content = record.get("question") or record.get("prompt") or ""
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    thinking = record["thinking"] or ""

    full_text = prompt_text + thinking
    enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids.to(model.device)
    seq_len = input_ids.shape[1]
    if seq_len > max_seq_len:
        input_ids = input_ids[:, :max_seq_len]
        seq_len = max_seq_len

    prompt_ids = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False
    ).input_ids
    prompt_len = prompt_ids.shape[1]

    flat_ids = input_ids.squeeze(0).tolist()
    step_positions, step_texts = find_step_end_positions(
        flat_ids, tokenizer, prompt_len
    )
    prompt_step_positions, _ = find_prompt_step_positions(
        flat_ids, tokenizer, prompt_len
    )

    n_layers = model.config.num_hidden_layers
    decoder_layers = get_decoder_layers(model)

    state: dict[str, Any] = {"sum": None, "seen": set()}
    handles = [
        decoder_layers[L].self_attn.register_forward_hook(
            _make_accumulate_hook(state, L, head_agg)
        )
        for L in range(n_layers)
    ]
    try:
        _ = model(input_ids, output_attentions=True)
    finally:
        for h in handles:
            h.remove()
    torch.cuda.empty_cache()

    if len(state["seen"]) != n_layers or state["sum"] is None:
        raise SystemExit(
            f"hook captured {len(state['seen'])} layers; expected {n_layers}. "
            f"Check attn_implementation='eager' and HF version returns "
            f"attn_weights at output index 1."
        )

    avg = state["sum"] / float(n_layers)
    sample_dir = output_dir / f"{category}_{sample_id}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    np.save(sample_dir / "attn_layer_avg.npy", avg.astype(np.float16))

    meta = {
        "id": sample_id,
        "category": category,
        "model_name": record.get("model_name"),
        "prompt_len": int(prompt_len),
        "thinking_len": int(seq_len - prompt_len),
        "seq_len": int(seq_len),
        "step_positions": [int(p) for p in step_positions],
        "n_steps": len(step_positions),
        "prompt_step_positions": [int(p) for p in prompt_step_positions],
        "n_layers": int(n_layers),
        "n_heads": int(model.config.num_attention_heads),
        "head_agg": head_agg,
        "layer_agg": "mean",
        "averaged_layers": sorted(state["seen"]),
    }
    with open(sample_dir / "meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(sample_dir / "step_token_text.json", "w") as f:
        json.dump(step_texts, f, ensure_ascii=False, indent=2)

    print(
        f"[layer-avg-attn] {category} id={sample_id}: "
        f"prompt_len={prompt_len} seq_len={seq_len} "
        f"n_steps={len(step_positions)} layers={n_layers} "
        f"head_agg={head_agg}"
    )
    return meta


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True)
    p.add_argument(
        "--records", nargs="+", required=True,
        help="JSONL files; each provides one sample (paired with --ids)",
    )
    p.add_argument(
        "--ids", nargs="+", type=int, required=True,
        help="Sample id within each --records file",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=Path("recur_code/exp/hf_attention_cache_total"),
    )
    p.add_argument("--gpu", default="0")
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument(
        "--head-agg", choices=HEAD_AGG_CHOICES, default="mean",
        help="How to collapse the head dim into a single (seq, seq) matrix",
    )
    args = p.parse_args()

    if len(args.records) != len(args.ids):
        raise SystemExit("--records and --ids must have equal length")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[layer-avg-attn] loading {args.model} on cuda (visible={args.gpu})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="cuda:0",
    )
    model.eval()
    print(
        f"[layer-avg-attn] model loaded; "
        f"layers={model.config.num_hidden_layers} "
        f"heads={model.config.num_attention_heads}"
    )

    for rec_path, sample_id in zip(args.records, args.ids):
        record = load_record(Path(rec_path), sample_id)
        if "category" not in record:
            stem = Path(rec_path).stem
            for suffix in ("_llama8b", "_qwen14b"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            record["category"] = stem
        extract_one_sample(
            model, tokenizer, record,
            args.output_dir, args.max_seq_len, args.head_agg,
        )

    print(f"[layer-avg-attn] done. cache dir = {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
