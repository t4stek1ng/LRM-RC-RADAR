"""Architecture-agnostic layer-averaged attention extraction for the 3 new
2026 models (GLM-4.7-Flash / Qwen3.6-27B / Gemma-4-31B-it), run under the
Recur2 (transformers 5.12.1) env.

Why a new script instead of reusing extract_layer_avg_attention[_loop].py:
those scripts' "hook" mode (eager attention + forward hook reading
self_attn's raw output) OOMs on long sequences for 30B+ models, and their
"manual" OOM-safe fallback reimplements RoPE/GQA by hand assuming a plain
Llama/Qwen2-style single q_proj/k_proj — which breaks on GLM-4.7's MLA
(q_lora/kv_lora, no q_proj) and would silently mis-derive attention on
Qwen3.6/Gemma-4 (QK-norm, interleaved RoPE, composite VLM wrapping with the
decoder nested under `model.language_model`).

Instead, this monkeypatches each architecture's own module-level
``eager_attention_forward`` function (the shared low-level "scores = QK^T,
softmax, matmul V" primitive every HF attention class calls into via
``ALL_ATTENTION_FUNCTIONS.get_interface("eager", eager_attention_forward)``
— transformers has no globally-registered "eager" implementation, so that
call always resolves to the calling module's own local function name,
looked up dynamically at call time). Patching only this primitive means
every architecture-specific step upstream of it — Q/K/V projection, LoRA,
QK-norm, RoPE (interleaved or not), composite-model routing — has already
run correctly by the time our code sees (query, key, value); we only need
to redo the O(seq^2) matmul+softmax in query-row chunks (bounding peak
memory to (heads, q_chunk, seq) instead of (heads, seq, seq)) and fold the
head-aggregated result directly into a running accumulator, so the full
per-head attention tensor is never materialized.

Output layout (same as extract_layer_avg_attention.py, so downstream
compute_step_trajectory.py needs no changes)::
    <cache-dir>/<category>_<id>/
        meta.json
        attn_layer_avg.npy        # (seq_len, seq_len) fp16
        step_token_text.json

Usage::
    PYTHONPATH=. /root/miniconda3/envs/Recur2/bin/python \\
        -m recur_code.src.analysis.extract_layer_avg_attention_v2 \\
        --model $RC_MODELS_ROOT/GLM-4.7-Flash \\
        --records reasoning_trajectory/GLM-4.7-Flash/from_total/concise_reasoning.jsonl \\
        --ids 0 1 2 --output-dir exp/hf_attention_cache_total/GLM-4.7-Flash \\
        --max-seq-len 4096
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
RECUR_CODE_DIR = THIS_DIR.parents[1]  # .../recur_code
sys.path.insert(0, str(RECUR_CODE_DIR / "script" / "experiments"))

from .extract_hf_attention import HEAD_AGG_CHOICES
from .step_metrics_from_attention import build_step_metrics
from .extract_step_attention import (
    find_prompt_step_positions,
    find_step_end_positions,
    load_record,
)

import _hf_backend as B  # noqa: E402


def _text_config(model) -> Any:
    return getattr(model.config, "text_config", model.config)


def _install_chunked_eager(model_type: str, state: dict, head_agg: str, q_chunk: int):
    """Monkeypatch <model_type>'s modeling module's eager_attention_forward.
    Returns (module, original_fn) so the caller can restore it afterwards."""
    mod = importlib.import_module(f"transformers.models.{model_type}.modeling_{model_type}")
    orig = mod.eager_attention_forward

    def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        n_rep = getattr(module, "num_key_value_groups", 1) or 1
        if n_rep > 1:
            key = key.repeat_interleave(n_rep, dim=1)
            value = value.repeat_interleave(n_rep, dim=1)
        bsz, n_heads, seq_len, _ = query.shape
        v_dim = value.shape[-1]
        k_t = key.transpose(2, 3)
        attn_output = torch.empty((bsz, n_heads, seq_len, v_dim),
                                   device=query.device, dtype=query.dtype)
        # Accumulate straight into the running sum. A separate per-layer
        # (seq, seq) float32 buffer doubled the largest tensor in play — 1.35 GiB
        # each at 19k tokens — and GLM-4.7-Flash has no headroom for that: its
        # weights take 55.8 GiB and the MoE activations another ~19 GiB at that
        # length, which is what pushed the attack trajectories into OOM.
        if state["sum"] is None:
            state["sum"] = torch.zeros((seq_len, seq_len), device=query.device,
                                       dtype=torch.float32)
        for i0 in range(0, seq_len, q_chunk):
            i1 = min(i0 + q_chunk, seq_len)
            scores = torch.matmul(query[:, :, i0:i1, :], k_t) * scaling
            if attention_mask is not None:
                scores = scores + attention_mask[..., i0:i1, : scores.shape[-1]]
            else:
                rows = torch.arange(i0, i1, device=query.device).unsqueeze(1)
                cols = torch.arange(scores.shape[-1], device=query.device).unsqueeze(0)
                causal = (cols > rows).unsqueeze(0).unsqueeze(0)
                scores = scores.masked_fill(causal, float("-inf"))
            attn_w = torch.softmax(scores.float(), dim=-1).to(query.dtype)
            attn_output[:, :, i0:i1, :] = torch.matmul(attn_w, value)
            if head_agg == "mean":
                agg = attn_w.float().mean(dim=1)
            elif head_agg == "sum":
                agg = attn_w.float().sum(dim=1)
            else:
                agg = attn_w.float().amax(dim=1)
            state["sum"][i0:i1, :] += agg[0]
            del scores, attn_w, agg
        attn_output = attn_output.transpose(1, 2).contiguous()
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is not None:
            state["seen"].add(int(layer_idx))
        return attn_output, None

    mod.eager_attention_forward = patched
    return mod, orig


def _find_loop_blocks(blocks: list[str]) -> tuple[int | None, int | None]:
    n = len(blocks)
    while n > 0 and blocks[n - 1] == "":
        n -= 1
    for i in range(n):
        for L in range(1, (n - i) // 2 + 1):
            if blocks[i:i + L] == blocks[i + L:i + 2 * L]:
                return i, L
    return None, None


def _truncated_thinking(thinking: str, max_iters: int) -> tuple[str, dict[str, Any]]:
    blocks = thinking.split("\n\n")
    loop_start, loop_period = _find_loop_blocks(blocks)
    info: dict[str, Any] = {
        "loop_detected": False, "loop_start_block": None,
        "loop_period_blocks": None, "kept_iterations": None,
        "max_iters_cap": max_iters,
    }
    if loop_start is None:
        return thinking, info
    n = len(blocks)
    while n > 0 and blocks[n - 1] == "":
        n -= 1
    available_iters = max(0, (n - loop_start) // loop_period)
    kept = min(max_iters, available_iters)
    if kept < 1:
        return thinking, info
    end_block_excl = loop_start + kept * loop_period
    truncated = "\n\n".join(blocks[:end_block_excl])
    info.update({
        "loop_detected": True, "loop_start_block": int(loop_start),
        "loop_period_blocks": int(loop_period), "kept_iterations": int(kept),
        "available_iterations": int(available_iters),
    })
    return truncated, info


@torch.no_grad()
def extract_one_sample(
    model: Any, tokenizer: Any, model_type: str, record: dict[str, Any],
    output_dir: Path, max_seq_len: int, max_iters: int, head_agg: str, q_chunk: int,
    metrics_dir: Path | None = None, save_attn: bool = True,
    attn_dtype: str = "fp16", metric_chunk: int = 256,
    max_thinking: int = 8192, include_capped: bool = False,
    nonloop_rep_as_stop: bool = False,
) -> dict[str, Any]:
    category = record.get("category", "sample")
    sample_id = record["id"]
    user_content = record.get("question") or record.get("prompt") or ""
    prompt_text = B.build_prompt_text(tokenizer, user_content, model_type)

    thinking_full = record["thinking"] or ""
    is_loop_sample = category == "repetitive_reasoning" and bool(record.get("loop"))
    if is_loop_sample:
        thinking_used, loop_info = _truncated_thinking(thinking_full, max_iters)
    else:
        thinking_used, loop_info = thinking_full, {
            "loop_detected": False, "loop_start_block": None,
            "loop_period_blocks": None, "kept_iterations": None,
            "max_iters_cap": max_iters,
        }

    full_text = prompt_text + thinking_used
    enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids.to(model.device)
    seq_len = input_ids.shape[1]
    if seq_len > max_seq_len:
        input_ids = input_ids[:, :max_seq_len]
        seq_len = max_seq_len

    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids
    prompt_len = prompt_ids.shape[1]

    flat_ids = input_ids.squeeze(0).tolist()
    step_positions, step_texts = find_step_end_positions(flat_ids, tokenizer, prompt_len)
    prompt_step_positions, _ = find_prompt_step_positions(flat_ids, tokenizer, prompt_len)

    tc = _text_config(model)
    n_layers_total = tc.num_hidden_layers
    n_heads = tc.num_attention_heads
    # Hybrid architectures interleave layer types. Some (e.g. Gemma-4's
    # "sliding_attention") are still genuine quadratic softmax attention —
    # just locally masked — and DO call eager_attention_forward, so they
    # belong in the average. Others (e.g. Qwen3.6's "linear_attention") are
    # recurrent/state-space layers with no (seq,seq) matrix at all and never
    # call it. Rather than guess which named types are quadratic, just trust
    # whatever the monkeypatch actually captures.

    state: dict[str, Any] = {"sum": None, "seen": set()}
    mod, orig = _install_chunked_eager(model_type, state, head_agg, q_chunk)
    t_fwd = time.perf_counter()
    try:
        # Only the attention matrices matter here, and the monkeypatch harvests
        # those inside the decoder. Letting the LM head run would materialise
        # (seq, vocab) float32 logits — 11.5 GiB for GLM at 20k tokens, 18.5 GiB
        # for Qwen3.6, against 1.5 GiB for the layer-average accumulator itself.
        # That is what pushed the 16k attack trajectories into OOM while the
        # short benign ones passed.
        try:
            _ = model(input_ids, output_attentions=False, logits_to_keep=1)
        except TypeError:
            # Older signatures take no such argument; run the decoder directly.
            _ = model.model(input_ids, output_attentions=False)
    finally:
        mod.eager_attention_forward = orig
    torch.cuda.synchronize()
    t_fwd = time.perf_counter() - t_fwd

    if state["sum"] is None or not state["seen"]:
        raise SystemExit(
            "chunked-eager captured no layers at all. "
            "Check attn_implementation='eager' and the monkeypatch target module."
        )

    n_avg_layers = len(state["seen"])
    avg = state["sum"] / float(n_avg_layers)
    sample_dir = output_dir / f"{category}_{sample_id}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Metrics first, while the layer average is still resident. Reading it back
    # off disk is what made the old attention stage slow: 191 s to load and
    # widen a single 648 MB cache, against 25 s for the metric that feeds the
    # classifier. Writing the .npy at all is now opt-in.
    metrics, t_metrics, t_save = None, 0.0, 0.0
    if metrics_dir is not None:
        t_metrics = time.perf_counter()
        tokens = [tokenizer.decode([i], skip_special_tokens=False) for i in flat_ids]
        metrics = build_step_metrics(
            avg, prompt_len, tokens, record, category, sample_id, seq_len,
            attn_dtype=attn_dtype, chunk=metric_chunk, max_thinking=max_thinking,
            include_capped=include_capped, nonloop_rep_as_stop=nonloop_rep_as_stop,
        )
        if metrics is not None:
            metrics_dir.mkdir(parents=True, exist_ok=True)
            with open(metrics_dir / f"{category}_{sample_id}.json", "w") as f:
                json.dump(metrics, f, ensure_ascii=False)
        torch.cuda.synchronize()
        t_metrics = time.perf_counter() - t_metrics

    if save_attn:
        t_save = time.perf_counter()
        np.save(sample_dir / "attn_layer_avg.npy",
                avg.to("cpu").numpy().astype(np.float16))
        t_save = time.perf_counter() - t_save
    del avg, state["sum"]
    state["sum"] = None
    torch.cuda.empty_cache()

    meta = {
        "id": sample_id, "category": category,
        "model_name": record.get("model_name"),
        "prompt_len": int(prompt_len),
        "thinking_len": int(seq_len - prompt_len),
        "seq_len": int(seq_len),
        "step_positions": [int(p) for p in step_positions],
        "n_steps": len(step_positions),
        "prompt_step_positions": [int(p) for p in prompt_step_positions],
        "n_layers": int(n_avg_layers), "n_layers_total": int(n_layers_total),
        "n_heads": int(n_heads),
        "head_agg": head_agg, "layer_agg": "mean",
        "averaged_layers": sorted(state["seen"]),
        "extract_mode": "chunked_eager_monkeypatch",
        "loop_truncation": loop_info,
    }
    with open(sample_dir / "meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(sample_dir / "step_token_text.json", "w") as f:
        json.dump(step_texts, f, ensure_ascii=False, indent=2)

    print(
        f"[layer-avg-attn-v2] {category} id={sample_id}: "
        f"prompt_len={prompt_len} seq_len={seq_len} n_steps={len(step_positions)} "
        f"layers={n_avg_layers}/{n_layers_total} loop={loop_info['loop_detected']} "
        f"kept_iters={loop_info.get('kept_iterations')} "
        f"forward={t_fwd:.1f}s metrics={t_metrics:.1f}s save_npy={t_save:.1f}s "
        f"({'fused' if metrics is not None else 'attn-only'})"
    )
    return meta


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--records", nargs="+", required=True)
    p.add_argument("--ids", nargs="+", type=int, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("recur_code/exp/hf_attention_cache_total"))
    p.add_argument("--gpu", default="0")
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--max-iters", type=int, default=5,
                   help="loop-truncation cap for repetitive_reasoning samples with loop=true")
    p.add_argument("--head-agg", choices=HEAD_AGG_CHOICES, default="mean")
    p.add_argument("--q-chunk", type=int, default=1024)
    p.add_argument("--metrics-dir", type=Path, default=None,
                   help="同时算逐位置指标并写 <metrics-dir>/<类别>_<id>.json，"
                        "省掉「落盘 (seq,seq) 注意力 → 读回 → 再算」的往返")
    p.add_argument("--save-attn", dest="save_attn", action="store_true", default=None,
                   help="即使给了 --metrics-dir 也仍然落盘 attn_layer_avg.npy（默认只在没给时落盘）")
    p.add_argument("--no-save-attn", dest="save_attn", action="store_false")
    p.add_argument("--attn-dtype", choices=["fp16", "fp32"], default="fp16",
                   help="算指标前把层平均量化到的精度。fp16 复现落盘路径的精度，"
                        "指标与既有 step_metrics 可比；fp32 更准但与历史数据不可混用")
    p.add_argument("--metric-chunk", type=int, default=256, help="指标计算的查询行分块")
    p.add_argument("--max-thinking", type=int, default=8192)
    p.add_argument("--include-capped", action="store_true")
    p.add_argument("--nonloop-rep-as-stop", action="store_true")
    args = p.parse_args()
    if args.save_attn is None:
        args.save_attn = args.metrics_dir is None

    if len(args.records) != len(args.ids):
        raise SystemExit("--records and --ids must have equal length")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_type = B.model_type_of(args.model)
    print(f"[layer-avg-attn-v2] loading {args.model} (model_type={model_type}) on cuda (visible={args.gpu})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="cuda:0",
    )
    model.eval()
    tc = _text_config(model)
    print(f"[layer-avg-attn-v2] model loaded; layers={tc.num_hidden_layers} heads={tc.num_attention_heads}")

    for rec_path, sample_id in zip(args.records, args.ids):
        record = load_record(Path(rec_path), sample_id)
        if "category" not in record:
            record["category"] = Path(rec_path).stem
        extract_one_sample(
            model, tokenizer, model_type, record,
            args.output_dir, args.max_seq_len, args.max_iters, args.head_agg, args.q_chunk,
            metrics_dir=args.metrics_dir, save_attn=args.save_attn,
            attn_dtype=args.attn_dtype, metric_chunk=args.metric_chunk,
            max_thinking=args.max_thinking, include_capped=args.include_capped,
            nonloop_rep_as_stop=args.nonloop_rep_as_stop,
        )

    print(f"[layer-avg-attn-v2] done. cache dir = {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
