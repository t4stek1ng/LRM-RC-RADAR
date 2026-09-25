"""Which layer's attention matrix is closest to the layer average?

For each sample we run one forward pass with the chunked-eager monkeypatch of
``extract_layer_avg_attention_v2`` (architecture-agnostic: it patches the
model's own ``eager_attention_forward`` primitive), but instead of folding
every layer into a running sum we KEEP each layer's head-averaged (seq, seq)
matrix. Then, at every detection checkpoint ``L``, we take the top-left
``L x L`` block of each layer (causal attention at position q < L does not
depend on tokens after L, so the block is exactly the attention the model
would have produced had generation stopped at L), form the layer average
``Abar_L``, and rank layers by the least-squares distance

    d_l(L) = || A_l[:L,:L] - Abar_L ||_F^2 .

Note on the leave-one-out concern: Abar contains A_l itself with weight 1/L.
Removing it gives Abar_{-l} = (L*Abar - A_l)/(L-1), and

    A_l - Abar_{-l} = L/(L-1) * (A_l - Abar)

which is a CONSTANT rescale, identical for every layer. So the leave-one-out
correction cannot change the ranking, only the absolute numbers. We report the
plain distance.

Everything else (sink handling, head aggregation, loop truncation) is left
exactly as the existing pipeline does it — this script only measures.

## 全部层的总排序（不只是 top-8）

每条样例、每个检查点各给一份排名，`aggregate()` 把它们汇成**一张覆盖所有层的
总排序**：主口径是**平均秩**（对检查点之间两个数量级的距离尺度差免疫，也是当年
聚合出 `CLOSEST8_LLAMA8B` 的口径），对照口径是**平均相对距离** d_l/‖Ābar‖²，
另外报每层的注意力汇占比。结果写进输出 json 的 `_aggregate` 键。
`--aggregate <json>` 可以对任何一份已有结果重算排序，不加载模型、不占 GPU。

## 多模型

`--model` 收模型名或路径，轨迹目录按同名解析。样例默认由 `pick_samples` 按
「每类真实 prompt+thinking 长度的 p25/p50/p90」自动选，因此不必为每个模型手写
清单；`--samples v3` 用回 Llama-8B 那份历史清单以复现旧结果。

混合注意力架构只对真正产出 (seq,seq) 矩阵的层排序——Qwen3.6 的 64 层里 48 层是
线性注意力，不走 eager 路径，也就没有可比的矩阵；层平均同样只在这些层上取。

Run (Recur2 env, from repo root):
    PYTHONPATH=. /root/miniconda3/envs/Recur2/bin/python \
        recur_code/script/experiments/layer_ls_closest.py \
        --model DeepSeek-R1-Distill-Llama-8B --gpu 1
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
RECUR_CODE = THIS_DIR.parents[1]
REPO_ROOT = RECUR_CODE.parent
sys.path.insert(0, str(THIS_DIR))

# The detection pipeline's ladder is {256..4096}; 128 is added here only so the
# short end of the concise class (class median ~166 thinking tokens) yields at
# least one checkpoint instead of an all-blank row.
DEFAULT_CHECKPOINTS = [128, 256, 512, 1024, 2048, 4096]
MODELS_ROOT = Path(os.environ.get("RC_MODELS_ROOT", "/root/project/models"))
TRAJ_ROOT = RECUR_CODE / "reasoning_trajectory"
CACHE_ROOT = RECUR_CODE / "exp" / "hf_attention_cache_total"

# 四类各取三条，取在**该类自己**思考长度的 p25/p50/p90 上。刻意不取「最长的几条」：
# 按长度挑会让简洁类的代表变成全类最不简洁的那条（Llama-8B 上该类最长 4832 token，
# 中位只有 166）。没跑到某个检查点的样例，那一格留空即可。
CLASS_FILES = [
    ("concise_reasoning",     "from_total/concise_reasoning.jsonl"),
    ("productive_reflection", "from_total/productive_reflection.jsonl"),
    ("repetitive_reasoning",  "from_total/repetitive_reasoning.jsonl"),
    ("repetitive_string",     "from_loopllm/repetitive_string.jsonl"),
]
PERCENTILES = (25, 50, 90)

# 历史留痕：v3（`exp/suppression/layer_ls_closest_v3.json`，硬编码常量
# CLOSEST8_LLAMA8B 就是从它聚合出来的）用的 Llama-8B 样例。现在样例由
# `pick_samples` 按同一条规则自动选，这份清单只用于复现旧结果（`--samples v3`）。
V3_LLAMA8B_SAMPLES = [
    ("concise_reasoning", 9), ("concise_reasoning", 28), ("concise_reasoning", 0),
    ("productive_reflection", 180), ("productive_reflection", 137),
    ("productive_reflection", 10009),
    ("repetitive_reasoning", 224), ("repetitive_reasoning", 228),
    ("repetitive_reasoning", 10102),
    ("repetitive_string", 1200), ("repetitive_string", 208),
    ("repetitive_string", 2103),
]


def true_token_len(tokenizer, rec, model_type) -> int:
    """重新分词得到的 prompt+thinking 真实长度。

    不能用记录里的 `thinking_tokens`：循环字符类的循环发生在字符级，重新编码时
    BPE 会把它合并回去（Llama-8B id=15：16465 字符的 "* * * ..." 只剩 161 token，
    而 `thinking_tokens` 写着 8192）。选样例必须按真实长度，否则该类的
    p25/p50/p90 会挑错。
    """
    import _hf_backend as B
    text = (B.build_prompt_text(
        tokenizer, rec.get("prompt") or rec.get("question") or "", model_type)
        + (rec.get("thinking") or ""))
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def pick_samples(traj_dir: Path, tokenizer, model_type, n_per_class: int,
                 verbose: bool = True) -> list[tuple[str, int, Path]]:
    """每类按真实长度的 p25/p50/p90 取样例；不足的类有几条取几条。

    小类（GLM 的循环推理只有 1 条、Qwen3.6 的循环字符只有 1 条）会让同一个分位
    重复命中同一条，这里按已取集合向后（再向前）顺延，保证取到的是互不相同的样例。
    """
    picked: list[tuple[str, int, Path]] = []
    for category, rel in CLASS_FILES:
        path = traj_dir / rel
        if not path.exists():
            if verbose:
                print(f"[ls]   {category:<22} 缺文件 {rel}，跳过")
            continue
        rows = [json.loads(l) for l in open(path) if l.strip()]
        if not rows:
            if verbose:
                print(f"[ls]   {category:<22} 空文件，跳过")
            continue
        lens = sorted((true_token_len(tokenizer, r, model_type), r["id"]) for r in rows)
        chosen: list[tuple[int, int, int]] = []
        seen: set[int] = set()
        for q in PERCENTILES[:n_per_class]:
            i = min(len(lens) - 1, max(0, int(round(q / 100 * (len(lens) - 1)))))
            for j in list(range(i, len(lens))) + list(range(i - 1, -1, -1)):
                if lens[j][1] not in seen:
                    seen.add(lens[j][1])
                    chosen.append((lens[j][1], lens[j][0], q))
                    break
        for sid, _tok, _q in chosen:
            picked.append((category, sid, path))
        if verbose:
            desc = ", ".join(f"id={s}(p{q}, {t}tok)" for s, t, q in chosen)
            print(f"[ls]   {category:<22} {len(rows):>4} 条 → {desc}")
    return picked


def install_per_layer_eager(model_type: str, store: dict, head_agg: str, q_chunk: int):
    """Same chunked-eager patch as extract_layer_avg_attention_v2, except the
    head-aggregated matrix is stored PER LAYER instead of accumulated."""
    import torch

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
        agg_buf = torch.zeros((seq_len, seq_len), device=query.device, dtype=torch.float32)
        for i0 in range(0, seq_len, q_chunk):
            i1 = min(i0 + q_chunk, seq_len)
            scores = torch.matmul(query[:, :, i0:i1, :], k_t) * scaling
            if attention_mask is not None:
                scores = scores + attention_mask[..., i0:i1, : scores.shape[-1]]
            else:
                rows = torch.arange(i0, i1, device=query.device).unsqueeze(1)
                cols = torch.arange(scores.shape[-1], device=query.device).unsqueeze(0)
                scores = scores.masked_fill((cols > rows).unsqueeze(0).unsqueeze(0), float("-inf"))
            attn_w = torch.softmax(scores.float(), dim=-1).to(query.dtype)
            attn_output[:, :, i0:i1, :] = torch.matmul(attn_w, value)
            if head_agg == "mean":
                agg = attn_w.float().mean(dim=1)
            elif head_agg == "sum":
                agg = attn_w.float().sum(dim=1)
            else:
                agg = attn_w.float().amax(dim=1)
            agg_buf[i0:i1, :] = agg[0]
            del scores, attn_w, agg
        layer_idx = getattr(module, "layer_idx", None)
        store[int(layer_idx)] = agg_buf.to("cpu").numpy().astype(np.float16)
        del agg_buf
        return attn_output.transpose(1, 2).contiguous(), None

    mod.eager_attention_forward = patched
    return mod, orig


def capture_sample(model, tokenizer, model_type, record, max_seq_len, max_iters,
                   head_agg, q_chunk, loop_trunc=True):
    """One forward pass -> {layer_idx: (seq,seq) fp16}, prompt_len, seq_len."""
    import torch
    from recur_code.src.analysis.extract_layer_avg_attention_v2 import (
        _truncated_thinking, _text_config,
    )
    import _hf_backend as B

    user_content = record.get("question") or record.get("prompt") or ""
    prompt_text = B.build_prompt_text(tokenizer, user_content, model_type)
    thinking_full = record.get("thinking") or ""
    is_loop_sample = (loop_trunc and record.get("category") == "repetitive_reasoning"
                      and bool(record.get("loop")))
    thinking_used, loop_info = (_truncated_thinking(thinking_full, max_iters)
                               if is_loop_sample else (thinking_full, {"loop_detected": False}))

    enc = tokenizer(prompt_text + thinking_used, return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids.to(model.device)
    if input_ids.shape[1] > max_seq_len:
        input_ids = input_ids[:, :max_seq_len]
    seq_len = input_ids.shape[1]
    prompt_len = tokenizer(prompt_text, return_tensors="pt",
                           add_special_tokens=False).input_ids.shape[1]

    store: dict[int, np.ndarray] = {}
    mod, orig = install_per_layer_eager(model_type, store, head_agg, q_chunk)
    try:
        with torch.no_grad():
            model(input_ids, output_attentions=False)
    finally:
        mod.eager_attention_forward = orig
    torch.cuda.empty_cache()
    return store, int(prompt_len), int(seq_len), loop_info


def analyse(store: dict[int, np.ndarray], checkpoints: list[int], seq_len: int):
    """Least-squares distance of every layer to the layer average, per checkpoint."""
    layers = sorted(store)
    out = {}
    for L in checkpoints:
        if L > seq_len:
            continue
        acc = np.zeros((L, L), dtype=np.float64)
        for l in layers:
            acc += store[l][:L, :L].astype(np.float64)
        abar = acc / len(layers)
        abar_norm2 = float((abar ** 2).sum())
        d = {}
        sink = {}
        for l in layers:
            a = store[l][:L, :L].astype(np.float64)
            d[l] = float(((a - abar) ** 2).sum())
            sink[l] = float(a[:, 0].mean())
        order = sorted(layers, key=lambda l: d[l])
        out[L] = {
            "distances": {str(l): d[l] for l in layers},
            "relative": {str(l): d[l] / abar_norm2 for l in layers},
            "sink_share": {str(l): sink[l] for l in layers},
            "ranking": order,
            "closest_layer": order[0],
            "abar_norm2": abar_norm2,
        }
    return out


def aggregate(results: dict) -> dict:
    """把逐样例逐检查点的排名汇成**全部层**的一张总排序。

    主口径是**平均秩**（每组排名里的位次取平均）。选它而不是直接平均距离，是因为
    绝对距离在不同检查点之间差两个数量级（L 越大矩阵越大，平方和越大），直接平均
    会让长序列的样例独占话语权；平均秩对这个尺度差免疫，也是当年聚合出
    `CLOSEST8_LLAMA8B` 的口径。

    同时报**平均相对距离** d_l / ‖Ābar‖²（本身已按层平均的模长归一）作为对照口径，
    以及每层的**注意力汇占比**（首位 key 上的平均质量）。两个口径给出同一头部，
    结论才算稳；不一致的地方要单独看。

    混合注意力模型只对**真正产出 (seq,seq) 矩阵的层**排序：Qwen3.6 的 64 层里
    48 层是线性注意力，压根不调用 eager 路径，也就没有可比的矩阵。`layers_ranked`
    记下实际参与排序的层，`n_layers_ranked` 与模型层数不等时就是这个原因。
    """
    import statistics
    from collections import defaultdict

    ranks: dict[int, list[int]] = defaultdict(list)
    rel: dict[int, list[float]] = defaultdict(list)
    sink: dict[int, list[float]] = defaultdict(list)
    n_groups = 0
    for _name, rec in results.items():
        for _L, r in rec["checkpoints"].items():
            order = r.get("ranking") or []
            if not order:
                continue
            n_groups += 1
            for pos, l in enumerate(order):
                ranks[int(l)].append(pos)
            for l, v in r["relative"].items():
                rel[int(l)].append(float(v))
            for l, v in r["sink_share"].items():
                sink[int(l)].append(float(v))

    rows = [{
        "layer": l,
        "mean_rank": statistics.mean(ranks[l]),
        "median_rank": statistics.median(ranks[l]),
        "mean_relative_distance": statistics.mean(rel[l]) if rel[l] else None,
        "mean_sink_share": statistics.mean(sink[l]) if sink[l] else None,
        "n_groups": len(ranks[l]),
    } for l in sorted(ranks)]

    by_rank = sorted(rows, key=lambda r: r["mean_rank"])
    by_rel = sorted(rows, key=lambda r: (r["mean_relative_distance"] is None,
                                         r["mean_relative_distance"]))
    for pos, r in enumerate(by_rank):
        r["order_by_mean_rank"] = pos
    for pos, r in enumerate(by_rel):
        r["order_by_relative_distance"] = pos
    return {
        "n_samples": len(results),
        "n_rank_groups": n_groups,
        "n_layers_ranked": len(rows),
        "layers_ranked": [r["layer"] for r in rows],
        "ranking_by_mean_rank": [r["layer"] for r in by_rank],
        "ranking_by_relative_distance": [r["layer"] for r in by_rel],
        "per_layer": rows,
    }


def print_ranking(agg: dict, title: str) -> None:
    """全部层的总排序表——不截断，不只报 top-8。"""
    print(f"\n=== {title}：全部 {agg['n_layers_ranked']} 层与层平均的距离总排序 "
          f"（{agg['n_samples']} 条样例 × 检查点 = {agg['n_rank_groups']} 组排名）===")
    print(f"{'名次':>4} {'层':>4} {'平均秩':>8} {'中位秩':>7} "
          f"{'平均相对距离':>13} {'注意力汇占比':>13} {'相对距离口径名次':>16}")
    rel_pos = {r["layer"]: r["order_by_relative_distance"] for r in agg["per_layer"]}
    for r in sorted(agg["per_layer"], key=lambda x: x["order_by_mean_rank"]):
        rd = r["mean_relative_distance"]
        sk = r["mean_sink_share"]
        print(f"{r['order_by_mean_rank'] + 1:>4} {r['layer']:>4} "
              f"{r['mean_rank']:>8.2f} {r['median_rank']:>7.1f} "
              f"{(f'{rd:.4f}' if rd is not None else '-'):>13} "
              f"{(f'{sk:.4f}' if sk is not None else '-'):>13} "
              f"{rel_pos[r['layer']] + 1:>16}")


def internal_check(store, n_layers_expected):
    """Self-contained validity of the capture: every layer seen exactly once and
    every attention row sums to 1. Needs no external file, so it is unaffected by
    the id drift between today's jsonl and the older attention cache."""
    layers = sorted(store)
    rowsum_err = 0.0
    for l in layers:
        rs = store[l].astype(np.float64).sum(axis=1)
        rowsum_err = max(rowsum_err, float(np.abs(rs - 1.0).max()))
    return {"n_layers": len(layers), "expected": int(n_layers_expected),
            "contiguous": layers == list(range(len(layers))),
            "max_rowsum_error": rowsum_err}


def compare_to_cache_by_sha1(store, prompt_sha1, seq_len, cache_dir: Path):
    """Informational only. The Llama attention cache predates the current id
    assignment, so we look the sample up by prompt_sha1, not by id. Even then the
    cached run used an older chat-template rendering (different prompt_len), so a
    mismatch here is input drift, not a capture bug."""
    if not prompt_sha1:
        return {"checked": False, "reason": "record has no prompt_sha1"}
    hit = None
    if not cache_dir.exists():
        return {"checked": False, "reason": f"no attention cache at {cache_dir}"}
    for meta_path in cache_dir.glob("*/meta.json"):
        try:
            m = json.load(open(meta_path))
        except Exception:
            continue
        if m.get("prompt_sha1") == prompt_sha1:
            hit = (meta_path.parent, m)
            break
    if hit is None:
        return {"checked": False, "reason": "no cached sample with this prompt_sha1"}
    d, m = hit
    cached = np.load(d / "attn_layer_avg.npy").astype(np.float32)
    n = min(seq_len, cached.shape[0], 512)
    acc = np.zeros((n, n), dtype=np.float64)
    for l in store:
        acc += store[l][:n, :n].astype(np.float64)
    diff = np.abs(acc / len(store) - cached[:n, :n].astype(np.float64))
    return {"checked": True, "cache_dir": d.name, "n": int(n),
            "cached_prompt_len": m.get("prompt_len"),
            "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean())}


def resolve_model(spec: str) -> tuple[Path, str]:
    """`--model` 允许写模型名或完整路径，返回 (模型目录, 模型名)。"""
    p = Path(spec)
    if not p.exists():
        p = MODELS_ROOT / spec
    if not p.exists():
        raise SystemExit(f"找不到模型 {spec}（也不在 {MODELS_ROOT} 下）")
    return p, p.name


def n_hidden_layers(model) -> int:
    """混合架构把层数放在 text_config 里，取不到就退回顶层 config。"""
    cfg = getattr(model.config, "text_config", None) or model.config
    return int(getattr(cfg, "num_hidden_layers", 0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="DeepSeek-R1-Distill-Llama-8B",
                    help="模型名（$RC_MODELS_ROOT 下）或完整路径")
    ap.add_argument("--traj-model", default=None,
                    help="轨迹目录用哪个模型名，默认与 --model 同名")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--checkpoints", type=int, nargs="+", default=DEFAULT_CHECKPOINTS)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--max-iters", type=int, default=5)
    ap.add_argument("--no-loop-trunc", action="store_true",
                    help="feed the FULL stored thinking for looping samples instead of "
                         "cutting repetitive_reasoning to --max-iters loop iterations")
    ap.add_argument("--head-agg", default="mean")
    ap.add_argument("--q-chunk", type=int, default=1024)
    ap.add_argument("--samples", default="auto", choices=("auto", "v3"),
                    help="auto = 每类按真实长度 p25/p50/p90 自动选；"
                         "v3 = 复现 layer_ls_closest_v3.json 的 Llama-8B 清单")
    ap.add_argument("--n-per-class", type=int, default=3)
    ap.add_argument("--aggregate", type=Path, default=None,
                    help="只对已有结果 json 重算总排序，不加载模型、不用 GPU")
    ap.add_argument("--out", type=Path, default=None,
                    help="默认 exp/suppression/layer_distance/<模型名>.json")
    args = ap.parse_args()

    # 纯聚合：不碰 GPU，可对任何一份已有结果重新排序
    if args.aggregate is not None:
        results = json.load(open(args.aggregate))
        results.pop("_aggregate", None)
        agg = aggregate(results)
        print_ranking(agg, args.aggregate.stem)
        return 0

    model_dir, model_name = resolve_model(args.model)
    traj_dir = TRAJ_ROOT / (args.traj_model or model_name)
    cache_dir = CACHE_ROOT / model_name
    out = args.out or (RECUR_CODE / "exp" / "suppression" / "layer_distance"
                       / f"{model_name}.json")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    sys.path.insert(0, str(REPO_ROOT))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from recur_code.src.analysis.extract_step_attention import load_record
    import _hf_backend as B

    model_type = B.model_type_of(str(model_dir))
    print(f"[ls] {model_name} (type={model_type}) on GPU {args.gpu}；"
          f"轨迹 {traj_dir.relative_to(RECUR_CODE)}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    print("[ls] 选样例（每类按真实 prompt+thinking 长度的 p25/p50/p90）：")
    if args.samples == "v3":
        samples = [(c, i, traj_dir / dict(CLASS_FILES)[c]) for c, i in V3_LLAMA8B_SAMPLES]
        print(f"[ls]   固定用 v3 清单，共 {len(samples)} 条")
    else:
        samples = pick_samples(traj_dir, tokenizer, model_type, args.n_per_class)
    if not samples:
        raise SystemExit(f"{model_name} 没有可用轨迹（{traj_dir}）")

    print(f"[ls] loading weights ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), dtype=torch.bfloat16, attn_implementation="eager",
        device_map="cuda:0")
    model.eval()
    n_expected = n_hidden_layers(model)

    results = {}
    for category, sample_id, rec_path in samples:
        rec = load_record(Path(rec_path), sample_id)
        rec.setdefault("category", category)
        print(f"[ls] {category} id={sample_id} ...", flush=True)
        store, prompt_len, seq_len, loop_info = capture_sample(
            model, tokenizer, model_type, rec, args.max_seq_len, args.max_iters,
            args.head_agg, args.q_chunk, loop_trunc=not args.no_loop_trunc)
        icheck = internal_check(store, n_expected)
        ccheck = compare_to_cache_by_sha1(store, rec.get("prompt_sha1"), seq_len,
                                          cache_dir)
        res = analyse(store, args.checkpoints, seq_len)
        results[f"{category}_{sample_id}"] = {
            "category": category, "id": sample_id,
            "prompt_len": prompt_len, "seq_len": seq_len,
            "thinking_tokens_field": rec.get("thinking_tokens"),
            "n_layers": len(store), "loop_truncation": loop_info,
            "internal_check": icheck, "cache_check": ccheck, "checkpoints": res,
        }
        print(f"    prompt_len={prompt_len} seq_len={seq_len} "
              f"layers={icheck['n_layers']}/{icheck['expected']} "
              f"rowsum_err={icheck['max_rowsum_error']:.2e}")
        for L, r in res.items():
            print(f"    L={L:5d}  closest={r['closest_layer']:2d}  "
                  f"top5={r['ranking'][:5]}")
        del store

    agg = aggregate(results)
    agg["model"] = model_name
    agg["model_type"] = model_type
    agg["n_layers_total"] = n_expected
    agg["max_seq_len"] = args.max_seq_len
    agg["checkpoints"] = args.checkpoints
    agg["samples"] = [f"{c}_{i}" for c, i, _ in samples]
    print_ranking(agg, model_name)
    if agg["n_layers_ranked"] != n_expected:
        print(f"\n[ls] 注意：模型共 {n_expected} 层，只有 {agg['n_layers_ranked']} 层"
              f"产出了 (seq,seq) 注意力矩阵并参与排序（混合注意力架构，"
              f"线性注意力层没有可比的矩阵）")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"_aggregate": agg, **results}, f, ensure_ascii=False, indent=2)
    print(f"[ls] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
