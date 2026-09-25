#!/usr/bin/env python3
"""逐层提示侧注意力占比：给动态选层离线确定候选层带。

对每条样例做一次前向（复用 layer_ls_closest 的逐层 chunked-eager 补丁），
在生成位置 q ∈ [P, T) 上逐层统计头平均注意力行的：
  prompt_share          Σ_{k<P}  A[q,k]              含注意力汇
  sink_share            Σ_{k∈汇位置} A[q,k]（汇位置按模型实测给出）
  prompt_share_nosink   Σ_{1≤k<P} A[q,k]             去掉汇
  prompt_share_nonsink  Σ_{1≤k<P} A[q,k] / (1−A[q,0])  非汇质量里提示侧的比例
再按「每类先平均、类间等权」汇成逐层剖面。

样例取 pipeline train records，四类各按 completion_tokens 的 p25/p50/p90 取 3 条。
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
MODELS_ROOT = Path(os.environ.get("RC_MODELS_ROOT", "/root/project/models"))
EXP = REPO / "recur_code/script/experiments"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(EXP))
spec = importlib.util.spec_from_file_location("lsc", EXP / "layer_ls_closest.py")
lsc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lsc)

CLASSES = [("concise_reasoning", "GSM8k"), ("productive_reasoning", "GPQA"),
           ("repetitive_reasoning", "Recur"), ("repetitive_string", "LoopLLM")]
VARIANTS = ["prompt_share", "sink_share", "prompt_share_nosink", "prompt_share_nonsink"]


def pick(model: str, n: int) -> list[dict]:
    out = []
    for cat, sub in CLASSES:
        p = REPO / f"recur_code/exp/detection/pipeline/{model}/train/records/{sub}/{cat}.jsonl"
        rows = [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []
        if not rows:
            print(f"[ps] {model} {cat}: 无样例，跳过", flush=True)
            continue
        rows.sort(key=lambda r: r.get("completion_tokens") or len(r.get("thinking") or ""))
        idx = sorted({min(len(rows) - 1, int(round(q / 100 * (len(rows) - 1)))) for q in (25, 50, 90)})
        for j in range(len(rows)):              # 小类不足 3 个不同分位时顺延补齐
            if len(idx) >= min(n, len(rows)):
                break
            if j not in idx:
                idx.append(j)
        out += [rows[i] for i in sorted(idx)[:n]]
    return out


def layer_stats(store: dict, P: int, T: int, sinks: list[int]) -> dict:
    res = {v: {} for v in VARIANTS}
    for l, a in store.items():
        a = a[P:T, :].astype(np.float64)
        if a.shape[0] == 0:
            continue
        sink = a[:, sinks].sum(axis=1)
        prm = a[:, :P].sum(axis=1)
        nos = prm - sink
        res["prompt_share"][l] = float(prm.mean())
        res["sink_share"][l] = float(sink.mean())
        res["prompt_share_nosink"][l] = float(nos.mean())
        res["prompt_share_nonsink"][l] = float((nos / np.clip(1.0 - sink, 1e-9, None)).mean())
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--models-root", type=Path, default=MODELS_ROOT,
                    help="模型所在目录（也可用环境变量 RC_MODELS_ROOT）；--model 给完整路径时忽略")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--n-per-class", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sink-positions", default="0",
                    help="逗号分隔的注意力汇位置（按模型实测：Llama 0，GLM 0,1，QwQ 1）")
    a = ap.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mdir = Path(a.model) if Path(a.model).is_dir() else a.models_root / a.model
    tok = AutoTokenizer.from_pretrained(mdir, trust_remote_code=True)
    import transformers
    # 4.x 叫 torch_dtype，5.x 改名 dtype；Llama-8B 必须在 4.x 下跑（5.x 的 LlamaTokenizer 会丢空格和换行）
    dkey = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(mdir, attn_implementation="eager", device_map="cuda:0",
                                                 **{dkey: torch.bfloat16})
    probe = "Hello world.\nagain again "
    assert tok.decode(tok(probe, add_special_tokens=False).input_ids) == probe, \
        f"{type(tok).__name__} 分词往返不一致（transformers {transformers.__version__}），换环境再跑"
    model.eval()
    # 按 config.json 取 model_type（与检测提取器一致）：Qwen3.6 经 AutoModelForCausalLM 加载后
    # config.model_type 是 `qwen3_5_text`，没有同名模块
    mt = json.load(open(mdir / "config.json"))["model_type"]
    n_layers = lsc.n_hidden_layers(model)
    samples = pick(a.model, a.n_per_class)
    sinks = [int(x) for x in a.sink_positions.split(",") if x != ""]
    print(f"[ps] {a.model} model_type={mt} layers={n_layers} samples={len(samples)}", flush=True)

    per = []
    for r in samples:
        store, P, T, _ = lsc.capture_sample(model, tok, mt, r, a.max_seq_len, 5, "mean", 1024,
                                            loop_trunc=False)
        if T <= P:
            print(f"[ps]   跳过 {r['category']} id={r['id']}：提示 {P} 已超过长度上限", flush=True)
            continue
        st = layer_stats(store, P, T, sinks)
        per.append({"category": r["category"], "subset": r["subset"], "id": r["id"],
                    "prompt_len": P, "seq_len": T, "n_layers_seen": len(store), "stats": st})
        top = sorted(st["prompt_share"], key=lambda l: -st["prompt_share"][l])[:8]
        print(f"[ps]   {r['category']:22s} id={r['id']} P={P} T={T} 层={len(store)} "
              f"含汇提示占比前8层={sorted(top)}", flush=True)
        del store
        torch.cuda.empty_cache()

    agg = {}
    for v in VARIANTS:
        by_cat = {}
        for cat, _ in CLASSES:
            rows = [p["stats"][v] for p in per if p["category"] == cat]
            if rows:
                ls = sorted(rows[0])
                by_cat[cat] = {l: float(np.mean([x[l] for x in rows])) for l in ls}
        ls = sorted(next(iter(by_cat.values())))
        overall = {l: float(np.mean([c[l] for c in by_cat.values()])) for l in ls}
        agg[v] = {"overall": {str(l): overall[l] for l in ls},
                  "by_category": {c: {str(l): d[l] for l in ls} for c, d in by_cat.items()},
                  "ranking": sorted(ls, key=lambda l: -overall[l])}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": Path(a.model).name, "n_layers": n_layers, "max_seq_len": a.max_seq_len, "sink_positions": sinks,
               "samples": [{k: p[k] for k in ("category", "subset", "id", "prompt_len", "seq_len",
                                               "n_layers_seen")} for p in per],
               "aggregate": agg, "per_sample": per}, open(a.out, "w"), ensure_ascii=False)
    for v in VARIANTS:
        o = agg[v]["overall"]
        print(f"\n[ps] {v}: 前8层={sorted(agg[v]['ranking'][:8])}")
        print("     " + " ".join(f"{l}:{float(o[str(l)]):.3f}" for l in range(n_layers) if str(l) in o))
    print(f"[ps] -> {a.out}")


if __name__ == "__main__":
    main()
