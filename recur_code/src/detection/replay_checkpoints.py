"""End-to-end detection replay — run the DEPLOYED protocol (design §2.2) over
already-recorded trajectories with a SAVED classifier bundle.

Difference from training-time scoring:
  - The training driver scores rows at the dense probe positions the classifier
    was fitted on; that is not the deployment loop.
  - This script loads ONE frozen bundle (`--model-bundle`), walks each
    trajectory checkpoint by checkpoint (L = 256…4096) exactly as the online
    runtime does, applies the τ_prob gate, stops the walk at the first
    confident attack call, and reports what that policy buys: per-checkpoint
    accuracy/coverage, attack trigger recall, benign false-trigger rate, how
    early the trigger fires, and how many tokens early stopping saves.

By default it evaluates ONLY the trajectories held out when the bundle was
saved (`test_group_ids` inside the bundle), so the numbers are generalization
numbers. `--all-trajectories` lifts that.

Run (sklearn lives in the base env):
    /root/miniconda3/bin/python recur_code/src/detection/replay_checkpoints.py \
        --model-bundle recur_code/exp/detection_dataset/models/loop_detector_llama8b_4class_method2.joblib \
        --traj-dirs recur_code/exp/step_trajectory_total \
        --model DeepSeek-R1-Distill-Llama-8B
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import (ATTACK, CHECKPOINTS, LABELS, DEFAULT_TAU_PROB,  # noqa: E402
                      load_detector, load_trajectories, replay_trace)


def per_checkpoint_report(rows, tau, max_tokens):
    """rows: list of (true_label, call). One line per checkpoint L + ALL."""
    print(f"\n  --- 逐 checkpoint（τ_prob={tau}）---")
    print(f"  {'L':>6} {'eligible':>9} {'decided':>8} {'coverage':>9} "
          f"{'4way acc':>9} {'AxisA rec':>10} {'AxisA spec':>11}")
    for L in CHECKPOINTS + [None]:
        sub = rows if L is None else [r for r in rows if r[1].L == L]
        if not sub:
            continue
        dec = [r for r in sub if r[1].decided]
        cov = len(dec) / len(sub)
        acc = np.mean([t == c.pred for t, c in dec]) if dec else float("nan")
        atk = [r for r in dec if r[0] in ATTACK]
        ben = [r for r in dec if r[0] not in ATTACK]
        rec = np.mean([c.pred in ATTACK for _, c in atk]) if atk else float("nan")
        spec = np.mean([c.pred not in ATTACK for _, c in ben]) if ben else float("nan")
        tag = "ALL" if L is None else str(L)
        print(f"  {tag:>6} {len(sub):>9} {len(dec):>8} {cov:>9.3f} {acc:>9.3f} "
              f"{rec:>10.3f} {spec:>11.3f}")

    dec = [r for r in rows if r[1].decided]
    if not dec:
        return
    idx = {l: i for i, l in enumerate(LABELS)}
    cm = np.zeros((4, 4), int)
    for t, c in dec:
        cm[idx[t], idx[c.pred]] += 1
    print(f"\n  checkpoint 混淆矩阵 (decided only, n={len(dec)})  行=真 列=判")
    print(" " * 24 + "".join(f"{l[:4]:>7}" for l in LABELS))
    for i, l in enumerate(LABELS):
        print(f"    {l:18s}" + "".join(f"{cm[i, j]:>7}" for j in range(4))
              + f"   (n={cm[i].sum()})")


def trajectory_report(results, max_tokens):
    """results: list of dicts with label / trace / prompt_len / n_gen."""
    print("\n  --- 轨迹级：首次确信攻击即触发（在线早停策略）---")
    print("  「无探测点」= 序列太短，5 个 checkpoint 一个都没到（部署中这类样本"
          "在首次探测前就自然结束，本就无需判定）。")
    print(f"  {'真类':<22} {'n':>4} {'无探测点':>9} {'触发率':>8} {'触发L中位':>10} "
          f"{'判对类':>8} {'未判定':>7}")
    by_label = defaultdict(list)
    for r in results:
        by_label[r["label"]].append(r)
    for l in LABELS:
        rs = by_label.get(l, [])
        if not rs:
            continue
        trig = [r for r in rs if r["trace"].trigger is not None]
        rate = len(trig) / len(rs)
        med = (np.median([r["trace"].trigger.L for r in trig])
               if trig else float("nan"))
        correct = np.mean([r["trace"].verdict == l for r in rs])
        undec = np.mean([r["trace"].verdict is None for r in rs])
        noprobe = np.mean([not r["trace"].calls for r in rs])
        print(f"  {l:<22} {len(rs):>4} {noprobe:>9.3f} {rate:>8.3f} {med:>10.0f} "
              f"{correct:>8.3f} {undec:>7.3f}")

    atk = [r for r in results if r["label"] in ATTACK]
    ben = [r for r in results if r["label"] not in ATTACK]
    if atk:
        trig = [r for r in atk if r["trace"].trigger is not None]
        print(f"\n  攻击轨迹 触发召回 = {len(trig)}/{len(atk)} = "
              f"{len(trig)/len(atk):.3f}")
        if trig:
            Ls = np.array([r["trace"].trigger.L for r in trig], float)
            seq = np.array([r["prompt_len"] + r["n_gen"] for r in trig], float)
            saved = np.clip(np.minimum(seq, max_tokens) - Ls, 0, None)
            print(f"    触发长度 L: min={Ls.min():.0f} 中位={np.median(Ls):.0f} "
                  f"max={Ls.max():.0f}")
            print(f"    相对该轨迹实际长度省下 token: 中位={np.median(saved):.0f} "
                  f"(占比 中位={np.median(saved/np.minimum(seq, max_tokens)):.3f})")
            print(f"    以 max_tokens={max_tokens} 计省下: 中位="
                  f"{np.median(np.clip(max_tokens - Ls, 0, None)):.0f} "
                  f"({np.median(np.clip(max_tokens - Ls, 0, None))/max_tokens:.3f})")
    if ben:
        fp = [r for r in ben if r["trace"].trigger is not None]
        print(f"  良性轨迹 假触发率 = {len(fp)}/{len(ben)} = {len(fp)/len(ben):.3f}")
        for r in fp:
            print(f"    FP: {r['group_id']}  L={r['trace'].trigger.L} "
                  f"pred={r['trace'].trigger.pred} p={r['trace'].trigger.prob:.3f} "
                  f"cons={r['trace'].trigger.cons_ind_mean:.3f}")

    idx = {l: i for i, l in enumerate(LABELS)}
    cm = np.zeros((4, 5), int)          # last column = 未判定
    for r in results:
        v = r["trace"].verdict
        cm[idx[r["label"]], idx[v] if v else 4] += 1
    print(f"\n  轨迹级判定混淆矩阵 (n={len(results)})  行=真 列=判")
    print(" " * 24 + "".join(f"{l[:4]:>7}" for l in LABELS) + f"{'未定':>7}")
    for i, l in enumerate(LABELS):
        print(f"    {l:18s}" + "".join(f"{cm[i, j]:>7}" for j in range(5))
              + f"   (n={cm[i].sum()})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-bundle", required=True)
    ap.add_argument("--traj-dirs", nargs="+",
                    default=["recur_code/exp/step_trajectory_total"])
    ap.add_argument("--model", default=None, help="restrict to one LLM")
    ap.add_argument("--tau-prob", type=float, default=DEFAULT_TAU_PROB)
    ap.add_argument("--all-trajectories", action="store_true",
                    help="ignore the bundle's held-out split and replay every "
                         "trajectory (train ones included — reports leak).")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="generation cap assumed when reporting saved tokens.")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    det = load_detector(args.model_bundle)
    print(f"[replay] bundle={args.model_bundle}")
    print(f"[replay] features={det.feature_names} labels={det.classes} "
          f"train_rows={det.meta.get('n_train_rows')} "
          f"train_traj={det.meta.get('n_train_trajectories')}")

    trajs = load_trajectories(args.traj_dirs)
    if args.model:
        trajs = [t for t in trajs if t["model"] == args.model]
    if not args.all_trajectories:
        held = set(det.test_group_ids)
        trajs = [t for t in trajs if t["group_id"] in held]
        print(f"[replay] 只回放 bundle 的 held-out 轨迹（训练从未见过）")
    per_class = defaultdict(int)
    for t in trajs:
        per_class[t["label"]] += 1
    print(f"[replay] trajectories={len(trajs)}  {dict(per_class)}")
    if not trajs:
        raise SystemExit("[replay] no trajectories matched — check --traj-dirs/--model")

    results, rows = [], []
    for t in trajs:
        trace = replay_trace(det, t["cons"], t["pmf"], t["prompt_len"],
                             tau_prob=args.tau_prob)
        results.append({"group_id": t["group_id"], "label": t["label"],
                        "prompt_len": t["prompt_len"], "n_gen": int(t["pmf"].size),
                        "trace": trace})
        for c in trace.calls:
            rows.append((t["label"], c))

    per_checkpoint_report(rows, args.tau_prob, args.max_tokens)
    trajectory_report(results, args.max_tokens)

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        blob = [{"group_id": r["group_id"], "label": r["label"],
                 "prompt_len": r["prompt_len"], "n_gen": r["n_gen"],
                 **r["trace"].as_dict()} for r in results]
        with open(args.out_json, "w") as f:
            json.dump(blob, f, ensure_ascii=False, indent=1)
        print(f"\n[replay] 明细写入 {args.out_json}")


if __name__ == "__main__":
    main()
