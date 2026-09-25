"""Summarise `run_suppression_eval.py` output into the per-arm tables.

Reads the results JSONL and prints, per prompt set and arm:

  开门 (gate_open_L)         total length at which the detector first made a
                            confident attack call and suppression started;
                            always one of the deployed trigger checkpoints,
                            since that is the only grid probed before detection
  仍判为循环 (still_loop)    share of runs whose LAST probe is still a loop class
  解除次数 (n_release)       how often the gate closed because the detector
                            stopped calling it a loop
  抑制占比 (suppressed_frac) share of decode steps that ran under suppression
  可触及质量 (supp_mass)     mean pre-suppression attention mass on the targeted
                            keys in the applied layer — the ceiling on what any
                            k can move
  文本成环 (looped)          `_common.loop_check_for_category` on the final text
  三元组多样度 (uniq3)        unique 3-gram ratio, 1.0 = no repetition at all

    /root/miniconda3/envs/Recur/bin/python \\
        recur_code/script/experiments/summarize_suppression_eval.py \\
        --results recur_code/exp/suppression/eval/<model>/<condition>

`--results` 传条件目录时读进该目录下所有臂文件（同条件才可比），传单个
JSONL 时只读那一个臂。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x, nd=3):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True,
                    help="条件目录（读其中所有臂文件 *.jsonl）或单个 JSONL 文件")
    ap.add_argument("--paired-only", action="store_true",
                    help="keep only prompts that every arm completed")
    a = ap.parse_args()

    src = Path(a.results)
    files = sorted(f for f in src.glob("*.jsonl")) if src.is_dir() else [src]
    if not files:
        raise SystemExit(f"{src} 下没有 *.jsonl")
    rows = [json.loads(l) for f in files for l in open(f)]
    arms = sorted({r["arm"] for r in rows})

    man = src / "manifest.json" if src.is_dir() else src.parent / "manifest.json"
    if man.exists():
        m = json.load(open(man))
        print(f"条件：模型={m.get('模型 (model)')} "
              f"温度={m.get('采样温度 (temperature)')} "
              f"生成上限={m.get('生成上限 (max_new_tokens)')}\n")
    if a.paired_only:
        seen = defaultdict(set)
        for r in rows:
            seen[(r["set"], r["id"])].add(r["arm"])
        keep = {k for k, v in seen.items() if v >= set(arms)}
        rows = [r for r in rows if (r["set"], r["id"]) in keep]

    by = defaultdict(list)
    for r in rows:
        by[(r["set"], r["arm"])].append(r)

    hdr = (f"{'集合':22s}{'臂':22s}{'n':>4}{'开门率':>8}{'开门L':>8}"
           f"{'仍判循环':>9}{'解除':>6}{'重触发':>7}{'抑制%':>7}{'可触及':>8}"
           f"{'文本成环':>9}{'多样度':>8}{'自然结束':>9}{'长度':>7}")
    print(hdr)
    print("-" * 145)
    for s in sorted({r["set"] for r in rows}):
        for arm in arms:
            g = by.get((s, arm))
            if not g:
                continue
            n = len(g)
            opened = [r for r in g if r.get("gate_open_L")]
            still = [r for r in g if r.get("final_pred") in
                     ("repetitive_reasoning", "repetitive_string")]
            looped = [r for r in g if r.get("looped")]
            print(f"{s:22s}{arm:22s}{n:>4}{len(opened) / n:>8.2f}"
                  f"{fmt(mean([r['gate_open_L'] for r in opened]), 0):>8}"
                  f"{len(still) / n:>9.2f}"
                  f"{fmt(mean([r.get('n_release') for r in g]), 2):>6}"
                  f"{fmt(mean([r.get('n_rearm') for r in g]), 2):>7}"
                  f"{fmt(100 * (mean([r.get('suppressed_frac') for r in g]) or 0), 1):>7}"
                  f"{fmt(mean([r.get('supp_mass_applied_layer') for r in g]), 4):>8}"
                  f"{len(looped) / n:>9.2f}"
                  f"{fmt(mean([r.get('uniq3') for r in g]), 3):>8}"
                  f"{fmt(mean([1.0 if r['stopped_naturally'] else 0.0 for r in g]), 2):>9}"
                  f"{fmt(mean([r['gen_tokens'] for r in g]), 0):>7}")
        print()

    # per-prompt view on the attack set
    print("逐条（攻击集）")
    print(f"{'id':>6}  {'臂':20s}{'开门L':>8}{'复检次数':>9}{'续压':>6}"
          f"{'最终判定':>22}{'攻击概率':>9}{'解除':>6}{'成环':>6}{'多样度':>8}")
    for r in sorted(rows, key=lambda r: (r["set"], r["id"], r["arm"])):
        if not r["set"].startswith("S1"):
            continue
        print(f"{r['id']:>6}  {r['arm']:20s}"
              f"{str(r.get('gate_open_L')):>8}"
              f"{str(r.get('n_probes')):>9}"
              f"{str(r.get('n_bursts')):>6}"
              f"{str(r.get('final_pred')):>22}"
              f"{fmt(r.get('final_attack_prob'), 3):>9}"
              f"{str(r.get('n_release')):>6}{str(r.get('looped')):>6}"
              f"{fmt(r.get('uniq3'), 3):>8}")
    print(f"\n共 {len(rows)} 条运行，来源 {', '.join(f.name for f in files)}"
          f"（{src}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
