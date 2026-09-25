#!/usr/bin/env python3
"""从已算好的逐位置指标重新取点，产出分类器训练集（行式 JSONL）。

用途：`run_model_detection_pipeline.py` 的 `train` 段只读它自己 `--out` 下的
`step_metrics/`，所以覆盖不到 `exp/step_trajectory_total/` 那份**已经算好的多模型**
逐位置指标。本脚本补上这条入口：从任意一批 step_trajectory JSON 取点，产出与流水线
`train/classifier_points.jsonl` **同 schema** 的文件，再交给

    run_model_detection_pipeline.py --stage train --train-points <本脚本产物> --out <目录>

训练。两边的取点公式与特征定义共用同一处实现（`detector.py` 的 `_running_mean` /
`_running_slope` / `true_label` / `load_trajectories`），所以合并训练与单模型训练算出的
特征可比。

两种取点口径：

  --n-segments N       每条轨迹**固定** N 个探针（生成段等分 N 份、各取段末）。
                       旧的 6 模型合并训练集就是这么取的：良性两类 N=10、攻击两类
                       N=40（攻击轨更长更稀疏）。
  --points-per-class M 每类**总共** M 个探针，名额按轨迹长度**反比**分配，与流水线
                       `train` 段的 `allocation()` 一致（默认 1000）。

两者互斥。不确定选哪个：要复现旧的合并数据集用 `--n-segments`，要和一键流程口径完全
对齐用 `--points-per-class`。

与归档的 `two_class_equal10_6models.jsonl` 有两处差异，都是故意的：

  1. **探针位置**：归档那份取 `round(gen_len*seg/n) - 1`，流水线取 `ceil(...) - 1`，约
     36% 的行差 1 个生成位置。本脚本默认跟流水线，`--legacy-probe-index` 可切回 round。
  2. **`L_feat` 的含义**：归档那份存的是**原始 L**（训练时再由 `--log-L` 取对数），本脚本
     和流水线一样直接存 `log L`。所以**归档那份不能直接喂给现在的 train 段**——它会把
     原始 L 当成第三个特征、训出不同的模型。这正是要用本脚本重建的理由。

用法：

    # 良性两类，等分 10 段（6 个模型一起）
    PYTHONPATH=. python recur_code/script/experiments/build_classifier_points.py \\
        --traj-dirs recur_code/exp/step_trajectory_total --n-segments 10 \\
        --classes concise productive \\
        --out recur_code/exp/detection/points/two_class_equal10_6models.jsonl

    # 攻击两类，等分 40 段
    PYTHONPATH=. python recur_code/script/experiments/build_classifier_points.py \\
        --traj-dirs recur_code/exp/step_trajectory_total --n-segments 40 \\
        --classes repetitive_reasoning repetitive_string \\
        --out recur_code/exp/detection/points/attack_two_class_equal40_6models.jsonl

    # 只要某一个模型（重训该模型的专属检测器）
    ... --models DeepSeek-R1-Distill-Qwen-14B --points-per-class 1000
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "recur_code/src/detection"))
from detector import (  # noqa: E402
    LABELS, MIN_GEN_FOR_SLOPE, _running_mean, _running_slope, load_trajectories,
)


def allocation(lengths: list[int], target: int) -> list[int]:
    """把 target 个名额按长度反比分给各条轨迹，至少 1 个。

    与 `run_model_detection_pipeline.py::allocation` 同口径：长轨迹不该仅因为位置多
    就贡献更多探针，否则类别先验被它们淹没。"""
    if not lengths:
        return []
    inv = np.asarray([1.0 / max(1, n) for n in lengths], dtype=float)
    share = inv / inv.sum() * target
    n = np.maximum(1, np.floor(share)).astype(int)
    # 把地板留下的余量按小数部分从大到小补回去
    short = target - int(n.sum())
    if short > 0:
        for i in np.argsort(-(share - np.floor(share)))[:short]:
            n[i] += 1
    return [int(x) for x in n]


def rows_for(traj: dict, n_probes: int, legacy: bool = False) -> list[dict]:
    """一条轨迹 → n_probes 行。

    第 seg 个探针的生成位置（seg = 1..n）：

        默认   i = ceil(gen_len * seg / n) - 1      与流水线 train 段逐字相同
        legacy i = round(gen_len * seg / n) - 1     归档那份 6 模型数据集的取法
                                                    （.5 按 Python 的银行家舍入）

    两者只在 `gen_len * seg / n` 不是整数时差 1 个位置（实测约 36% 的行），对累计
    均值/斜率的影响可忽略。**默认用流水线口径**，这样本脚本取的点与一键流程自己取的
    点特征可比；只有要与归档数据集逐行对齐时才加 `--legacy-probe-index`——
    已验证 legacy 模式能逐行复现归档的 12000 行（`L_feat` 除外，见模块文档）。"""
    pmf, cons = traj["pmf"], traj["cons"]
    out: list[dict] = []
    for seg in range(1, n_probes + 1):
        x = pmf.size * seg / n_probes
        i = (round(x) if legacy else math.ceil(x)) - 1
        if i < MIN_GEN_FOR_SLOPE - 1:
            continue
        if not np.isfinite(pmf[: i + 1]).all() or not np.isfinite(cons[: i + 1]).all():
            continue
        cm = _running_mean(cons, i)
        sl = _running_slope(pmf, i)
        if not (np.isfinite(cm) and np.isfinite(sl)):
            continue
        L = int(traj["prompt_len"]) + i + 1
        out.append({
            "model": traj["model"], "category": traj["category"], "id": traj["id"],
            "label": traj["label"], "prompt_len": int(traj["prompt_len"]),
            "gen_len": int(pmf.size), "seg_idx": seg, "gen_pos": i, "L": L,
            "L_feat": float(np.log(L)), "cons_ind_mean": cm, "pmf_slope": sl,
            "group_id": traj["group_id"],
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dirs", nargs="+", required=True,
                    help="step_trajectory JSON 所在目录（递归扫描），"
                         "例如 recur_code/exp/step_trajectory_total")
    ap.add_argument("--out", type=Path, required=True, help="输出 JSONL")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--n-segments", type=int, default=None,
                   help="每条轨迹固定这么多探针（生成段等分）")
    g.add_argument("--points-per-class", type=int, default=None,
                   help="每类总探针数，名额按轨迹长度反比分配（默认口径，1000）")
    ap.add_argument("--classes", nargs="*", default=None,
                    help=f"只要这些标签，默认全要：{LABELS}")
    ap.add_argument("--models", nargs="*", default=None, help="只要这些模型目录名")
    ap.add_argument("--legacy-probe-index", action="store_true",
                    help="探针位置用归档数据集那套 floor 取法，而不是流水线的 ceil；"
                         "只在需要与归档文件逐行对齐时加")
    a = ap.parse_args()
    if a.n_segments is None and a.points_per_class is None:
        a.points_per_class = 1000

    trajs = load_trajectories(a.traj_dirs)
    if not trajs:
        raise SystemExit(f"[points] {a.traj_dirs} 下没有可用的 step_trajectory JSON")
    keep = set(a.classes) if a.classes else set(LABELS)
    bad = keep - set(LABELS)
    if bad:
        raise SystemExit(f"[points] 未知标签 {sorted(bad)}，可选 {LABELS}")
    trajs = [t for t in trajs if t["label"] in keep]
    if a.models:
        trajs = [t for t in trajs if t["model"] in set(a.models)]
    if not trajs:
        raise SystemExit("[points] 过滤后没有轨迹剩下")

    rows: list[dict] = []
    if a.n_segments is not None:
        for t in trajs:
            rows += rows_for(t, a.n_segments, a.legacy_probe_index)
    else:
        by_label: dict[str, list[dict]] = defaultdict(list)
        for t in trajs:
            by_label[t["label"]].append(t)
        for label, group in by_label.items():
            counts = allocation([t["pmf"].size for t in group], a.points_per_class)
            for t, n in zip(group, counts):
                rows += rows_for(t, n, a.legacy_probe_index)
    if not rows:
        raise SystemExit("[points] 一行都没取到——生成段普遍短于斜率所需的 3 个点？")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(
        json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows))

    # 覆盖度报告：类别 / 模型 / 段号三个维度，任一维度极度不均都会毁掉先验估计
    per_label = Counter(r["label"] for r in rows)
    per_model = Counter(r["model"] for r in rows)
    per_seg = Counter(r["seg_idx"] for r in rows)
    n_traj = len({r["group_id"] for r in rows})
    mode = (f"n_segments={a.n_segments}" if a.n_segments is not None
            else f"points_per_class={a.points_per_class}（按长度反比分配）")
    print(f"[points] {mode}")
    print(f"[points] {len(rows)} 行 / {n_traj} 条轨迹 → {a.out}")
    print(f"[points] 逐类别 : {dict(sorted(per_label.items()))}")
    print(f"[points] 逐模型 : {dict(sorted(per_model.items()))}")
    print(f"[points] 段号范围: {min(per_seg)}–{max(per_seg)}，"
          f"最少的段 {min(per_seg.values())} 行、最多的段 {max(per_seg.values())} 行")
    dropped = len(trajs) - n_traj
    if dropped:
        print(f"[points] 有 {dropped} 条轨迹一行都没取到（生成段太短或指标含 NaN）")


if __name__ == "__main__":
    main()
