"""W3 — paired evaluation of the loop-suppression runtime.

One model instance, one prompt at a time, every arm run back to back on that
prompt (greedy, temperature from config.json), so arms differ ONLY by the
intervention:

    A0 off        基线：只检测、不干预
    M1 suppress   **主方案**（v5 class_aware）：分类器在部署检查点判出循环后连续
                  抑制，乘子按判出的类别选参考曲线——
                    · 提示侧：整段乘常数 c，把提示侧强度对齐正常思考轨迹。
                      主档瞄 pmf 目标；pmf 目标要求的提示侧质量占比 ≥ 1（长提示上
                      必然如此）时改瞄参考轨迹的提示侧质量占比。闭式、永远有解、
                      不设量程，注意力汇不作特殊处理。
                    · 生成侧：把「相对提示侧被过度使用」的 top-m 词串逐串压回它们
                      在提示里的质量占比（只压不抬），其余位置回填，生成侧总质量
                      精确不变。
                  两半都逐层自解自施（零滞后），施加层由 --flatten-layers 给；
                  Llama-8B 上逐层扫描的结论是**第 15 层**，换模型要重扫。
                  见 suppression_runtime.py 模块文档 v5 与
                  docs/drafts/suppression_v5_class_aware_design.md。

消融臂（A1/A2/A3、C2/C3、D1、F1/F2、G1–G9、H1、V2–V6）已于 2026-08-27 删除：
现在的目标是提高主方案的效果，不是做消融。历史结果仍在
exp/suppression/eval/<模型>/ 下，臂名保留在那些目录的 manifest.json 里。

Every arm carries the detector, so A0 records what the detector *would* have
said with no intervention at all: that is the denominator for "did suppression
change the verdict", and it costs nothing (both detection features are per-row
quantities read off the decode step the runtime already measures).

Prompt sets (Llama-8B numbers; other models scale down):
    S1 attack·loop     dataset/attack_resample/<model>/{repetitive_reasoning,
                       repetitive_string}.jsonl ∩ known_good_ids.json
                       — prompts that are known to loop ON THIS MODEL, which is
                       the only honest denominator for a loop-break rate
    S3 benign·concise  reasoning_trajectory/<model>/from_total/concise_reasoning
    S4 benign·product. reasoning_trajectory/<model>/from_total/productive_reflection
    S5 utility         dataset/gsm8k/gsm8k_test.jsonl (answer accuracy)

Per run we record: loop verdict (`_common.loop_check_for_category`, the same
span convention the attack sets were built with), natural EOS, generated
tokens, unique-3gram ratio, the cons_ind history, when the gate fired and how
long it stayed on. Results stream to JSONL and the script is resumable
((set, id, arm) already present is skipped).

    PYTHONPATH=. /root/miniconda3/envs/Recur/bin/python \\
        recur_code/script/experiments/run_suppression_eval.py --gpu 1
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
RECUR_CODE = THIS_DIR.parents[1]
REPO_ROOT = RECUR_CODE.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(THIS_DIR))

from _common import loop_check, loop_check_for_category  # noqa: E402

from recur_code.src.detection import row_reshape as RR  # noqa: E402
from recur_code.src.detection.attention_reference import (  # noqa: E402
    DEFAULT_NORMALIZE,
)
from recur_code.src.detection.suppression_runtime import (  # noqa: E402
    DEFAULT_FLATTEN_LAYERS, DEFAULT_PROMPT_LAYERS, OFF_RULES,
    PROMPT_TARGET_OFFSETS, SuppressionRuntime,
)

CONFIG = json.load(open(RECUR_CODE / "config.json"))

ARMS = [
    # 基线：只检测不干预。每条臂都带着检测器，所以 A0 记录的是「完全不干预时
    # 检测器会怎么判」——那是「抑制有没有改变结局」的分母，而且不额外花钱
    # （两个检测特征都是逐行量，就在运行时已经测过的那一步里读出来）。
    {"arm": "A0_off", "mode": "off"},
    # 主方案。设置写全在这里，`--arms A0_off,M1_suppress` 一条命令即可复现。
    #   layers=all          乘子允许落在任何层（实际施加面由下面两个键决定）
    #   plan_source=own_layer  每层用自己这一步的行解自己的系数，零滞后
    #   prompt_layers=gen   提示侧与生成侧作用在同一批层，其余层原样不动
    #   flatten_layers=15   Llama-8B 逐层扫描的结论；换模型必须重扫
    #   gen_op=prompt_align 生成侧逐串对齐提示侧占比（只压不抬）
    #   topm_rank=share_diff  名额按「生成侧占比 − 提示侧占比」发，全部落在
    #                         真正被过度使用的串上
    #   top_m=3             被压的词串个数
    #   burst_steps=0       门开后连续压，不按突发预算断续
    {"arm": "M1_suppress", "mode": "classifier",
     "layers": "all", "plan_source": "own_layer", "prompt_layers": "gen",
     "flatten_layers": "15", "gen_op": "prompt_align",
     "topm_rank": "share_diff", "top_m": 3, "burst_steps": 0},
    # 主方案 + 目标平移。与 M1 只差 prompt_target_offset：曲线目标整体平移到
    # 「开始抑制那一刻的实测水平」，于是起点不跳、之后实际 pmf 轨迹与参考曲线
    # 平行，而不是第一步就跳到曲线的水平上（`suppression_runtime._offset_target`）。
    # 注意它会顺带让封顶档几乎不再触发，所以这条臂同时也是「弱干预」的一次测量。
    {"arm": "M2_offset", "mode": "classifier",
     "layers": "all", "plan_source": "own_layer", "prompt_layers": "gen",
     "flatten_layers": "15", "gen_op": "prompt_align",
     "topm_rank": "share_diff", "top_m": 3, "burst_steps": 0,
     "prompt_target_offset": "trigger"},
    # 对比基线：AUSteer（arXiv:2602.04428）。与主方案**正交**——不碰注意力行，
    # 按离线 AU 计划缩放 FFN/o_proj 输入的 ≤100 个标量维度，且没有在线决策
    # （mode=always，从第 0 步压到结束）。计划、变体、k、α 由 --au-* 给。
    {"arm": "B1_austeer", "mode": "always", "suppressor": "austeer",
     "layers": "all", "flatten_layers": "none", "prompt_layers": "none"},
    # 消融（不是基线）：AUSteer 的算子 + 我们的门。与 B1 只差门，与 M1 只差算子，
    # 三条臂合起来才能把「门控」与「干预方式」的贡献拆开。
    {"arm": "B1g_austeer_gated", "mode": "classifier", "suppressor": "austeer",
     "layers": "all", "flatten_layers": "none", "prompt_layers": "none",
     "burst_steps": 0, "au_class": "auto"},
]
DEFAULT_ARMS = "A0_off,M1_suppress"
DEFAULT_DETECTOR = (RECUR_CODE / "exp" / "detection_dataset" / "models" /
                    "loop_detector_llama8b_4class_method2.json")



def expand_altref_arms(arms, spec: str):
    """给主方案臂再加一条**只换参考曲线**的臂：`--alt-reference <曲线.json>`。

    臂名取曲线文件名里 `attention_reference_<模型>` 之后的那一截（例如
    `..._deg1.json` → `M1_deg1`），其余参数逐键复制 M1_suppress，所以两条臂之间
    只差曲线本身。A0_off 不读曲线，仍然是共同的分母。
    """
    base = next((x for x in arms if x["arm"] == "M1_suppress"), None)
    if base is None:
        raise SystemExit("ARMS 里找不到 M1_suppress，无法展开换曲线臂")
    out = list(arms)
    for path in [p for p in spec.replace(" ", "").split(",") if p]:
        stem = Path(path).stem
        tag = stem.split("attention_reference_", 1)[-1]
        tag = tag.split("_")[-1] if "_" in tag else "alt"
        one = dict(base)
        one["arm"] = f"M1_{tag}"
        one["reference"] = str(Path(path).resolve())
        out.append(one)
    return out


def expand_sweep_arms(arms, spec: str):
    """把主方案臂按**施加层**展开成一层一臂：`11-18` -> M1_layer11 … M1_layer18。

    单因素扫描要求除施加层外一切相同，所以直接复制 M1_suppress 的全部参数，
    只改 flatten_layers。臂名进文件名，断点续跑与 manifest 都按臂名走，
    因此同一条件目录下可以随时补跑新的层。
    （历史上的 G9_layer<N> 臂已于 2026-08-27 随消融臂一起删除，这里重建的是
      同一件事的可参数化版本。）
    """
    layers: list[int] = []
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            layers.extend(range(int(lo), int(hi) + 1))
        else:
            layers.append(int(part))
    base = next((x for x in ARMS if x["arm"] == "M1_suppress"), None)
    if base is None:
        raise SystemExit("ARMS 里找不到 M1_suppress，无法展开逐层扫描臂")
    out = [x for x in arms if x["arm"] != "M1_suppress"]
    for L in layers:
        one = dict(base)
        one["arm"] = f"M1_layer{L}"
        one["flatten_layers"] = str(L)
        out.append(one)
    return out


def expand_combo_arms(arms, spec: str):
    """把主方案臂按**层的组合**展开：`12,14,15;13,17` -> 两个臂，各自同时在
    组内所有层上施加。组之间用 `;` 分隔，组内沿用运行时的层规格语法
    （逗号 + `a-b` 闭区间），臂名形如 M1_L12_14_15 / M1_L11to18。

    与 --sweep-layers 的区别：那个是一层一臂的单因素扫描，这个是一组一臂的
    联合施加。两者都只改 flatten_layers，其余参数与主方案完全相同；
    prompt_layers="gen" 会让提示侧自动跟随同一批层。
    """
    base = next((x for x in ARMS if x["arm"] == "M1_suppress"), None)
    if base is None:
        raise SystemExit("ARMS 里找不到 M1_suppress，无法展开联合施加臂")
    out = [x for x in arms if x["arm"] != "M1_suppress"]
    for grp in spec.split(";"):
        grp = grp.replace(" ", "")
        if not grp:
            continue
        one = dict(base)
        one["arm"] = "M1_L" + grp.replace(",", "_").replace("-", "to")
        one["flatten_layers"] = grp
        out.append(one)
    return out

def expand_dyn_arms(arms, spec: str):
    """把主方案臂按**动态选层规格**展开：`cons_low:2:11-18;thresh:0.4:11-18`
    -> 两个臂。排序类规则（cons_low/cons_high/random）在门开那一刻按逐层一致性
    定下 n 层并冻结；阈值类规则（thresh/reldrop）**逐步逐层**判定，参数是阈值。

    与 --sweep-layers / --combo-layers 的区别：那两个把施加层写死在配置里，
    这个只写「怎么选」，层由运行时逐样例定。臂名形如
    M1_dyn_cons_low_n2（规则 + 层数），候选层写进 manifest 与每条记录。
    """
    base = next((x for x in ARMS if x["arm"] == "M1_suppress"), None)
    if base is None:
        raise SystemExit("ARMS 里找不到 M1_suppress，无法展开动态选层臂")
    out = [x for x in arms if x["arm"] != "M1_suppress"]
    for grp in spec.split(";"):
        grp = grp.replace(" ", "")
        if not grp:
            continue
        rule, param, _cands = grp.split(":")
        one = dict(base)
        # 阈值类规则的参数是小数，写进臂名时按百分数取整（0.40 -> t40），
        # 免得文件名里出现小数点
        tag = (f"n{param}" if rule in ("cons_low", "cons_high", "random")
               else f"t{round(float(param) * 100):02d}")
        one["arm"] = f"M1_dyn_{rule}_{tag}"
        one["dyn_select"] = grp
        # 施加层由运行时定，静态规格留空避免误读
        one["flatten_layers"] = "none"
        out.append(one)
    return out


def expand_au_arms(arms, spec: str):
    """把 AUSteer 基线按 **(k, α)** 展开成一格一臂：`50:0.5,100:1` -> 两个臂。

    超参扫描要求除 k、α 外一切相同，所以逐键复制 B1_austeer，只改这两个值；
    臂名形如 `B1_austeer_k50_a0.5`，进文件名，断点续跑与 manifest 都按臂名走。
    注意扫描**必须在 train/spare 的样本上做**（`--pipeline-test` 指向别处），
    test 是最终报告口径。
    """
    base = next((x for x in ARMS if x["arm"] == "B1_austeer"), None)
    if base is None:
        raise SystemExit("ARMS 里找不到 B1_austeer，无法展开超参扫描臂")
    out = [x for x in arms if x["arm"] != "B1_austeer"]
    for grp in spec.replace(" ", "").split(","):
        if not grp:
            continue
        k, alpha = grp.split(":")
        one = dict(base)
        one["arm"] = f"B1_austeer_k{k}_a{alpha}"
        one["au_top_k"] = int(k)
        one["au_alpha"] = float(alpha)
        out.append(one)
    return out


def condition_name(temperature: float, max_new_tokens: int) -> str:
    """结果目录名 = 一次评测里所有臂共享的条件（采样温度 + 生成上限）。

    臂之间只能在同一条件下比较，所以条件进目录名、臂进文件名：
        exp/suppression/eval/<模型>/t0.0_cap8k/M1_suppress.jsonl
    """
    cap = (f"{max_new_tokens // 1024}k" if max_new_tokens % 1024 == 0
           else str(max_new_tokens))
    return f"t{temperature:g}_cap{cap}" if temperature != int(temperature) \
        else f"t{float(temperature):.1f}_cap{cap}"


def _rel(path) -> str:
    """路径尽量写成仓库相对形式，manifest 换机器仍可读。"""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def write_manifest(out_dir: Path, model_name: str, a) -> None:
    """把该条件目录下每个臂的实际参数与样本数落成 manifest.json。

    从磁盘上的臂文件反读，而不是从命令行参数写——目录可能是多次运行、
    不同臂陆续累积起来的，只有文件本身才是权威。
    """
    old_arms = {}
    if (out_dir / "manifest.json").exists():
        try:
            old_arms = json.load(open(out_dir / "manifest.json")).get(
                "实验臂 (arms)", {})
        except Exception:
            old_arms = {}
    arms = {}
    for f in sorted(out_dir.glob("*.jsonl")):
        rows = []
        for line in open(f):
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
        if not rows:
            continue
        r0 = rows[0]
        sets = {}
        for r in rows:
            sets[r["set"]] = sets.get(r["set"], 0) + 1
        arms[f.stem] = {
            "样本数 (n_runs)": len(rows),
            "提示集分布 (sets)": sets,
            "门控模式 (mode)": r0["mode"],
            "干预算子 (suppressor)": r0.get("suppressor", "row_reshape"),
            # AUSteer 基线的复现字段；主方案臂这几项都是 null
            "AU 计划 (au_plan)": r0.get("au_plan"),
            "AU 变体 (au_variant)": r0.get("au_variant"),
            "AU 类别 (au_class)": r0.get("au_class"),
            "AU 强度 (au_alpha)": r0.get("au_alpha"),
            "被引导的激活数 (#Acts)": r0.get("au_n_acts"),
            "施加层 (layers)": r0["layers"],
            "计划来源 (plan_source)": r0.get("plan_source", "own_layer"),
            "生成侧算子 (gen_op)": r0.get("gen_op"),
            "提示侧层 (prompt_layers)": r0.get("prompt_layers_spec"),
            "生成侧层 (flatten_layers)": r0.get("flatten_layers_spec"),
            "动态选层 (dyn_select)": r0.get("dyn_spec"),
            "top-m 排序 (topm_rank)": r0.get("topm_rank"),
            # 产出日期累积：旧 manifest 里的日期 + 该臂文件最后写入日期
            "产出日期": sorted(set(old_arms.get(f.stem, {}).get("产出日期", []))
                           | {datetime.date.fromtimestamp(
                               f.stat().st_mtime).isoformat()}),
        }
    man_path = out_dir / "manifest.json"
    old = {}
    if man_path.exists():
        try:
            old = json.load(open(man_path))
        except Exception:
            old = {}
    man = {
        "模型 (model)": model_name,
        "采样温度 (temperature)": a.temperature,
        "生成上限 (max_new_tokens)": a.max_new_tokens,
        "随机种子 (seed)": a.seed,
        "门控分类器 (detector)": _rel(a.detector),
        "参考曲线 (attention_reference)": _rel(a.reference) if a.reference else "按模型名默认",
        "说明": old.get("说明", ""),
        "实验臂 (arms)": arms,
        "复现": ("PYTHONPATH=. /root/miniconda3/envs/Recur/bin/python "
               "recur_code/script/experiments/run_suppression_eval.py "
               f"--gpu <空闲卡> --temperature {a.temperature} "
               f"--max-new-tokens {a.max_new_tokens} --arms <臂名>"),
    }
    json.dump(man, open(man_path, "w"), ensure_ascii=False, indent=2)


# ------------------------------------------------------------------ prompt sets
# 一键流程的四个类别名 -> 抑制评测里的集合标签。良性/攻击的温度按类别分开设，
# 因为轨迹本来就是按不同温度生成的，评测要对齐生成口径。
PIPELINE_CATEGORIES = {
    "concise_reasoning": ("T1_benign_concise", "benign"),
    "productive_reasoning": ("T2_benign_productive", "benign"),
    "repetitive_reasoning": ("T3_attack_reasoning", "attack"),
    "repetitive_string": ("T4_attack_string", "attack"),
}


def load_pipeline_test(root: Path):
    """`pipeline/<模型>/test/records/<数据集>/<类别>.jsonl` 里的全部提示。

    只收 `<类别>.jsonl`，不收 `<类别>_spare.jsonl` —— 备用集与 test 同目录、
    靠文件名区分，用 glob 会把它一起捞进来。

    同一条样例可能有多条轨迹（攻击类在三个温度下各生成一次），抑制评测是
    **按样例**跑的，所以这里去重；id 取该样例第一次出现时的 id，便于回查原轨迹。

    去重键是 (prompt, prefill) 而不是 prompt：concat 的样例共用少数几条指令，
    彼此只差种在 <think> 里的重复词（prefill），只按 prompt 去重会把 25 条压成
    4 条。prefill 随样本带给 `SuppressionRuntime.run`，否则种词丢失、攻击根本
    没有施加。其余子集 prefill 为空，行为不变。
    """
    samples: list[dict[str, Any]] = []
    for cat, (tag, kind) in PIPELINE_CATEGORIES.items():
        seen: set[tuple[str, str]] = set()
        for path in sorted((root / "test/records").glob(f"*/{cat}.jsonl")):
            subset = path.parent.name
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                prompt = str(r.get("prompt", ""))
                prefill = str(r.get("prefill") or "")
                if not prompt or (prompt, prefill) in seen:
                    continue
                seen.add((prompt, prefill))
                samples.append({"set": tag, "category": cat, "kind": kind,
                                "subset": subset, "id": f"{subset}_{r['id']}",
                                "prefill": prefill,
                                "prompt": prompt})
    return samples


def load_sets(model_name: str, n_attack: int, n_benign: int, n_utility: int,
              all_attack: bool = False):
    """all_attack=True 时 S1 用该模型攻击集的**全部**提示，而不是只用
    known_good_ids（已知能在该模型上成环的子集）。

    两种口径回答的问题不同：known_good 是「打断率」的诚实分母（对象必须先存在），
    全部提示则是部署视角——里面本来就混着不成环的攻击，正好用来同时读出检测的
    漏报/误报与抑制的打断效果。"""
    samples: list[dict[str, Any]] = []
    known = json.load(open(RECUR_CODE / "dataset" / "source" / "attack_resample" /
                           "known_good_ids.json")).get(model_name, {})
    for cat in ("repetitive_reasoning", "repetitive_string"):
        path = (RECUR_CODE / "dataset" / "source" / "attack_resample" / model_name /
                f"{cat}.jsonl")
        if not path.exists():
            continue
        rows = {r["id"]: r for r in map(json.loads, open(path))}
        ids = (list(rows) if all_attack else list(known.get(cat, [])))
        if n_attack > 0:
            ids = ids[:n_attack]
        if not ids:
            continue
        for i in ids:
            r = rows.get(i)
            if r is None:
                continue
            samples.append({"set": "S1_attack_loop", "category": cat,
                            "id": i, "prompt_sha1": r.get("prompt_sha1"),
                            "known_good": i in set(known.get(cat, [])),
                            "prompt": r["prompt"]})

    for cat, tag in (("concise_reasoning", "S3_benign_concise"),
                     ("productive_reflection", "S4_benign_productive")):
        path = (RECUR_CODE / "reasoning_trajectory" / model_name / "from_total" /
                f"{cat}.jsonl")
        if not path.exists():
            continue
        rows = [json.loads(l) for l in open(path)][:n_benign]
        for r in rows:
            samples.append({"set": tag, "category": cat, "id": r["id"],
                            "prompt_sha1": r.get("prompt_sha1"),
                            "uid": r.get("uid"), "prompt": r["prompt"]})

    # known_good（已知能在该模型上成环）排在前面：全量攻击集里大部分提示根本
    # 不成环、跑几百 token 就自然结束，先跑它们等于把「抑制能不能打断循环」这个
    # 主问题推到最后。断点续跑时顺序不影响正确性，只影响先看到什么。
    samples.sort(key=lambda x: (x["set"] != "S1_attack_loop",
                                not x.get("known_good", False)))

    gsm = RECUR_CODE / "dataset" / "source" / "gsm8k" / "gsm8k_test.jsonl"
    if n_utility and gsm.exists():
        for i, line in enumerate(open(gsm)):
            if i >= n_utility:
                break
            r = json.loads(line)
            samples.append({"set": "S5_utility", "category": "gsm8k", "id": i,
                            "gold": r["answer"].split("####")[-1].strip(),
                            "prompt": r["question"]})
    return samples


# ------------------------------------------------------------------- utilities
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def last_number(text: str) -> str | None:
    m = _NUM.findall(text.replace(",", ""))
    return m[-1].rstrip(".") if m else None


def uniq_ngram_ratio(text: str, n: int = 3) -> float:
    w = text.split()
    if len(w) <= n:
        return 1.0
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return len(set(grams)) / len(grams)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=CONFIG["models"][CONFIG["model_id"]])
    ap.add_argument("--gpu", default=str(CONFIG.get("gpu", "0")))
    ap.add_argument("--arms", default=DEFAULT_ARMS)
    ap.add_argument("--sweep-layers", default="",
                    help="逐层扫描：把 M1_suppress 按施加层展开成一层一臂，\n如 `11-18` 或 `11,15,18`，臂名 M1_layer<N>；其余参数与主方案完全相同")
    ap.add_argument("--dyn-arms", default="",
                    help="动态选层臂，`;` 分隔，每项写 <规则>:<n>:<候选层>，"
                         "如 cons_low:2:11-18；规则见 DYN_RULES")
    ap.add_argument("--dyn-stride", type=int, default=8,
                    help="每多少步测一次候选层的逐层一致性")
    ap.add_argument("--dyn-window", type=int, default=0,
                    help="选层时只看最近多少次测量；0 = 门开前全部")
    ap.add_argument("--combo-layers", default="",
                    help="联合施加：一组层一臂，组间用 `;` 分隔，\n如 `12,14,15;13,17;11-18`，臂名 M1_L12_14_15 / M1_L11to18")
    ap.add_argument("--layers", default="last",
                    help="last = suppress the final layer with a same-step "
                         "(lag-free) plan; closest8/all/random:<n> are lagged")
    ap.add_argument("--detector", default=str(DEFAULT_DETECTOR),
                    help="classifier JSON export; carried by every arm")
    # ---- 正常思考的提示侧强度参考曲线 + 生成侧算子参数 ----------------------
    ap.add_argument("--reference", default=None,
                    help="attention_reference.py 生成的参考曲线；默认取"
                         "exp/detection_dataset/models/attention_reference_<模型>.json")
    ap.add_argument("--ref-normalize", default=DEFAULT_NORMALIZE,
                    choices=["position", "position_prompt_len"],
                    help="目标口径：position_prompt_len（默认，合并曲线必须带"
                         "长度项）/ position（只按生成位置，对照）")
    ap.add_argument("--ref-quantile", type=float, default=0.5)
    ap.add_argument("--sink-positions", type=int, default=1,
                    help="只影响诊断字段 sink_share；提示侧整段同乘一个 c，\n"
                         "注意力汇不作任何特殊处理")
    ap.add_argument("--prompt-cap", type=float, default=0.0,
                    help="提示侧常数 c 的量程；0 = 不设限制（默认）")
    ap.add_argument("--prompt-direction", default="both",
                    choices=["both", "expected", "off"])
    ap.add_argument("--alt-reference", default="",
                    help="逗号分隔的参考曲线 JSON；每条再加一条与 M1_suppress "
                         "只差曲线的臂（臂名取文件名尾巴，如 _deg1 → M1_deg1）")
    ap.add_argument("--prompt-target-offset", default="none",
                    choices=list(PROMPT_TARGET_OFFSETS),
                    help="曲线目标是否平移到开始抑制那一刻的实测水平（臂里没写"
                         "时的默认值）：none / trigger / trigger_log")
    ap.add_argument("--flatten-layers", default=DEFAULT_FLATTEN_LAYERS,
                    help="生成侧削峰施加的层；none 关闭")
    ap.add_argument("--prompt-layers", default=DEFAULT_PROMPT_LAYERS,
                    help="提示侧常数施加的层：all（默认，方案原文=全部施加层）"
                         "| gen（与生成侧算子同层，两半施加面重合）| 层号写法")
    ap.add_argument("--gen-op", default="flatten", choices=list(RR.GEN_OPS),
                    help="生成侧算子：flatten = 把 top-m 削向均匀；"
                         "prompt_align = 把生成侧占比高于提示侧的 top-m 词串"
                         "逐串压回它们在提示侧的质量占比（只压不抬）")
    ap.add_argument("--top-m", type=int, default=8)
    ap.add_argument("--topm-rank", default=RR.DEFAULT_TOPM_RANK,
                    choices=list(RR.TOPM_RANKS),
                    help="top-m 的名额按什么排：gen_mass = 生成侧质量（初版）；"
                         "share_diff = 两侧占比差（生成侧占比 − 提示侧占比）")
    ap.add_argument("--flatten-strength", type=float, default=1.0)
    ap.add_argument("--a-min", type=float, default=0.05,
                    help="削峰算子的系数下限（对齐算子不设量程限制）")
    ap.add_argument("--b-max", type=float, default=5.0,
                    help="削峰算子的回填上限（同上）")
    ap.add_argument("--stop-after-think", dest="stop_after_think",
                    action="store_true", default=True,
                    help="思考段结束（</think>）之后停止抑制（默认开）")
    ap.add_argument("--suppress-after-think", dest="stop_after_think",
                    action="store_false", help="思考段结束后继续抑制（旧行为）")
    ap.add_argument("--ca-hist-every", type=int, default=32,
                    help="逐步诊断的落盘间隔（步）")
    ap.add_argument("--probe-every", type=int, default=32)
    ap.add_argument("--tau-prob", type=float, default=0.5)
    ap.add_argument("--off-rule", default="argmax", choices=OFF_RULES,
                    help="latch = 触发后锁死，不再复检、不解除；\n"
                         "默认 argmax = 复检判出不再是攻击类就解除")
    ap.add_argument("--burst-steps", type=int, default=8,
                    help="steps suppressed per loop verdict (0 = continuous)")
    ap.add_argument("--temperature", type=float,
                    default=float(CONFIG.get("temperature", 0)),
                    help="sampling temperature; defaults to config.json "
                         "(0 = greedy). Only temperature is set — no top-p/top-k")
    ap.add_argument("--seed", type=int, default=0,
                    help="re-seeded per run, so sampled runs are reproducible")
    # --- AUSteer 对比基线 ---------------------------------------------------
    ap.add_argument("--au-plan", default=None,
                    help="AU 计划 JSON（austeer_localize.py 产出）；"
                         "跑 B1_austeer / B1g_austeer_gated 时必需")
    ap.add_argument("--au-class", default="union",
                    help="用计划里的哪一类 AU：union（默认）| "
                         "repetitive_reasoning | repetitive_string | auto")
    ap.add_argument("--au-top-k", type=int, default=None,
                    help="截到前 k 个 AU；不给则用计划里的全部")
    ap.add_argument("--au-alpha", type=float, default=None,
                    help="全局强度因子 α，覆盖计划里记录的值")
    ap.add_argument("--au-decode-only", dest="au_prefill",
                    action="store_false", default=True,
                    help="只在解码步施加，提示编码阶段放过")
    ap.add_argument("--au-arms", default="",
                    help="把 B1_austeer 按 (k, α) 展开：`50:0.5,100:1`")
    ap.add_argument("--trigger-checkpoints", default=None,
                    help="comma list; default = the deployed 256..4096 grid")
    ap.add_argument("--plan-every", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--utility-max-new-tokens", type=int, default=768)
    ap.add_argument("--n-attack", type=int, default=8,
                    help="每个攻击类别取几条；<=0 表示不截断")
    ap.add_argument("--all-attack", action="store_true",
                    help="S1 用攻击集的全部提示，而不是只用 known_good_ids")
    ap.add_argument("--n-benign", type=int, default=4, help="per benign category")
    ap.add_argument("--n-utility", type=int, default=0)
    ap.add_argument("--loop-n", type=int, default=CONFIG.get("times", 5))
    ap.add_argument("--sets", default="", help="comma filter on set names")
    ap.add_argument("--subsets", default="",
                    help="按数据集名过滤（逗号分隔，如 concat）；只对 --pipeline-test 的样本有意义")
    ap.add_argument("--only-ids", default="",
                    help="只跑这些攻击提示 id（逗号分隔）——单样例排查用，"
                         "配 --n-attack 0 免得先被截断")
    ap.add_argument("--shard", default="",
                    help="\"i/n\"：只跑第 i 个分片（1 基）。按 samples 顺序取步长 n，"
                         "所以各分片的集合构成基本一致。仅作用于样本选取，"
                         "不改变任何一条的跑法——用于在同一张卡上并行多个进程")
    ap.add_argument("--pipeline-test", default=None,
                    help="改用一键流程 test split 的全部提示："
                         "传 exp/detection/pipeline/<模型>，读 test/records/<数据集>/<类别>.jsonl")
    ap.add_argument("--benign-temperature", type=float, default=None,
                    help="良性两类的采样温度（默认沿用 --temperature）")
    ap.add_argument("--attack-temperature", type=float, default=None,
                    help="成环两类的采样温度（默认沿用 --temperature）")
    ap.add_argument("--max-total-len", type=int, default=0,
                    help="序列总长上限；>0 时每条的 max_new_tokens 取 "
                         "「上限 − 该条提示的 token 数」，与生成轨迹时的口径一致")
    ap.add_argument("--out-dir", default=None,
                    help="结果目录，默认 exp/suppression/eval/<模型>/<条件>；"
                         "条件由温度与生成上限决定（如 t0.0_cap16k）")
    a = ap.parse_args()

    model_name = Path(a.model).name
    out_dir = Path(a.out_dir or (RECUR_CODE / "exp" / "suppression" / "eval" /
                                 model_name / condition_name(a.temperature,
                                                             a.max_new_tokens)))
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = [x for x in ARMS if x["arm"] in a.arms.split(",")]
    if a.sweep_layers:
        arms = expand_sweep_arms(arms, a.sweep_layers)
    if a.combo_layers:
        arms = expand_combo_arms(arms, a.combo_layers)
    if a.dyn_arms:
        arms = expand_dyn_arms(arms, a.dyn_arms)
    if a.au_arms:
        arms = expand_au_arms(arms, a.au_arms)
    if any(x.get("suppressor") == "austeer" for x in arms) and not a.au_plan:
        raise SystemExit("有 AUSteer 臂但没给 --au-plan（austeer_localize.py 产出）")
    if a.alt_reference:
        arms = expand_altref_arms(arms, a.alt_reference)
    # 参考曲线：抑制臂的目标值来源，必需
    reference = a.reference or str(
        RECUR_CODE / "exp" / "detection_dataset" / "models" /
        f"attention_reference_{model_name}.json")
    if not Path(reference).exists():
        raise SystemExit(
            f"缺少参考曲线 {reference}；先跑 "
            f"`python -m recur_code.src.detection.attention_reference "
            f"--model {model_name}`")
    only_ids = {int(x) for x in a.only_ids.replace(" ", "").split(",") if x}
    if a.pipeline_test:
        samples = load_pipeline_test(Path(a.pipeline_test))
        print(f"[sets] 一键流程 test split：{len(samples)} 条去重提示 "
              f"<- {a.pipeline_test}")
    else:
        samples = load_sets(model_name, a.n_attack, a.n_benign, a.n_utility,
                            all_attack=a.all_attack)
    if a.sets:
        keep = set(a.sets.split(","))
        samples = [s for s in samples if s["set"] in keep]
    if a.subsets:
        keep = set(a.subsets.split(","))
        samples = [s for s in samples if s.get("subset") in keep]
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        assert 1 <= i <= n, f"--shard {a.shard} 越界"
        total = len(samples)
        samples = samples[i - 1::n]
        print(f"[shard] {i}/{n}：从 {total} 条里取 {len(samples)} 条")
    if only_ids:
        # 只作用于攻击集：良性/效用样本的 id 空间与攻击集不同，混着筛会误伤
        samples = [s for s in samples
                   if s["set"] != "S1_attack_loop" or s["id"] in only_ids]
        got = {s["id"] for s in samples if s["set"] == "S1_attack_loop"}
        missing = only_ids - got
        if missing:
            print(f"[eval] 警告：--only-ids 里这些 id 不在样本集里（可能不是 "
                  f"known_good，需要 --all-attack）：{sorted(missing)}")

    # 断点续跑：扫同条件目录下所有臂文件，(集合, id, 臂) 已在即跳过
    done = set()
    for f in sorted(out_dir.glob("*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
                done.add((r["set"], r["id"], r["arm"]))
            except Exception:
                pass

    print(f"[eval] model={model_name} samples={len(samples)} arms="
          f"{[x['arm'] for x in arms]} done={len(done)} -> {out_dir}")

    rt = SuppressionRuntime(
        a.model, gpu=a.gpu, mode="off", layers=a.layers,
        plan_every=a.plan_every, max_new_tokens=a.max_new_tokens,
        temperature=a.temperature, seed=a.seed,
        detector=a.detector, probe_every=a.probe_every, tau_prob=a.tau_prob,
        off_rule=a.off_rule, burst_steps=a.burst_steps,
        trigger_checkpoints=([int(x) for x in a.trigger_checkpoints.split(",")]
                             if a.trigger_checkpoints else None),
        reference=reference, ref_normalize=a.ref_normalize,
        ref_quantile=a.ref_quantile,
        sink_positions=a.sink_positions, prompt_cap=a.prompt_cap,
        prompt_direction=a.prompt_direction,
        prompt_target_offset=a.prompt_target_offset,
        flatten_layers=a.flatten_layers, prompt_layers=a.prompt_layers,
        gen_op=a.gen_op, top_m=a.top_m, topm_rank=a.topm_rank,
        flatten_strength=a.flatten_strength, a_min=a.a_min, b_max=a.b_max,
        ca_hist_every=a.ca_hist_every, stop_after_think=a.stop_after_think,
        dyn_stride=a.dyn_stride, dyn_window=a.dyn_window,
        au_plan=a.au_plan, au_class=a.au_class, au_top_k=a.au_top_k,
        au_alpha=a.au_alpha, au_prefill=a.au_prefill)

    t_start = time.time()
    for si, s in enumerate(samples, 1):
        rt.max_new_tokens = (a.utility_max_new_tokens if s["set"] == "S5_utility"
                             else a.max_new_tokens)
        # 类别相关的温度：轨迹本来就是良性 T=0.5、成环 T=0 生成的，
        # 评测不对齐这一点，比较的就不只是「有没有抑制」这一个变量。
        kind = s.get("kind")
        if kind == "benign" and a.benign_temperature is not None:
            rt.temperature = a.benign_temperature
        elif kind == "attack" and a.attack_temperature is not None:
            rt.temperature = a.attack_temperature
        else:
            rt.temperature = a.temperature
        if a.max_total_len > 0:
            # 与生成侧同一个口径：总长受限，所以生成预算 = 上限 − 本条提示长度。
            # 这里必须按套完对话模板后的长度算，模板本身也占 token。
            chat = rt.tokenizer.apply_chat_template(
                [{"role": "user", "content": s["prompt"]}],
                tokenize=False, add_generation_prompt=True)
            n_prompt = len(rt.tokenizer(chat, add_special_tokens=False).input_ids)
            rt.max_new_tokens = max(1, a.max_total_len - n_prompt)
        for arm in arms:
            key = (s["set"], s["id"], arm["arm"])
            if key in done:
                continue
            rt.configure_arm(mode=arm["mode"],
                             plan_source=arm.get("plan_source"),
                             layers=arm.get("layers", a.layers),
                             burst_steps=arm.get("burst_steps", a.burst_steps),
                             flatten_layers=arm.get("flatten_layers",
                                                    a.flatten_layers),
                             prompt_layers=arm.get("prompt_layers",
                                                   a.prompt_layers),
                             gen_op=arm.get("gen_op", a.gen_op),
                             prompt_cap=arm.get("prompt_cap", a.prompt_cap),
                             a_min=arm.get("a_min", a.a_min),
                             b_max=arm.get("b_max", a.b_max),
                             top_m=arm.get("top_m", a.top_m),
                             topm_rank=arm.get("topm_rank", a.topm_rank),
                             flatten_strength=arm.get("flatten_strength",
                                                      a.flatten_strength),
                             ref_normalize=arm.get("ref_normalize",
                                                   a.ref_normalize),
                             prompt_direction=arm.get("prompt_direction",
                                                      a.prompt_direction),
                             prompt_target_offset=arm.get(
                                 "prompt_target_offset",
                                 a.prompt_target_offset),
                             reference=arm.get("reference"),
                             off_rule=arm.get("off_rule", a.off_rule),
                             dyn_select=arm.get("dyn_select", ""),
                             suppressor=arm.get("suppressor"),
                             au_class=arm.get("au_class", a.au_class),
                             au_top_k=arm.get("au_top_k", a.au_top_k),
                             au_alpha=arm.get("au_alpha", a.au_alpha),
                             au_prefill=arm.get("au_prefill", a.au_prefill))
            r = rt.run(s["prompt"], meta={k: v for k, v in s.items()
                                          if k != "prompt"},
                       prefill=s.get("prefill", ""))
            r["arm"] = arm["arm"]
            try:
                looped = loop_check_for_category(r["full_text"], r["thinking"],
                                                 s["category"], a.loop_n)
                # 干预会把循环**搬家**：c 不设量程的那轮里，3002 的思考段被提前
                # 结束（模型吐出 </think>），循环整段挪进答案段，而
                # loop_check_for_category 对 repetitive_reasoning 只看思考段，
                # 于是报 looped=False。按类别选段是建数据集时的正确口径，但
                # 评测抑制时必须两段都查，否则「把循环挪个地方」会被记成打断。
                answer = r["full_text"].split("</think>")[-1]
                spans = {"thinking": bool(r["thinking"])
                         and loop_check(r["thinking"], a.loop_n),
                         "answer": len(answer) > 200
                         and loop_check(answer, a.loop_n)}
                r["loop_spans"] = spans
                r["looped_any"] = bool(looped) or any(spans.values())
            except Exception as e:
                looped = None
                r["loop_check_error"] = str(e)
            r["looped"] = looped
            r["uniq3"] = round(uniq_ngram_ratio(r["full_text"]), 4)
            if s["set"] == "S5_utility":
                pred = last_number(r["full_text"].split("</think>")[-1])
                r["pred"] = pred
                r["correct"] = (pred is not None and
                                pred == s["gold"].replace(",", ""))
            r.pop("prompt", None)
            with open(out_dir / f"{arm['arm']}.jsonl", "a") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[{si}/{len(samples)}] {s['set']:22s} id={s['id']:<6} "
                  f"{arm['arm']:20s} loop={looped}"
                  f"{'(挪到答案段)' if not looped and r.get('looped_any') else ''}"
                  f" gen={r['gen_tokens']:<5} "
                  f"eos={r['stopped_naturally']} "
                  f"gate={r.get('gate_open_L')} final={r.get('final_pred')} "
                  f"{('dyn=' + str(r.get('dyn_selected')) + ' ') if r.get('dyn_spec') else ''}"
                  f"p={r.get('final_attack_prob')} rel={r.get('n_release')} "
                  f"bursts={r.get('n_bursts')} supp={r.get('suppressed_frac')} "
                  # AUSteer 臂的进度看 au=（施加的层×前向次数）；注意力算子的
                  # L/mass/c/a 对它没有意义，所以那几项只在主方案臂上打印
                  + (f"au={r.get('au_applied')} "
                     if r.get("suppressor") == "austeer" else
                     f"L={r.get('apply_layers')} "
                     f"mass={r.get('supp_mass_applied_layer')} "
                     f"c={r.get('c_median')} a={r.get('a_median')} ")
                  + f"uniq3={r['uniq3']} {r['seconds']}s", flush=True)
    write_manifest(out_dir, model_name, a)
    print(f"[eval] done in {(time.time() - t_start) / 60:.1f} min -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
