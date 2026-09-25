"""在线循环抑制运行时 —— v5「按检测类别重塑注意力」的接线。

把 `row_reshape.py` 的行内算子（纯 numpy）真正放进一个活的解码回路：

    提示 ──▶ 手写贪心/采样解码（KV cache + eager attention）
         ──▶ 每一步，打过补丁的 `eager_attention_forward` 记下每层对当前查询
             token 的头平均注意力行；它们在层上的平均，就是整个检测栈所定义的
             那个「层平均行」
         ──▶ 分类器在部署检查点上判类（特征就从这一行里读，不额外花钱）
         ──▶ 判出循环后，门打开，类别定下参考曲线
         ──▶ 门开期间，在施加层上把 softmax 之后的注意力行乘上乘子并重归一，
             模型于是在被重塑过的注意力下继续解码。

乘子：两半，按检测类别选参考曲线
--------------------------------
门（分类器）不只决定**何时**压，还决定**按哪条参考曲线**压。乘子由两半组成，
两半的算子都在 `row_reshape.py` 里，目标值来自 `attention_reference.py`：

  0. **目标平移**（`--prompt-target-offset`，默认 `none` = 不平移）。下面第 1 步
     是逐步**对齐水平**：每一步都把当前行的 pmf 送到参考曲线在该位置的取值上。
     但开始抑制那一刻的实测值远离曲线，于是第一步就是一个跳变，此后实际 pmf
     轨迹的斜率必然高于参考曲线、抬升量整体偏大。`trigger` / `trigger_log` 把
     整条目标曲线平移触发点上的那个差值（线性 / 对数域），使起点 c = 1、之后
     实际轨迹与参考曲线**平行**。副作用见 `_offset_target`：平移后封顶档几乎
     不再触发，长提示上的强干预会退回到由曲线斜率决定的弱干预。
  1. **提示侧对齐正常思考**（`--prompt-layers` 给的层、所有头；默认 `all` =
     所有施加层，`gen` = 只在生成侧算子那几层上做，两半的施加面因此重合）。
     当前行两侧的每 token 平均强度之比
     ρ 与该模型正常思考轨迹在同一生成位置上的参考值之比，就是要乘的常数
     c = ρ_目标 / ρ_当前。判为**循环思考** → 对齐**有效反思**曲线（c > 1，把偏低的
     提示侧强度抬上去）；判为**循环字符** → 对齐**简洁推理**曲线（c < 1，把偏高的
     压下来）。闭式，无求解器。
  2. **生成侧算子**（只在 `--flatten-layers` 上，`--gen-op` 二选一）。
     `flatten`（削向均匀）：生成侧按 token 串聚合后质量最高的 top-m 个串，其全部
     出现位置乘 a < 1，其余生成位置乘 b > 1。
     `prompt_align`（对齐提示侧占比）：只动「top-m 词串」与「提示里出现过的词串」
     的**交集**，每个串各乘一个系数，使它在生成侧的质量占比等于它在**提示侧**的
     质量占比（例：某串生成侧占 25%、提示侧占 10% → 它的每个生成位置乘 0.4）；
     **只压不抬**：占比已不高于提示侧的串、以及提示中没出现过的串都**原样不动**，
     只有 top-m 之外的位置共用一个回填系数。系数不设量程限制。
     两者都保证**生成侧总质量精确不变**——正因如此，第 2 步不会破坏第 1 步解出的
     提示侧占比。
     top-m 的名额按什么发由 `--topm-rank` 决定：`gen_mass`（初版）按生成侧质量，
     `share_diff` 按「生成侧占比 − 提示侧占比」——前者的名额常被「提示里本来就
     重、因而压不动的词」占掉，后者把 m 个名额全发给真正被过度使用的串
     （`row_reshape.py` §2c）。

系数算在哪一行上，由 `--plan-source` 决定：

  **own_layer（v5 默认）—— 每层用自己这一层、这一步的行，算自己的系数。**
  调用点在 patched attention 内部，行取于 softmax 之后、乘子之前，所以计划与施加
  同层同步：**零滞后**。提示侧常数 c 在**所有施加层**上各算各的（目标值与层无关，
  但当前值逐层不同，解出的 c 因此逐层不同）；生成侧 token 抑制只在
  `--flatten-layers` 上算，且**词串集合与系数也各层独立**（top-m 按本层行的生成侧
  质量选，交集与提示侧占比也都读本层这一行）。施加在 l < 最后一层时，本层输出的
  改变经残差流改写第 l+1.. 层为当前位置写入的 K/V —— 干预具备跨步记忆。

  **layer_avg（G1–G5 的口径，`--plan-source layer_avg` 复现）**：系数算在**上一步**
  的 32 层平均行上，全体施加层共用同一组系数，滞后一步；位置集合按当前 token 列表
  重建，所以刚吐出的重复 token 不会漏。

**门与这两半无关**：分类器的特征定义在 32 层平均行上，那一行在最后一层才完整，
所以门始终读层平均、且本步的门状态只对下一步生效。它读的是**抑制前**的行（钩子
里的累加发生在乘子之前），因此改动抑制不会改动分类器的输入——抑制只通过「后续
生成出什么 token」影响检测。

三种模式（评测的臂）：
    off          A0_off —— 不干预，但照常测量（基线，也是「抑制有没有改变结局」
                 的分母：它记录的是完全不干预时检测器会怎么判）
    classifier   M1_suppress —— **部署形态的门**：训练好的四类检测器说了算
    always       B1_austeer —— 常开，没有在线决策。为 **AUSteer 对比基线** 而设
                 （见下）；`_active` 保持 False，于是检测器仍按「门未开」的协议在
                 五个部署检查点上跑，这条臂因此带着与 A0_off 同口径的诊断。

对比基线：AUSteer（`--suppressor austeer`）
------------------------------------------
Feng et al., ICLR 2026, *Fine-Grained Activation Steering*（arXiv:2602.04428）。
它与本文件原有的算子**正交**：不碰注意力行，而是按一份离线 AU 计划，对
`mlp.down_proj` / `self_attn.o_proj` 的**输入**里选中的 ≤100 个标量维度做
`x̂_i = x_i + γ_i x_i`。实现见 `austeer.py`，计划由
`script/experiments/austeer_localize.py` 产出。

算子与门因此构成一个 2×2，`--suppressor` × `--mode` 直接给出四条臂：

    A0_off   row_reshape × off        不干预
    M1       row_reshape × classifier 我们的方案
    B1       austeer     × always     论文原样的基线
    B1-g     austeer     × classifier 消融：它的算子 + 我们的门

（`row_reshape × always` 被显式拒绝：注意力算子要按检测类别选参考曲线，常开时
没有判类，乘子会静默退化成不干预——见 `_check_layer_sets`。）

门（mode `classifier`）
-----------------------
这个门对齐的是部署侧的检测器，而不是原始指标上的手调阈值：

  - every decode step the layer-averaged row of the current query yields BOTH
    detection features of `detector.py` at no extra cost —
    `attn_consistency_independent`（`attn_consistency.consistency`，一次 top-θ
    遍历）与 `prompt_mean_fraction`（同一行的提示侧质量）。
    No full-prefix forward is needed, unlike `detection_runtime.py`, because
    both features are per-row quantities;
  - the running features `(cons_ind_mean, pmf_slope, log L)` over gen positions
    [0, i] — the SAME running statistics the classifier was trained on — are
    accumulated every step and handed to the classifier on the probe grid;
  - **probe grid**: while the gate is CLOSED the detector runs only on the
    deployed checkpoints (`trigger_checkpoints`, 256/512/1024/2048/4096) — its
    own protocol, nothing to re-check yet. Once the gate is OPEN it runs every
    `probe_every` (32) tokens — **unless `off_rule=latch`, where it never runs
    again** (see below);
  - **on**: a confident attack call (argmax ∈ {repetitive_reasoning,
    repetitive_string} and p ≥ tau_prob) at a deployed checkpoint;
  - **burst**: opening the gate buys `burst_steps` (8) steps of suppression,
    then decoding runs free. Each re-check that still says "loop" buys another
    burst — so suppression covers burst_steps/probe_every of the steps, not all
    of them. `burst_steps=0` restores continuous suppression;
  - **off**: at a re-check the class is no longer a loop-attack class
    (`off_rule=argmax`, the literal reading of "no longer classified as a
    loop"; `off_rule=confident` additionally demands p ≥ tau_prob before
    releasing). The gate then waits for another deployed-checkpoint trigger.

简化协议（`off_rule=latch`，评测臂 H1）
--------------------------------------
一条单向轨道，去掉上面所有反复横跳的状态：

    检测只在 256/512/1024/2048/4096 这五个位置进行
      → 任一位置判出循环（argmax ∈ 攻击类且 p ≥ tau_prob）
      → 从下一步起逐步抑制，直到 EOS 或生成上限
      → 期间不再调用检测器：无复检、无解除、无类别改判

类别在触发那一刻定下（class_aware 的参考曲线随之固定）。`burst_steps`
在这条规则下被强制置 0，否则突发预算永远得不到补充，等于「压几步就
永久停手」。

这条规则不影响逐步指标的采集：`cons_hist` / `pmf_hist` / `ca_hist` 照旧每步
（或每 32 步）落盘。它们不进入任何在线决策，只是事后复盘的原始材料；
去掉它们会让「抑制期间轨迹到底发生了什么」变成不可观测。

Why the probe reads the PRE-suppression row: if it read the suppressed row the
gate would be scoring its own output — scaling the loop token down mechanically
raises consistency, so it would always release one probe after firing. Reading
the raw row means the release can only come from the model actually generating
differently, which is the claim under test. The already-applied suppression
still influences the number, through the tokens it caused to be emitted.

Runs in the `Recur` env (torch + transformers 4.52, no sklearn needed — pass
the JSON export of the classifier from `detector.py --export`)::

    PYTHONPATH=. /root/miniconda3/envs/Recur/bin/python \\
        -m recur_code.src.detection.suppression_runtime \\
        --model $RC_MODELS_ROOT/DeepSeek-R1-Distill-Llama-8B \\
        --detector recur_code/exp/detection_dataset/models/loop_detector_llama8b_4class_method2.json \\
        --reference recur_code/exp/detection_dataset/models/attention_reference_DeepSeek-R1-Distill-Llama-8B.json \\
        --gpu 1 --mode classifier --prompt "..." \\
        --plan-source own_layer --prompt-layers gen --flatten-layers 15 \\
        --gen-op prompt_align --topm-rank share_diff --top-m 3 --burst-steps 0

不加载模型的自测：`... --selftest --reference <曲线 json>`
"""

from __future__ import annotations

import importlib
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from recur_code.src.detection.detector import (  # noqa: E402
    ATTACK, CHECKPOINTS, DEFAULT_TAU_PROB, MIN_GEN_FOR_SLOPE, features_at,
    load_detector,
)
from recur_code.src.detection.attn_consistency import consistency  # noqa: E402
from recur_code.src.detection.attention_reference import (  # noqa: E402
    DEFAULT_NORMALIZE, EXPECTED_DIRECTION, NORMALIZE_MODES, REFERENCE_OF,
    AttentionReference,
)
from recur_code.src.detection import row_reshape as RR  # noqa: E402
from recur_code.src.detection import austeer as AU  # noqa: E402

THINK_CLOSE = "</think>"


def _rel_to_repo(path: str | None) -> str | None:
    """路径尽量写成仓库相对形式，结果文件换机器仍可读。"""
    if not path:
        return None
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)

# W1 fidelity ranking (exp/suppression/layer_ls_closest_v{2,3}.json, aggregated
# by mean rank of the per-checkpoint least-squares distance to the layer
# average). Top-8 of v3; v2 gives the same set up to order, and 7 of the 8 are
# in both top-8 lists, so the ranking is stable across sample sets. Only usable
# in the lagged regime — see the module docstring.
CLOSEST8_LLAMA8B = [17, 6, 20, 28, 26, 18, 21, 16]

# 门的三种形态。`off` / `classifier` 是我们自己的两条臂（不干预的基线、部署形态
# 的分类器门）；`always` 是 2026-09-20 为 **AUSteer 对比基线** 重新引入的常开门
# ——那条基线按定义没有在线决策，从第 0 步压到结束。
# （历史上的原始指标窗口门 gated 与随机对照随旧乘子一起删除，没有恢复：随机对照
#   在现行乘子上无处安放，乘子按注意力行的分布解，不是「选若干位置乘同一系数」。）
MODES = ("off", "classifier", "always")
# 干预算子。两者正交于门，构成「算子 × 门」的 2×2：
#   row_reshape —— 我们的主方案，重塑 softmax 之后的注意力行
#   austeer     —— 对比基线，按离线 AU 计划缩放线性层输入的若干标量维度
#                  （arXiv:2602.04428，见 `austeer.py`）
SUPPRESSORS = ("row_reshape", "austeer")
# AUSteer 计划里可选的类别：两类循环各一套 AU，`union` 是合并，`auto` 表示
# 按分类器在门开那一刻判定的类别现选（只在 mode=classifier 下有意义）。
AU_CLASS_AUTO = "auto"
OFF_RULES = ("argmax", "confident", "latch")
# 参考曲线给的 pmf 目标是否平移到「开始抑制那一刻的实测水平」，见 `_offset_target`：
#   none        —— 逐步命中曲线本身的水平（原口径）。起点上实测远低于曲线，
#                  第一步就是一个跳变，之后实际曲线的斜率必然高于参考曲线。
#   trigger     —— 线性平移：目标 = 曲线(t) − (曲线(t0) − 实测(t0))
#   trigger_log —— 对数域平移（= 线性域按比例缩放），与曲线自身的残差分位数
#                  平移（`attention_reference._resid_offset`）同一口径，且恒为正
PROMPT_TARGET_OFFSETS = ("none", "trigger", "trigger_log")
# 指标（以及词集与 k）算在哪一行上：
#   layer_avg —— 全部层的平均行，只有跑完第 31 层才完整，因此只有末层能零滞后
#   own_layer —— 每个施加层用它自己的行，计划与施加同层同步，零滞后且不限层
PLAN_SOURCES = ("layer_avg", "own_layer")
THINK_END = "</think>"          # 思考段结束标记（stop_after_think 的触发条件）
# 生成侧削峰的默认层带：第 8–14 层（32 层模型）的**非汇可用质量**最高
# （0.39–0.47，末尾几层只有 0.21–0.22），按深度比例折算成 mid:<n>。
# 注：主方案臂实测最优是单独第 15 层（`--flatten-layers 15`），这个默认值只在
# 没显式给层时生效。
DEFAULT_FLATTEN_LAYERS = "mid:8"
# 提示侧常数施加在哪些层：`all`（方案原文：所有层所有头）| `gen` = 与生成侧算子
# 同一批层（两半施加面重合）| `_resolve_layers` 的任何层号写法。永远与施加层取交。
DEFAULT_PROMPT_LAYERS = "all"
MID_BAND_CENTER = 0.35          # mid:<n> 的中心深度（8–14 层 / 32 层 ≈ 0.34）
# 动态选层：门开的那一刻，用**每层自己的行**算出的一致性指标（逐层版的
# attn_consistency_independent）在候选层里挑 n 层当生成侧施加面，之后冻结。
#   cons_low  —— 取一致性最低的 n 层（假设：本层提示侧/生成侧分歧最大 =
#                这一层最「陷在循环里」，也最值得压）
#   cons_high —— 反向对照
#   random    —— 同规模随机对照（选层无信息时的基线）
# 上面三条是「门开时排序取前 n 层再冻结」，要等本步所有候选层都算完才知道名次。
# 下面两条是**逐步逐层**判定：走到第 li 层、这一行刚出来时就地决定压不压，
# 不需要知道别的层的值，也没有一步滞后——施加面每一步都可能不同。
#   thresh    —— 本层当前一致性 < τ 就压（τ 是绝对阈值，取值同指标，0–1）
#   reldrop   —— 本层当前一致性 < ρ × **本层门开前的自身均值** 就压。把静态的
#                层间剖面与样例整体尺度都除掉，只看「这一层比它自己平时低多少」
DYN_RULES = ("cons_low", "cons_high", "random", "thresh", "reldrop")
DYN_PERSTEP_RULES = ("thresh", "reldrop")


def _resolve_layers(spec: str, n_layers: int, rng: np.random.Generator) -> list[int]:
    """`last` | `all` | `none` | `closest8` | `random:<n>` | `mid:<n>` |
    逗号分隔的层号，其中每项可以是 `a-b` 闭区间。"""
    if spec in ("none", ""):
        return []
    if spec == "last":
        return [n_layers - 1]
    if spec == "all":
        return list(range(n_layers))
    if spec == "closest8":
        return [l for l in CLOSEST8_LLAMA8B if l < n_layers]
    if spec.startswith("random:"):
        n = int(spec.split(":", 1)[1])
        return sorted(rng.choice(n_layers, size=min(n, n_layers), replace=False).tolist())
    if spec.startswith("mid:"):
        # 以深度比例定位，跨模型可复用（32 层 → 7..14，64 层 → 18..25）
        n = min(int(spec.split(":", 1)[1]), n_layers)
        start = int(round(MID_BAND_CENTER * n_layers - n / 2))
        start = max(0, min(start, n_layers - n))
        return list(range(start, start + n))
    out: set[int] = set()
    for item in spec.replace(" ", "").split(","):
        if not item:
            continue
        if "-" in item.lstrip("-"):
            lo, hi = item.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(item))
    return sorted(out)


class SuppressionRuntime:
    """Greedy decode under a target LLM with per-step attention suppression."""

    def __init__(self, model_path: str, *, gpu: str = "1", mode: str = "classifier",
                 layers: str | None = None,
                 plan_every: int = 8, threshold: float = 0.9,
                 max_new_tokens: int = 2048,
                 temperature: float = 0.0, dtype: str = "bfloat16",
                 same_step: bool | None = None, seed: int = 0,
                 detector: str | None = None, probe_every: int = 32,
                 tau_prob: float = DEFAULT_TAU_PROB,
                 off_rule: str = "argmax", burst_steps: int = 8,
                 trigger_checkpoints: list[int] | None = None,
                 plan_source: str | None = None,
                 reference: str | None = None,
                 ref_normalize: str = DEFAULT_NORMALIZE,
                 ref_quantile: float = 0.5,
                 sink_positions: int = 1,
                 prompt_target_offset: str = "none",
                 prompt_cap: float = 0.0, prompt_direction: str = "both",
                 flatten_layers: str = DEFAULT_FLATTEN_LAYERS,
                 prompt_layers: str = DEFAULT_PROMPT_LAYERS,
                 gen_op: str = "flatten",
                 top_m: int = 8, topm_rank: str = RR.DEFAULT_TOPM_RANK,
                 flatten_strength: float = 1.0,
                 a_min: float = 0.05, b_max: float = 5.0,
                 flatten_group_by: str = "string",
                 stop_after_think: bool = True,
                 ca_hist_every: int = 32,
                 ca_hist_layers: str = "sample",
                 dyn_select: str = "", dyn_stride: int = 8,
                 dyn_window: int = 0,
                 suppressor: str = "row_reshape",
                 au_plan: str | None = None,
                 au_class: str = AU.UNION,
                 au_top_k: int | None = None,
                 au_alpha: float | None = None,
                 au_prefill: bool = True):
        assert mode in MODES, f"mode must be one of {MODES}"
        assert suppressor in SUPPRESSORS, \
            f"suppressor must be one of {SUPPRESSORS}"
        assert gen_op in RR.GEN_OPS, f"gen_op must be one of {RR.GEN_OPS}"
        assert topm_rank in RR.TOPM_RANKS, \
            f"topm_rank must be one of {RR.TOPM_RANKS}"
        assert prompt_target_offset in PROMPT_TARGET_OFFSETS, \
            f"prompt_target_offset must be one of {PROMPT_TARGET_OFFSETS}"
        # 两半都算在**各层自己的行**上，所以默认逐层；显式传 layer_avg 可以复现
        # 「系数算在上一步的层平均行上」的滞后一步口径。
        if plan_source is None:
            plan_source = "own_layer"
        if layers is None:
            # 提示侧常数按方案定义可落在任何层；实际施加面由 prompt_layers /
            # flatten_layers 取交决定。
            layers = "all"
        assert plan_source in PLAN_SOURCES, \
            f"plan_source must be one of {PLAN_SOURCES}"
        assert off_rule in OFF_RULES, f"off_rule must be one of {OFF_RULES}"
        if mode == "classifier":
            if not detector:
                raise ValueError("mode=classifier needs --detector (the JSON export)")
            if suppressor == "row_reshape" and not reference:
                raise ValueError("mode=classifier 需要 --reference "
                                 "（attention_reference.py 生成的参考曲线）")
        if suppressor == "austeer" and not au_plan:
            raise ValueError("suppressor=austeer 需要 --au-plan "
                             "（austeer_localize.py 产出的 AU 计划 JSON）")
        # `--gpu cpu` 只为自测留：小模型在 CPU 上跑通接线，不占生产卡。
        self.device_map = "cpu" if str(gpu).lower() == "cpu" else "cuda:0"
        if self.device_map != "cpu":
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.mode = mode
        self.plan_every = plan_every
        self.threshold = threshold
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # --- 类别自适应乘子 ---------------------------------------------------
        # 提示侧：把两侧每 token 平均强度之比 ρ 对齐到「正常思考」参考曲线
        # （循环思考 → 有效反思，循环字符 → 简洁推理）；生成侧：把被过度使用的
        # top-m 词串压回提示侧占比，总质量不变。两个算子都在 row_reshape.py 里。
        self.reference_path = reference
        self.reference = AttentionReference.load(reference) if reference else None
        # 目标量的口径由曲线文件自己声明（`AttentionReference.load` 会拒绝
        # 口径对不上的旧文件）；曲线目标越界那一档由 row_reshape 的封顶占比
        # `SHARE_CEILING` 接手，与曲线无关。
        self.ref_metric = (self.reference.metric if self.reference
                           else RR.PROMPT_METRIC)
        self.ref_normalize = ref_normalize
        self.ref_quantile = ref_quantile
        self.sink_positions = sink_positions
        # 曲线目标是否平移到「开始抑制那一刻的实测水平」（见 `_offset_target`）。
        # none = 原口径（逐步命中曲线的水平）；trigger/trigger_log = 平移，
        # 起点不跳、之后与参考曲线平行。
        self.prompt_target_offset = prompt_target_offset
        # prompt_cap ≤ 0 表示**不给 c 设量程**（默认，评测臂用的就是这个）。
        # c 恒为「目标/当前」这个有限值，不再有「目标够不着」的兜底分支，
        # 见 row_reshape §1。
        self.prompt_cap = prompt_cap
        self.prompt_direction = prompt_direction
        self.flatten_spec = flatten_layers
        # 提示侧常数施加在哪些层。`gen` = 与生成侧算子同层（两半施加面重合），
        # 见 `_resolve_prompt_layers`。
        self.prompt_spec = prompt_layers
        # 生成侧算子：flatten = 把 top-m 削向均匀；prompt_align = 把「top-m ∩
        # 提示里出现过的词串」逐串对齐到它们在提示侧的质量占比（row_reshape §2b）
        self.gen_op = gen_op
        self.top_m = top_m
        # top-m 的名额按生成侧质量发（gen_mass）还是按两侧占比差发（share_diff，
        # row_reshape §2c）
        self.topm_rank = topm_rank
        self.flatten_strength = flatten_strength
        # a_min / b_max 只作用于削峰算子；对齐算子按定义不设量程限制
        self.a_min = a_min
        self.b_max = b_max
        self.flatten_group_by = flatten_group_by
        # 思考段结束（吐出 </think>）之后是否停止抑制。干预可能把思考段逼得提前
        # 收尾，随后循环整段挪进答案段——那时继续压等于在压一段本不该干预的
        # 文本；置 True 时 </think> 一出现就永久关闭干预（检测照常记录）。
        self.stop_after_think = bool(stop_after_think)
        # 逐步诊断的落盘间隔：16k 长度下每 8 步一条会让单条结果多出几百 KB
        self.ca_hist_every = max(1, int(ca_hist_every))
        # 逐层路径下每步有 32 条诊断（层平均路径只有 1 条），全记会把结果撑到
        # 几百 MB，所以按层抽样，见 `_resolve_hist_layers`。
        self.ca_hist_spec = ca_hist_layers
        self.dyn_spec = dyn_select or ""
        self.dyn_stride = max(1, int(dyn_stride))
        self.dyn_window = int(dyn_window)

        # --- AUSteer 对比基线 --------------------------------------------------
        # 计划是离线产物（austeer_localize.py），整个运行期不变；k 与 α 可以按臂
        # 覆盖，所以钩子在 `_install_au` 里按当前配置重建。
        self.suppressor = suppressor
        self.au_plan_path = str(au_plan) if au_plan else None
        self.au_doc = AU.load_plan_file(au_plan) if au_plan else None
        self.au_class = au_class
        self.au_top_k = au_top_k
        self.au_alpha = au_alpha
        self.au_prefill = bool(au_prefill)
        self._au_hooks: list = []
        self._au_plans: dict[str, AU.AUPlan] = {}

        # --- deployed-detector gate ------------------------------------------
        self.probe_every = probe_every
        self.recent_window = 64          # diagnostic window, not a gate input
        self.tau_prob = tau_prob
        self.off_rule = off_rule
        self.burst_steps = burst_steps   # 0 = suppress the whole interval
        if off_rule == "latch" and self.burst_steps:
            # latch 下没有复检，突发预算永远不会被补充——留着 burst_steps>0
            # 等于「压 burst_steps 步后永久停手」，与这条规则的语义相反。
            print(f"[gate] off_rule=latch → burst_steps {self.burst_steps} → 0"
                  f"（锁定后连续抑制到结束）")
            self.burst_steps = 0
        # The gate may only OPEN on the deployed detection grid; the every-32
        # re-checks that keep it open (or release it) are a finer grid laid on
        # top. CHECKPOINTS are all multiples of 32, so the trigger grid is a
        # subset of the re-check grid and no trigger can be missed.
        self.trigger_checkpoints = set(trigger_checkpoints or CHECKPOINTS)
        off_grid = [c for c in self.trigger_checkpoints if c % probe_every]
        if off_grid:
            raise ValueError(f"trigger checkpoints {sorted(off_grid)} are not "
                             f"multiples of probe_every={probe_every}, so they "
                             f"would never be probed")
        self.detector_path = detector
        self.det = load_detector(detector) if detector else None
        # With a detector attached the features must be the running statistics
        # over EVERY gen position (that is how the classifier was trained), so
        # the row is measured every step; without one, only at plan cadence.
        self.measure_every = 1 if self.det is not None else plan_every

        print(f"[supp] loading {model_path} (eager, {dtype}) ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=getattr(torch, dtype),
            attn_implementation="eager",     # required: we patch the eager primitive
            device_map=self.device_map)
        self.model.eval()
        self.n_layers = self.model.config.num_hidden_layers
        self.last_layer = self.n_layers - 1
        self.layers_spec = layers
        self.apply_layers = set(_resolve_layers(layers, self.n_layers, self.rng))
        self.flatten_layers = set(_resolve_layers(flatten_layers, self.n_layers,
                                                  self.rng)) & self.apply_layers
        self.prompt_layers = self._resolve_prompt_layers()
        self.ca_hist_layers = self._resolve_hist_layers()

        # 时序由「计划算在哪一行上」决定，见 _resolve_timing。
        self.plan_source = plan_source
        self._same_step_override = same_step
        self.same_step = self._resolve_timing()
        self._resolve_dyn()
        self._check_layer_sets()

        # 停止符取 tokenizer 与 generation_config 的并集。GLM 用 <|user|> /
        # <|observation|> 结束回合、QwQ 还有 <|endoftext|>，这些只登记在
        # generation_config 里（vLLM 生成轨迹时读的就是它）；只认
        # tokenizer.eos_token_id 会让模型答完后接着写到长度上限。
        gen_eos = getattr(self.model.generation_config, "eos_token_id", None)
        eos = {self.tokenizer.eos_token_id}
        eos.update(gen_eos if isinstance(gen_eos, (list, tuple)) else [gen_eos])
        self.eos_ids = sorted(i for i in eos if i is not None)
        # `</think>` is deliberately NOT a stop token: repetitive_string
        # degenerates in the ANSWER (see _common.loop_check_for_category).

        self._state: dict[str, Any] = {}
        self._reset_run_state()
        self._install_patch()
        self._install_au()
        if reference:
            print(f"[supp] 参考曲线={Path(reference).name} "
                  f"口径={ref_normalize} 分位={ref_quantile} "
                  f"汇诊断位数={sink_positions} "
                  f"上限={prompt_cap} 方向={prompt_direction} "
                  f"目标平移={prompt_target_offset}")
            print(f"[supp]   提示侧层={self.prompt_spec}"
                  f"{sorted(self.prompt_layers)} "
                  f"生成侧层={sorted(self.flatten_layers)}")
            print(f"[supp]   top_m={top_m}(名额按 {topm_rank} 排) "
                  f"强度={flatten_strength} "
                  f"a_min={a_min} b_max={b_max} 聚合={flatten_group_by}")
        print(f"[supp] mode={mode} layers={sorted(self.apply_layers)} "
              f"plan_source={self.plan_source} "
              f"timing={'same_step' if self.same_step else 'lagged'} "
              f"plan_every={plan_every}")
        if self.det is not None:
            print(f"[supp] detector={Path(detector).name} "
                  f"classes={self.det.classes} probe_every={probe_every} "
                  f"tau_prob={tau_prob} off_rule={off_rule} "
                  f"burst_steps={burst_steps} "
                  f"trigger_at={sorted(self.trigger_checkpoints)}")

    def _resolve_prompt_layers(self) -> set[int]:
        """提示侧常数施加在哪些层（`--prompt-layers`）。

        `all`（默认，方案原文口径）= 全部施加层；`gen`/`flatten` = **与生成侧算子
        同一批层**，即两半的施加面重合——提示侧常数与 token 抑制作用在同一处，
        其余层完全不动；也可以直接给层号（`_resolve_layers` 的语法）。永远与
        `apply_layers` 取交：不在施加层里的乘子不会被用到。"""
        spec = getattr(self, "prompt_spec", DEFAULT_PROMPT_LAYERS)
        if spec in ("gen", "flatten"):
            return set(self.flatten_layers)
        return set(_resolve_layers(spec, self.n_layers, self.rng)) \
            & self.apply_layers

    def _resolve_dyn(self) -> None:
        """解析动态选层规格 `<规则>:<层数 n>:<候选层>`（如 `cons_low:2:11-18`）。

        空串 = 关闭，施加层由 `--flatten-layers` 静态给定（原行为）。开启时**门开
        之前施加层是空集**：候选层的逐层一致性一路测着，门一开就按规则定层并冻结，
        所以整条轨迹只有一路输出，不需要分叉探测。
        """
        spec = (getattr(self, "dyn_spec", "") or "").strip()
        self.dyn_rule = None
        self.dyn_n = 0
        self.dyn_perstep = False
        self.dyn_thresh = float("nan")
        self.dyn_cands: list[int] = []
        if not spec:
            return
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"dyn_select 要写成 <规则>:<n>:<候选层>，收到 {spec!r}")
        rule, param, cand_spec = parts
        if rule not in DYN_RULES:
            raise ValueError(f"dyn_select 的规则必须是 {DYN_RULES} 之一，收到 {rule!r}")
        cands = [l for l in _resolve_layers(cand_spec, self.n_layers, self.rng)
                 if l in self.apply_layers]
        if not cands:
            raise ValueError(f"dyn_select 的候选层 {cand_spec!r} 解出来是空集")
        self.dyn_rule = rule
        self.dyn_cands = cands
        self.dyn_perstep = rule in DYN_PERSTEP_RULES
        if self.dyn_perstep:
            self.dyn_thresh = float(param)
            self.dyn_n = 0
            # 逐步逐层判定：候选层全部进施加面，压不压由每一步各自的判定决定
            self.flatten_layers = set(cands) & self.apply_layers
        else:
            self.dyn_thresh = float("nan")
            self.dyn_n = max(1, min(int(param), len(cands)))
            # 门开之前什么都不压：静态的 flatten_layers 让位给动态结果
            self.flatten_layers = set()
        self.prompt_layers = self._resolve_prompt_layers()
        self.ca_hist_layers = self._resolve_hist_layers()

    def _dyn_measure(self, li: int, row_t) -> None:
        """记下第 li 层这一步的**逐层一致性指标**（只在门开之前需要）。

        算子与门读的那个完全相同（`attn_consistency.consistency`：把 top-0.9 质量
        按词串分组，提示侧/生成侧各归一后取 1 − L1 差，值域 0–1，越低表示这一层
        在生成侧读到的词串分布越偏离它在提示里的样子），只是行从「32 层平均」
        换成了**本层自己的头平均行**——而这一行在补丁后的注意力
        里每层每步都已经在手上，所以单路输出就能算，代价是每 dyn_stride 步对每
        个候选层做一次 top-θ 遍历。"""
        q = len(self._tokens) - 1
        if q < 0:
            return
        row = row_t.cpu().numpy().astype(np.float64)[: q + 1]
        cons = consistency(row, self._prompt_len, self._tokens,
                           threshold=self.threshold)
        if np.isfinite(cons):
            self._dyn_hist[li].append(round(float(cons), 4))

    def _dyn_gate_layer(self, li: int, row: np.ndarray) -> bool:
        """逐步逐层判定：**就在第 li 层这一行刚算出来的时候**决定这一步压不压它。

        这是与 `cons_low` 那类规则的本质区别——名次要等本步所有候选层都算完才
        知道（对第 11 层来说等于用了未来信息，实现上只能滞后一步），而阈值只用
        本层此刻的值，走到哪层判哪层，施加面逐步变化。

        规则：
          thresh   本层一致性 < τ
          reldrop  本层一致性 < ρ × 本层**门开前**的自身均值（基线在门开那一刻
                   冻结，之后不再更新——否则被压过的步会把基线拖下去，形成
                   「压得越多、门槛越低」的自反馈）
        """
        q = len(self._tokens) - 1
        if q < 0:
            return False
        cons = consistency(row, self._prompt_len, self._tokens,
                           threshold=self.threshold)
        self._dyn_evals[li] = self._dyn_evals.get(li, 0) + 1
        if not np.isfinite(cons):
            return False
        if self.dyn_rule == "thresh":
            hit = cons < self.dyn_thresh
        else:                                    # reldrop
            base = self._dyn_base.get(li)
            if base is None or base <= 0:
                return False
            hit = cons < self.dyn_thresh * base
        if hit:
            self._dyn_hits[li] = self._dyn_hits.get(li, 0) + 1
            self._dyn_hit_steps.add(self._step)
        return bool(hit)

    def _dyn_decide(self) -> None:
        """门开的那一刻定下生成侧施加层，之后不再改。

        取每个候选层到此刻为止的逐层一致性均值（dyn_window > 0 时只取最近这么
        多次测量），按规则排序取前 n 层。提示侧层照 `--prompt-layers` 的语义跟随
        （`gen` 时两半重合）。"""
        if not self._dyn_pending:
            return
        self._dyn_pending = False
        if self.dyn_perstep:
            # 逐步规则不在这里定层：只把每层的基线冻结下来（reldrop 用），
            # 施加面已经是全部候选层，压不压逐步逐层判。
            self._dyn_base = {l: float(np.mean(v))
                              for l, v in self._dyn_hist.items() if v}
            self._dyn_means = {str(l): round(v, 4)
                               for l, v in sorted(self._dyn_base.items())}
            self._dyn_step = self._step
            return
        w = self.dyn_window
        means = {l: float(np.mean(v[-w:] if w > 0 else v))
                 for l, v in self._dyn_hist.items() if v}
        cands = sorted(self._dyn_hist)
        if self.dyn_rule == "random":
            sel = sorted(int(x) for x in self.rng.choice(
                cands, size=min(self.dyn_n, len(cands)), replace=False))
        elif means:
            order = sorted(means, key=lambda l: means[l],
                           reverse=(self.dyn_rule == "cons_high"))
            sel = sorted(order[: self.dyn_n])
        else:
            # 一次都没测到（门在第一次测量之前就开了）——退回候选层的前 n 层，
            # 并把这件事记在结果里（dyn_cons_by_layer 为空）。
            sel = cands[: self.dyn_n]
        self._dyn_selected = sel
        self._dyn_step = self._step
        self._dyn_means = {str(l): round(v, 4) for l, v in sorted(means.items())}
        self.flatten_layers = set(sel) & self.apply_layers
        self.prompt_layers = self._resolve_prompt_layers()
        self.ca_hist_layers = self._resolve_hist_layers()

    # ------------------------------------------------------- AUSteer 基线
    def _install_au(self) -> None:
        """按当前配置（类别、k、α）重建 AUSteer 的钩子。

        `au_class=auto` 时为计划里的每个类别各挂一套，钩子自己判断当前类别是否
        是它那一类——这样分类器门的消融臂（AUSteer 算子 + 我们的门）能按门开那
        一刻定下的类别选 AU 集合，而不必中途重挂。`mode=always` 下没有在线判类，
        所以 `auto` 在那里被拒绝。
        """
        for h in self._au_hooks:
            h.remove()
        self._au_hooks, self._au_plans = [], {}
        if self.suppressor != "austeer":
            return
        if self.au_doc is None:
            raise ValueError("suppressor=austeer 但没有加载 AU 计划")
        if self.au_class == AU_CLASS_AUTO:
            if self.mode != "classifier":
                raise ValueError("au_class=auto 需要 mode=classifier"
                                 "（类别由门开那一刻的判定给出）")
            classes = [c for c in (self.au_doc.get("plans") or {})
                       if c != AU.UNION] or [AU.UNION]
        else:
            classes = [self.au_class]
        for cls in classes:
            plan = AU.select_plan(self.au_doc, cls, top_k=self.au_top_k,
                                  alpha=self.au_alpha)
            if not plan.entries:
                raise ValueError(f"AU 计划里类别 {cls} 是空的")
            self._au_plans[cls] = plan
            if self.au_class == AU_CLASS_AUTO:
                def gate(_cls=cls):
                    return (self._suppress_this_step()
                            and self._attack_class == _cls)
            else:
                gate = self._suppress_this_step
            self._au_hooks.append(AU.AUSteerHooks(
                self.model, plan, gate=gate, apply_prefill=self.au_prefill))
        first = next(iter(self._au_plans.values()))
        print(f"[au] 计划={Path(self.au_plan_path).name} "
              f"变体={first.variant} 类别={self.au_class} "
              f"α={first.alpha} #Acts={ {c: p.n_acts for c, p in self._au_plans.items()} } "
              f"prefill={'施加' if self.au_prefill else '放过'}")

    def _au_applied(self) -> int:
        """本次 run 里 AUSteer 钩子实际施加的 (层, 前向) 次数。"""
        return sum(h.n_applied for h in self._au_hooks)

    def _check_layer_sets(self) -> None:
        """层平均路径要求生成侧的层落在提示侧的层里面。

        那条路径每步只产出两个乘子（提示侧层用一个、生成侧层用一个，后者是在
        提示侧常数之上叠加生成侧算子得到的）。某个生成侧层若不在提示侧层里，就
        需要第三个「只有生成侧」的乘子——与其悄悄按两个乘子里的任一个施加，不如
        在配置阶段直接拒绝。逐层路径（own_layer，v5 默认）每层各建各的乘子，
        没有这个限制。"""
        if self.mode == "always" and self.suppressor == "row_reshape":
            # 注意力算子按**检测类别**选参考曲线，常开时没有判类，乘子会静默地
            # 退化成「什么都不做」（`_class_aware_layer_multiplier` 的首行）。
            # 与其让这条臂看起来在跑、实际等于 A0_off，不如直接拒绝。
            # 「我们的算子常开」那条消融要先决定用哪条参考曲线（例如按真值类别
            # 强制），那是另一个设计问题，不在这里默认。
            raise ValueError("mode=always 目前只支持 suppressor=austeer；"
                             "注意力算子常开需要先指定参考曲线的类别")
        if self.suppressor == "austeer" or not self.attention_action:
            # 注意力行完全不动（施加面为空是这条臂的**定义**，不是配置错误）：
            # 干预全部发生在 FFN / o_proj 的输入维度上。动态选层同理关闭——
            # 那是给注意力算子选施加层的，AUSteer 的施加面由计划自己给定。
            self.prompt_layers = set()
            self.flatten_layers = set()
            self.dyn_rule = None
            self.dyn_cands = []
            return
        if not (self.prompt_layers or self.flatten_layers or self.dyn_rule):
            raise ValueError(
                f"乘子的两半都没有施加层（--prompt-layers "
                f"{self.prompt_spec} 与 --flatten-layers {self.flatten_spec} "
                f"解出来都是空集），这条臂等于不干预；要跑不干预的基线请用 "
                f"mode=off，而不是把层配空")
        if (self.plan_source == "layer_avg"
                and self.flatten_layers - self.prompt_layers):
            raise ValueError(
                f"plan_source=layer_avg 下生成侧层 "
                f"{sorted(self.flatten_layers - self.prompt_layers)} 不在提示侧层"
                f"（--prompt-layers {self.prompt_spec}）里；改用 "
                f"--plan-source own_layer，或把 --prompt-layers 放宽")

    def _resolve_hist_layers(self) -> set[int]:
        """逐步诊断记哪些层（只对 plan_source=own_layer 有意义）。

        `all` 全记（结果会很大）、`gen` 只记生成侧算子的层、`sample`（默认）在
        它们之外再均匀补 4 个只做提示侧的层，用来看 c 随深度怎么变；也可以直接
        给层号（`_resolve_layers` 的语法）。"""
        spec = getattr(self, "ca_hist_spec", "sample")
        if spec == "all":
            return set(self.apply_layers)
        if spec == "gen":
            return set(self.flatten_layers)
        if spec == "sample":
            others = sorted(self.prompt_layers - self.flatten_layers)
            step = max(1, len(others) // 4)
            return set(self.flatten_layers) | set(others[::step][:4])
        return set(_resolve_layers(spec, self.n_layers, self.rng)) \
            & set(self.apply_layers)

    def _resolve_timing(self) -> bool:
        """same_step 由 plan_source 决定，而不再由层集合决定。"""
        # own_layer（默认）：每个施加层用**自己这一层、这一步**的行解自己的系数，
        # 计划与施加同层同步 —— 乘子零滞后。门仍读 32 层平均（只有跑完最后一层
        # 才完整），所以 same_step=True 让测量在最后一层的前向里发生。
        # layer_avg：乘子由上一步的层平均行算出，滞后一步。
        if self.plan_source == "own_layer" and self._same_step_override is False:
            raise ValueError("plan_source=own_layer 本身就是零滞后，"
                             "不能同时要求 same_step=False")
        return self.plan_source == "own_layer"

    def configure_arm(self, *, mode: str | None = None,
                      layers: str | None = None,
                      plan_source: str | None = None,
                      flatten_layers: str | None = None,
                      prompt_layers: str | None = None,
                      gen_op: str | None = None,
                      prompt_cap: float | None = None,
                      a_min: float | None = None,
                      b_max: float | None = None,
                      top_m: int | None = None,
                      topm_rank: str | None = None,
                      flatten_strength: float | None = None,
                      ref_normalize: str | None = None,
                      prompt_direction: str | None = None,
                      prompt_target_offset: str | None = None,
                      reference: str | None = None,
                      off_rule: str | None = None,
                      dyn_select: str | None = None,
                      burst_steps: int | None = None,
                      suppressor: str | None = None,
                      au_class: str | None = None,
                      au_top_k: int | None = None,
                      au_alpha: float | None = None,
                      au_prefill: bool | None = None) -> None:
        """按臂切换配置，复用同一个已加载的模型。

        评测脚本一次加载、多臂串跑，臂之间差 mode、施加层与计划来源，所以校验
        逻辑集中在这里，不在调用方散落。"""
        if mode is not None:
            assert mode in MODES, f"mode must be one of {MODES}"
            self.mode = mode
        if suppressor is not None:
            assert suppressor in SUPPRESSORS, \
                f"suppressor must be one of {SUPPRESSORS}"
            self.suppressor = suppressor
        else:
            # 臂没写就回到主方案的算子，而不是沿用上一条臂（否则这条臂测的
            # 其实是另一个干预方式）。
            self.suppressor = "row_reshape"
        if reference is not None and str(reference) != str(self.reference_path):
            # 按臂换参考曲线（例如同一套算子、比较不同阶数的拟合）。曲线是个小
            # JSON，重载的代价可以忽略，模型不必重新加载。
            self.reference_path = str(reference)
            self.reference = AttentionReference.load(reference)
            self.ref_metric = self.reference.metric
            print(f"[supp] 参考曲线 → {Path(self.reference_path).name}")
        if self.mode == "classifier" and self.reference is None \
                and self.suppressor == "row_reshape":
            raise ValueError("mode=classifier 需要构造时传入 reference")
        if off_rule is not None:
            assert off_rule in OFF_RULES, f"off_rule must be one of {OFF_RULES}"
            self.off_rule = off_rule
        if burst_steps is not None:
            self.burst_steps = int(burst_steps)
        if self.off_rule == "latch":
            self.burst_steps = 0            # 见 __init__ 同名分支
        if gen_op is not None:
            assert gen_op in RR.GEN_OPS, f"gen_op must be one of {RR.GEN_OPS}"
            self.gen_op = gen_op
        if topm_rank is not None:
            assert topm_rank in RR.TOPM_RANKS, \
                f"topm_rank must be one of {RR.TOPM_RANKS}"
            self.topm_rank = topm_rank
        if prompt_target_offset is not None:
            assert prompt_target_offset in PROMPT_TARGET_OFFSETS, \
                f"prompt_target_offset must be one of {PROMPT_TARGET_OFFSETS}"
            self.prompt_target_offset = prompt_target_offset
        for name, val in (("top_m", top_m),
                          ("flatten_strength", flatten_strength),
                          ("ref_normalize", ref_normalize),
                          ("prompt_direction", prompt_direction),
                          # 对齐算子要求的系数常比削峰算子极端得多（循环词两侧
                          # 占比能差上百倍），所以夹子也按臂配
                          ("a_min", a_min), ("b_max", b_max),
                          ("prompt_cap", prompt_cap)):
            if val is not None:
                setattr(self, name, val)
        if dyn_select is not None:
            self.dyn_spec = dyn_select
        if flatten_layers is not None:
            self.flatten_spec = flatten_layers
        if prompt_layers is not None:
            self.prompt_spec = prompt_layers
        if plan_source is not None:
            assert plan_source in PLAN_SOURCES, \
                f"plan_source must be one of {PLAN_SOURCES}"
            self.plan_source = plan_source
        else:
            # 臂没写就回到默认的逐层自解自施，而不是沿用上一臂（否则这条臂测的
            # 其实是另一个时序）。
            self.plan_source = "own_layer"
        if layers is not None:
            self.layers_spec = layers
            self.apply_layers = set(
                _resolve_layers(layers, self.n_layers, self.rng))
        # 生成侧算子的层永远是施加层的子集（不在施加层里的乘子不会被用到）；
        # 提示侧层同理，且可以写成 `gen` —— 与生成侧算子同层，所以要在它之后解。
        self.flatten_layers = set(_resolve_layers(
            self.flatten_spec, self.n_layers, self.rng)) & self.apply_layers
        self.prompt_layers = self._resolve_prompt_layers()
        self.ca_hist_layers = self._resolve_hist_layers()
        self.same_step = self._resolve_timing()
        self._resolve_dyn()
        self._check_layer_sets()
        # AU 钩子按臂重建：类别 / k / α 都可以逐臂扫，而模型只加载一次。
        for name, val in (("au_class", au_class), ("au_top_k", au_top_k),
                          ("au_alpha", au_alpha), ("au_prefill", au_prefill)):
            if val is not None:
                setattr(self, name, val)
        self._install_au()

    # ------------------------------------------------------------- run state
    def _reset_run_state(self) -> None:
        self._tokens: list[str] = []
        self._prompt_len = 0
        self._step = 0
        self._active = False
        self._n_pos = 0
        self._active_steps = 0
        self._first_active_step: int | None = None
        self._dyn_pending = bool(getattr(self, "dyn_rule", None))
        self._dyn_hist: dict[int, list[float]] = {
            l: [] for l in getattr(self, "dyn_cands", [])}
        self._dyn_selected: list[int] | None = None
        self._dyn_step: int | None = None
        self._dyn_means: dict[str, float] = {}
        self._dyn_base: dict[int, float] = {}
        self._dyn_hits: dict[int, int] = {}
        self._dyn_evals: dict[int, int] = {}
        self._dyn_hit_steps: set[int] = set()
        if self._dyn_pending and not self.dyn_perstep:
            # 每条样例都从「还没选层」开始，否则第二条会沿用上一条的结果
            self.flatten_layers = set()
            self.prompt_layers = self._resolve_prompt_layers()
            self.ca_hist_layers = self._resolve_hist_layers()
        self._cons_hist: list[float] = []
        self._trace: list[dict[str, Any]] = []
        self._trace_every = 0
        # per-gen-position detection features (index j == gen token j)
        self._cons_arr: list[float] = []
        self._pmf_arr: list[float] = []
        self._probes: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []
        self._spans: list[list[int]] = []
        self._span_start: int | None = None
        self._gate_open: dict[str, Any] | None = None
        self._burst_left = 0
        self._bursts = 0
        self._query_pos = -1
        self._supp_step = -1        # 本步门决定的快照步号（每步只算一次）
        self._supp_flag = False
        self._supp_counted = -1     # 本步是否已计入突发预算与 active_steps
        # 曲线目标的平移量：门开之后第一次真正抑制时锁定，此后整段不变。
        # 逐层路径下每个施加层各一份（各层的实测 pmf 不同），层平均路径只有
        # 一份（键 "avg"）。门重开、解除、类别改判时清空。见 `_offset_target`。
        self._pmf_offset: dict[Any, float] = {}
        # 逐层路径：类别按解码步快照（门可能在本步的最后一层改判）
        self._ca_step, self._ca_class = -1, None
        self._think_closed = False
        self._think_closed_step: int | None = None
        # --- 每步状态 ---------------------------------------------------------
        self._attack_class: str | None = None   # 分类器最近一次判出的攻击类别
        self._last_row: np.ndarray | None = None  # 上一步测到的层平均行
        self._ca_hist: list[dict[str, Any]] = []
        self._ca_status: dict[str, int] = {}
        # token 串 → 稠密 id 的增量表：生成侧按串聚合走 bincount 而不是 Python
        # 字典循环，否则 16k 长度下每步的解释器开销会超过前向本身
        self._vocab: dict[str, int] = {}
        self._id_names: list[str] = []
        self._ids_buf = np.empty(4096, dtype=np.int64)
        self._ids_len = 0

    # ------------------------------------------------------------------ patch
    def _install_patch(self) -> None:
        """Patch the model's own module-level `eager_attention_forward`.

        Every HF attention class routes into this primitive after Q/K/V
        projection, GQA, QK-norm and RoPE have already run, so patching here is
        architecture-agnostic (same injection point as
        extract_layer_avg_attention_v2 / layer_ls_closest)."""
        torch = self.torch
        # 取模型类自己所在的模块，而不是按 config.model_type 拼路径：多模态检查点
        # 经 AutoModelForCausalLM 加载后拿到的是文本子配置（Qwen3.6 的
        # model_type 是 `qwen3_5_text`），并不存在同名模块。
        mod = importlib.import_module(type(self.model).__module__)
        self._mod, self._orig_fn = mod, mod.eager_attention_forward
        repeat_kv = mod.repeat_kv
        st = self._state

        def patched(module, query, key, value, attention_mask, scaling,
                    dropout=0.0, **kwargs):
            n_rep = getattr(module, "num_key_value_groups", 1) or 1
            key_states = repeat_kv(key, n_rep)
            value_states = repeat_kv(value, n_rep)
            attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = torch.softmax(attn_weights, dim=-1,
                                         dtype=torch.float32).to(query.dtype)

            if query.shape[2] == 1 and st.get("recording"):
                li = getattr(module, "layer_idx", -1)
                # head mean of THIS layer's single query row, pre-suppression
                row = attn_weights[0, :, 0, :].float().mean(dim=0)
                if st["row_sum"] is None:
                    st["row_sum"] = row.clone()
                else:
                    st["row_sum"] += row
                st["n_seen"] += 1

                if self._dyn_pending and li in self._dyn_hist \
                        and self._step % self.dyn_stride == 0:
                    self._dyn_measure(li, row)

                if self.same_step and li == self.last_layer:
                    # 本步所有层都已进 row_sum → 当前查询的层平均是精确的。
                    # 门控读层平均（分类器就是在层平均特征上训练的），但层平均
                    # 要到这里才完整，所以本步的门状态只对下一步生效。乘子不走
                    # 这条路，见下面两个分支。
                    self._measure_and_gate(st["row_sum"] / st["n_seen"])

                if self.plan_source == "own_layer":
                    # 逐层自解自施：本层这一行就在手里（softmax 之后、乘子之前），
                    # 提示侧常数与生成侧算子都算在它上面，零滞后。两半都不覆盖的
                    # 层这一步什么都不做，行连搬到 CPU 都省了。
                    # li < 最后一层时，本层输出的改变会经残差流改掉第 li+1.. 层
                    # 为当前位置写入的 K/V，之后每一步都读到被改过的键值——干预
                    # 因此具备跨步记忆。
                    mult = (self._class_aware_layer_multiplier(li, row)
                            if (li in self.prompt_layers
                                or li in self.flatten_layers) else None)
                else:
                    # 层平均路径（滞后一步）：提示侧常数施加在 prompt_layers、
                    # 生成侧算子只在 flatten_layers（⊆ prompt_layers，见
                    # `_check_layer_sets`）上，所以这一步有两个乘子，按层取用
                    # （两者的提示侧部分相同）；两边都不在的层不动。
                    mult = (st.get("mult_flat") if li in self.flatten_layers
                            else (st.get("mult")
                                  if li in self.prompt_layers else None))

                if mult is not None and li in self.apply_layers:
                    T = attn_weights.shape[-1]
                    m = mult
                    if m.shape[0] < T:               # plan older than last tokens
                        m = torch.cat([m, torch.ones(T - m.shape[0],
                                                     device=m.device, dtype=m.dtype)])
                    elif m.shape[0] > T:
                        m = m[:T]
                    # How much attention the intervention actually has hold of:
                    # the pre-suppression mass sitting on the targeted positions,
                    # in THIS layer's own row vs in the layer average the plan
                    # was computed from. If the applied layer puts little mass
                    # there, no k can make the intervention bite — that is the
                    # cost of choosing where to apply by timing rather than by
                    # fidelity to the layer average.
                    hit = (m < 1.0)
                    # 提示侧的常数可能小于 1（循环字符要压提示侧），那时整个提示
                    # 侧都会落进 hit，这个诊断就变成了「提示侧质量」而不是「被削
                    # 的峰」。只留生成侧。
                    hit[: self._prompt_len] = False
                    st["mass_layer"] += float(row[hit].sum())
                    if self.plan_source == "layer_avg":
                        st["mass_avg"] += float(
                            (st["row_sum"][: T][hit] / st["n_seen"]).sum())
                    w = attn_weights.float() * m
                    w = w / w.sum(dim=-1, keepdim=True)
                    attn_weights = w.to(query.dtype)
                    st["applied_steps"] += 1

            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            return attn_output, attn_weights

        mod.eager_attention_forward = patched

    def close(self) -> None:
        self._mod.eager_attention_forward = self._orig_fn

    # ------------------------------------------------------- measure and gate
    def _measure(self, row_np: np.ndarray, query_pos: int) -> float:
        """一次 top-θ 遍历，给出这一行的注意力一致性，并把**两个检测特征**都记进
        这个生成位置。

        `prompt_mean_fraction` 与离线 `_compute_per_token_ratios` 同口径：提示侧
        每 token 平均注意力 ÷ 全行每 token 平均注意力。

        乘子算在同一行上，但要等门（分类器）更新完类别才建，所以这里只把行存
        下来，见 `_class_aware_multipliers`。"""
        q, P = query_pos, self._prompt_len
        self._query_pos = q
        row = row_np[: q + 1]
        cons = consistency(row, P, self._tokens, threshold=self.threshold)
        total = float(row.sum())
        pmf = (((float(row[:P].sum()) / P) / (total / (q + 1)))
               if P > 0 and total > 0 else float("nan"))
        self._cons_arr.append(cons)
        self._pmf_arr.append(pmf)
        self._last_row = row
        if np.isfinite(cons):
            self._cons_hist.append(round(cons, 4))
        return cons

    def _classifier_probe(self, query_pos: int) -> None:
        """Run the deployed detector and move the gate.

        The probe grid depends on the gate, exactly as the pipeline specifies:

          - **gate closed** — only the deployed checkpoints
            (`trigger_checkpoints`). Before an attack has been detected there is
            nothing to re-check; the detector runs on its own protocol grid.
          - **gate open** — every `probe_every` (32) tokens, the re-check that
            keeps suppression alive or releases it.

        Note this restricts *detector calls*, not measurement: `cons_ind` and
        `prompt_mean_fraction` are still accumulated every step, because the
        classifier's features are running statistics over ALL gen positions from
        index 0 — that is the definition it was trained on, so the arrays have to
        be complete by the time the first checkpoint is reached."""
        L = query_pos + 1                       # total length incl. this token
        if self._active:
            if self.off_rule == "latch":
                # 「一旦开始抑制，不再进行检测」——门锁死，检测器不再被调用，
                # 也就没有复检、没有解除、没有类别改判。逐步指标
                # （cons_hist/pmf_hist）仍照常累积：它们是事后复盘的原始材料，
                # 不参与任何在线决策，所以留着不违反这条规则。
                return
            if L % self.probe_every:
                return
        elif L not in self.trigger_checkpoints:
            return
        i = query_pos - self._prompt_len        # gen index of the current row
        if i < MIN_GEN_FOR_SLOPE - 1:
            return
        feat = features_at(np.asarray(self._cons_arr, float),
                           np.asarray(self._pmf_arr, float),
                           self._prompt_len, i,
                           log_L=self.det.log_L, use_L=self.det.use_L)
        if feat is None:
            return
        pred, prob, probs = self.det.predict_one(feat)
        # Diagnostic only — never fed to the classifier. The feature the gate
        # reads is a RUNNING mean over [0, i], so it carries every looping token
        # emitted before the trigger and moves slowly once i is large. The mean
        # over the most recent `recent_window` positions shows whether the
        # intervention changed local behaviour even while the running mean is
        # still pinned by the pre-trigger history.
        recent = [c for c in self._cons_arr[-self.recent_window:] if np.isfinite(c)]
        rec = {"step": self._step, "L": L, "gen_pos": i,
               "cons_ind_mean": round(float(feat[0]), 4),
               "cons_recent": round(float(np.mean(recent)), 4) if recent else None,
               "pmf_slope": round(float(feat[1]), 6),
               "pred": pred, "prob": round(prob, 4),
               "attack_prob": round(sum(v for c, v in probs.items()
                                        if c in ATTACK), 4),
               "active": self._active, "action": None}
        self._probes.append(rec)
        if self.mode != "classifier":
            return
        if self.stop_after_think and self._think_closed:
            # 思考段结束后门已关死：探针继续记录（事后复盘要看检测怎么判），
            # 但不再驱动任何状态迁移。
            return

        attack_call = pred in ATTACK and prob >= self.tau_prob
        if not self._active:
            # Reached only on the deployed grid — see the probe-grid rule above.
            if attack_call:
                self._active = True
                # 新的一段抑制 → 平移量在它的第一步重新锁定
                self._pmf_offset.clear()
                # 动态选层就发生在这一刻：门刚开、还没压过任何一步，用的全是
                # 无干预轨迹上测到的逐层一致性。
                self._dyn_decide()
                self._attack_class = pred           # 按类别选参考曲线
                self._span_start = self._step
                self._burst_left = self.burst_steps
                self._bursts += 1
                rec["action"] = "on"
                if self._gate_open is None:
                    self._gate_open = dict(rec)
                if self._first_active_step is None:
                    self._first_active_step = self._step
        else:
            release = (pred not in ATTACK and
                       (self.off_rule == "argmax" or prob >= self.tau_prob))
            if release:
                self._active = False
                self._attack_class = None
                self._burst_left = 0
                self._pmf_offset.clear()
                self._spans.append([self._span_start, self._step])
                self._span_start = None
                rec["action"] = "off"
            else:
                # still a loop at this re-check → suppress the next burst
                if pred in ATTACK:
                    # 复检可以改判类别（思考循环 → 字符循环），参考曲线随之切换，
                    # 平移量是相对旧曲线量出来的，也必须重新锁定
                    if pred != self._attack_class:
                        self._pmf_offset.clear()
                    self._attack_class = pred
                self._burst_left = self.burst_steps
                self._bursts += 1
                rec["action"] = "burst"
        if rec["action"]:
            self._events.append({k: rec[k] for k in
                                 ("step", "L", "pred", "prob", "attack_prob",
                                  "action")})

    def _note_think_closed(self) -> None:
        """刚吐出 </think>：记下位置，并按 stop_after_think 收掉正在开的抑制区间。

        检测器不受影响——它照常每步累积特征、照常在检查点/复检时被调用，只是
        它的判定不再驱动任何干预。"""
        if self._think_closed:
            return
        self._think_closed = True
        self._think_closed_step = self._step
        if not self.stop_after_think:
            return
        # 门就地关死：否则后续复检还会照旧记 on/burst/off，抑制区间与门事件会
        # 显示「一直在压」，而 active_steps 说的是 0.26%——两个字段互相打架。
        if self._span_start is not None:
            self._spans.append([self._span_start, self._step])
        self._span_start = None
        self._active = False
        self._attack_class = None
        self._burst_left = 0
        self._pmf_offset.clear()
        self._events.append({"step": self._step, "L": len(self._tokens),
                             "pred": None, "prob": None, "attack_prob": None,
                             "action": "off_think"})

    def _gate(self, query_pos: int) -> None:
        """门只有一种：接了检测器就让它判，判出循环类就开门。"""
        if self.det is not None:
            self._classifier_probe(query_pos)

    def _apply_this_step(self) -> bool:
        """Whether the intervention runs on THIS decode step.

        Classifier modes suppress in bursts: a re-check that still says "loop"
        buys `burst_steps` steps of suppression, then decoding runs free until
        the next re-check. `burst_steps=0` means the whole interval (the gate
        is simply on). The other modes have no burst notion — the gate flag is
        the whole story."""
        on = self._gate_on_now()
        if on:
            self._consume_burst()
        return on

    def _gate_on_now(self) -> bool:
        """门是否允许本步抑制——只读，不消耗突发预算。"""
        if self.stop_after_think and self._think_closed:
            return False            # </think> 之后不再干预（见 __init__ 同名字段）
        if self.mode == "always":
            # 常开：没有在线决策，从第 0 步压到结束（AUSteer 基线的定义）。
            # 注意这里**不**去动 `_active`：让它保持 False，检测器就仍按「门未开」
            # 的协议在五个部署检查点上跑，给这条臂留下与 A0_off 同口径的诊断。
            return True
        if self.mode != "classifier":
            return self._active
        if not self._active:
            return False
        if not self.burst_steps:
            return True
        return self._burst_left > 0

    #: 本类的抑制动作是否发生在注意力行上。子类把动作挂在 `_filter_logits`
    #: （输出侧）时置 False，`_check_layer_sets` 便不再要求施加层非空——施加面
    #: 为空是那种臂的**定义**，与 AUSteer 同理。
    attention_action = True

    def _filter_logits(self, logits, seq, step: int):
        """采样前改写 logits 的挂载点；本类是**空操作**。

        主线的抑制动作全在注意力行上（`row_reshape`）或激活上（`austeer`），都在
        模型内部完成，不碰 logits。加这个点是为了让 `improve/` 里那些**输出侧**的
        改进方案（门控 + 硬约束等）能复用整套门控与记录逻辑，而不必复制 `run()`。

        `seq` 是到本步为止的完整 token id 列表（含提示），`step` 是解码步号。
        子类应返回改写后的 logits（原地改也可以）。
        """
        return logits

    def _consume_burst(self) -> None:
        """消耗一步突发预算；调用方负责保证每个解码步只消耗一次。"""
        if self.mode == "classifier" and self.burst_steps \
                and self._burst_left > 0:
            self._burst_left -= 1

    def _suppress_this_step(self) -> bool:
        """本步的门决定，每个解码步只算一次并快照。

        逐层路径下多个施加层会先后问同一个问题，而消耗突发预算会改变
        `_burst_left`；若每层各自现算，就会出现「第 30 层压了、第 31 层看到
        预算已空」的层间不一致。"""
        if self._supp_step != self._step:
            self._supp_step = self._step
            self._supp_flag = self._gate_on_now()
        return self._supp_flag

    # --------------------------------------------------- 类别自适应乘子
    def _push_token_id(self, tok: str) -> None:
        """把一个 token 串登记进增量 id 表（容量翻倍，视图零拷贝）。"""
        i = self._vocab.get(tok)
        if i is None:
            i = self._vocab[tok] = len(self._id_names)
            self._id_names.append(tok)
        if self._ids_len == self._ids_buf.size:
            self._ids_buf = np.concatenate(
                [self._ids_buf, np.empty(self._ids_buf.size, dtype=np.int64)])
        self._ids_buf[self._ids_len] = i
        self._ids_len += 1

    def _prompt_target(self, cls: str, gen_pos: int, P: int) -> float:
        """提示侧的 pmf 目标 —— **参考曲线在该生成位置上的原始取值**。

        它要求的提示侧质量占比 ≥ 1 时，求解器改瞄封顶占比 `SHARE_CEILING`
        （`row_reshape` §1）——那一档不查曲线，所以这里只取这一个值。
        `--prompt-target-offset` 的平移发生在这之后，见 `_offset_target`。"""
        return self.reference.target_ratio(
            cls, gen_pos, P, normalize=self.ref_normalize,
            quantile=self.ref_quantile)

    def _offset_target(self, key, raw: float, row: np.ndarray,
                       P: int) -> tuple[float, float]:
        """把曲线目标平移到「开始抑制那一刻的实测水平」，返回 (目标, 平移量)。

        **要解决的问题**：`solve_prompt_scale` 是逐步**对齐水平**——每一步都把当前
        行的 pmf 送到曲线在该位置的取值上。但开始抑制的那一刻实测 pmf 远低于曲线
        （循环思考；循环字符则远高于），于是第一步就是一个跳变，此后实际 pmf 轨迹
        的斜率必然高于（低于）参考曲线的斜率，抬升量整体偏大。

        **平移**：用触发点上的差值 Δ 把整条目标曲线搬到实测那一侧

            trigger      目标(t) = 曲线(t) − Δ,   Δ = 曲线(t0) − 实测(t0)
            trigger_log  目标(t) = 曲线(t) / Δ,   Δ = 曲线(t0) / 实测(t0)

        两式在 t0 上都给出 目标 = 实测 ⇒ c = 1（起点不跳），之后目标随曲线走而
        平移量不变 ⇒ **实际 pmf 轨迹与参考曲线平行**（`trigger` 在线性域平行，
        `trigger_log` 在对数域平行——后者与曲线自身的拟合域、以及残差分位数平移
        `attention_reference._resid_offset` 同一口径，且目标恒为正）。

        Δ 必须量在**没被压过的**行上，所以只在门开后第一次真正施加的那一步锁定
        （`row` 取于本层乘子之前；施加层不止一层时更深的层仍会看到浅层这一步的
        改动，这与水平对齐口径下的既有情形相同），此后整段沿用；门重开、解除或
        复检改判类别时清空（`_classifier_probe`），因为那是新的一段、新的曲线。
        逐层路径下 `key` 是层号——各层的实测 pmf 不同，共用一个 Δ 会偏。

        **副作用，必须知道**：平移后的目标 ≈ 实测水平，它要求的提示侧质量占比
        ≈ 触发点的实测占比（< 1），于是 `solve_prompt_scale` 的**封顶档几乎不再
        被触发**（`row_reshape` §1）。长提示上原本由 `SHARE_CEILING`=0.95 决定的
        强干预因此退回到由曲线斜率决定的弱干预——`ca_hist` 里的 `target_kind`
        会从 `cap` 翻成 `pmf`，这是判断这条路走不走得通的第一个观察点。
        """
        if self.prompt_target_offset == "none":
            return raw, 0.0
        off = self._pmf_offset.get(key)
        if off is None:
            now = RR.prompt_mean_fraction(row, P)
            if not (np.isfinite(now) and now > 0
                    and np.isfinite(raw) and raw > 0):
                return raw, 0.0          # 行退化：这一步不锁定，下一步再试
            off = (raw - now if self.prompt_target_offset == "trigger"
                   else raw / now)
            self._pmf_offset[key] = float(off)
        return ((raw - off) if self.prompt_target_offset == "trigger"
                else (raw / off)), float(off)

    def _ca_hist_record(self, ps, fp, ap, *, T: int, gen_pos: int, cls,
                        n_pos: int, layer: int | None = None,
                        raw_target: float = float("nan"),
                        offset: float = 0.0) -> dict:
        """一条逐步诊断。两条路径（层平均 / 逐层）共用，逐层时多一个 layer 字段。

        `ps=None` 表示这一层不在 `prompt_layers` 里（提示侧那一半整个跳过）：
        所有提示侧字段记 None、`prompt_status="off"`，事后统计一律跳过它们。"""
        gen = fp if fp is not None else ap
        rd = (lambda v: round(v, 4)) if ps is not None else (lambda v: None)
        rec = {
            "step": self._step, "L": T, "gen_pos": gen_pos, "class": cls,
            # ratio_* 是主档口径（prompt_mean_fraction）的当前值/目标/落点。
            # target_kind 说明这一步实际瞄的是主档 pmf 还是封顶占比，
            # target_share 是换算成「提示侧质量占比」之后实际瞄的那个数。
            "ratio": rd(ps.ratio_now if ps else 0.0),
            # ratio_target 是**实际瞄的**那个目标（`--prompt-target-offset` 平移
            # 之后）；ratio_target_raw 是参考曲线本身给的，pmf_offset 是两者之差
            # （trigger_log 下是两者之比）。不平移时 raw 与 target 相等。
            "ratio_target": rd(ps.ratio_target if ps else 0.0),
            "ratio_target_raw": (rd(raw_target) if np.isfinite(raw_target)
                                 else None),
            "pmf_offset": (round(float(offset), 4)
                           if self.prompt_target_offset != "none" else None),
            "ratio_after": rd(ps.ratio_after if ps else 0.0),
            "c": rd(ps.c if ps else 0.0),
            "prompt_share": rd(ps.prompt_share if ps else 0.0),
            "prompt_share_after": rd(ps.prompt_share_after if ps else 0.0),
            "sink_share": rd(ps.sink_share if ps else 0.0),
            "prompt_status": (ps.status if ps is not None else "off"),
            "target_kind": (ps.target_kind if ps is not None else None),
            "target_share": rd(ps.target_share if ps else 0.0),
            "n_pos": n_pos,
            "gen_op": self.gen_op,
            # 下面几个键在两种生成侧算子里都有意义，含义随算子变（见 gen_op）：
            #   a          flatten = 唯一的削峰系数；prompt_align = 最强的那档
            #   b          回填系数，两者相同含义
            #   top_mass   被本算子直接改动的那部分占生成侧的比例（施加前）
            #   gen_status 该算子这一步的状态（ok / 被夹住 / 不动）
            "a": None, "b": None, "top_mass": None,
            "gen_status": (gen.status if gen is not None else "off"),
            "top_tokens": (gen.tokens[:6] if gen is not None else []),
        }
        if layer is not None:
            rec["layer"] = layer
        if fp is not None:
            rec.update(a=round(fp.a, 4), b=round(fp.b, 4),
                       top_mass=round(fp.top_mass, 4),
                       uniform_mass=round(fp.uniform_mass, 4))
        elif ap is not None:
            a_vals = list(ap.a_by_id.values())
            rec.update(a=(round(min(a_vals), 4) if a_vals else None),
                       b=round(ap.b, 4), top_mass=round(ap.sel_mass, 4),
                       # 对齐目标：被压的这些串在**提示侧**的占比之和
                       target_mass=round(ap.target_mass, 4),
                       # 生成侧三段（占生成侧）：被压 / 不动 / 回填
                       unchanged_mass=round(ap.unchanged_mass, 4),
                       rest_mass=round(ap.rest_mass, 4),
                       keep_tokens=ap.keep_tokens[:6],
                       below_tokens=ap.below_tokens[:6],
                       n_top=ap.n_top, n_hit=len(ap.a_by_id),
                       n_unchanged_pos=ap.n_unchanged_pos,
                       a_by_token=ap.a_by_token, gen_share=ap.gen_share,
                       prompt_share_by_token=ap.prompt_share)
        return rec

    def _class_aware_multipliers(self):
        """本步的两个乘子：(所有层用的提示侧乘子, 生成侧算子层用的完整乘子)。

        方案（v5）分两半，都算在**上一步测到的层平均行**上：

          1. 提示侧对齐正常思考。当前行的两侧每 token 平均强度之比 ρ，与该模型
             正常思考轨迹在**同一生成位置**上的参考值之比，就是要乘的常数
             c = ρ_目标 / ρ_当前（`row_reshape.solve_prompt_scale`，闭式，无近似）。
             判为循环思考 → 对齐有效反思（c > 1，抬）；判为循环字符 → 对齐简洁
             推理（c < 1，压）。**提示很长时 ρ_目标 会要求提示侧质量占比 ≥ 1
             （无解）**，那一步改瞄封顶占比 `row_reshape.SHARE_CEILING`（0.95），
             强度由这个常数而不是参考曲线决定（`row_reshape` §1）。c 施加在
             `prompt_layers` 的所有头上（默认全部施加层；`--prompt-layers gen`
             则与生成侧算子同层）。
          2. 生成侧算子，二选一，只施加在 `flatten_layers` 上：

             `gen_op="flatten"`（削向均匀，v5 初版）：生成侧按 token 串聚合后
             质量最高的 top-m 个串，其全部出现位置乘 a < 1，其余生成位置乘
             b > 1（`row_reshape.solve_gen_flatten`）。

             `gen_op="prompt_align"`（对齐提示侧占比）：只动「top-m 词串」与
             「提示里出现过的词串」的**交集**，每个串各乘一个系数，使它在生成侧
             的质量占比等于它在提示侧的质量占比；交集之外的生成位置（含 top-m
             里提示中没出现过的串）共用一个回填系数
             （`row_reshape.solve_gen_prompt_align`）。

             两者都保证**生成侧总质量精确不变**，所以都不会破坏第 1 步解出的
             提示侧占比。

        滞后一步是层平均的固有代价（要跑完最后一层才完整），但滞后的只是**系数**
        与**词串集合**：位置集合按当前 token 列表重建，所以刚吐出的重复 token 只要
        词串已在集合里就会被覆盖。乘子在本步前向开始前就已知，因此可以零滞后地
        施加到任意层——包括低层，从而改写下游各层为当前位置写入的 KV（跨步记忆）。
        """
        torch = self.torch
        row = self._last_row
        if row is None or self._attack_class not in REFERENCE_OF:
            return None, None
        if not self._apply_this_step():
            return None, None
        P, T = self._prompt_len, len(self._tokens)
        gen_pos = max(0, T - 1 - P)
        raw_target = self._prompt_target(self._attack_class, gen_pos, P)
        # 层平均路径只有一条行，平移量也就只有一份（键 "avg"）
        target, offset = self._offset_target("avg", raw_target, row, P)
        ps = RR.solve_prompt_scale(
            row, P, target,
            sink_positions=self.sink_positions, cap=self.prompt_cap,
            direction=(EXPECTED_DIRECTION[self._attack_class]
                       if self.prompt_direction == "expected"
                       else self.prompt_direction))
        ids = self._ids_buf[: self._ids_len]
        fp = ap = None
        if self.flatten_layers:
            if self.gen_op == "prompt_align":
                # 对齐算子只压不抬、不带夹子：a_t 是精确对齐所需的值，b 由
                # 守恒唯一解出（a_min/b_max 只属于削峰算子）。
                ap = RR.solve_gen_prompt_align(
                    row, P, token_ids=ids, id_names=self._id_names,
                    top_m=self.top_m, strength=self.flatten_strength,
                    rank_by=self.topm_rank)
            else:
                fp = RR.solve_gen_flatten(
                    row, P, token_ids=ids, id_names=self._id_names,
                    top_m=self.top_m, strength=self.flatten_strength,
                    a_min=self.a_min, b_max=self.b_max,
                    group_by=self.flatten_group_by, rank_by=self.topm_rank)
        gen = fp if fp is not None else ap
        gen_active = gen is not None and gen.active
        if not ps.active and not gen_active:
            return None, None

        # 乘子按**当前**长度重建：词串集合来自上一步测到的行，位置集合是现在的，
        # 所以这一步刚吐出的重复 token 只要词串已在集合里就会被覆盖。对齐算子
        # 直接按 id 数组落位（`build_row_multiplier(align=...)`），削峰算子的
        # 位置列表在这里重算。
        if fp is not None and fp.active and fp.top_ids:
            fp.positions = (np.nonzero(np.isin(ids[P:T],
                                               np.asarray(fp.top_ids)))[0]
                            + P).tolist()
        m_prompt = RR.build_row_multiplier(T, P, c=ps.c)
        m_flat = (RR.build_row_multiplier(
            T, P, c=ps.c, flatten=fp, align=ap,
            token_ids=ids) if gen_active else m_prompt)

        if fp is not None and fp.active:
            self._n_pos = len(fp.positions)
        elif ap is not None and ap.active:
            self._n_pos = int(np.isin(ids[P:T],
                                      np.asarray(list(ap.a_by_id))).sum())
        else:
            self._n_pos = 0
        self._active_steps += 1
        if self._first_active_step is None:
            self._first_active_step = self._step
        for st_key in (f"prompt:{ps.status}",
                       f"gen:{gen.status if gen is not None else 'off'}"):
            self._ca_status[st_key] = self._ca_status.get(st_key, 0) + 1
        if self._step % self.ca_hist_every == 0:
            self._ca_hist.append(self._ca_hist_record(
                ps, fp, ap, T=T, gen_pos=gen_pos, cls=self._attack_class,
                n_pos=self._n_pos, raw_target=raw_target, offset=offset))
        dev = self.model.device
        return (torch.from_numpy(m_prompt).to(dev),
                torch.from_numpy(m_flat).to(dev))

    def _class_aware_layer_multiplier(self, li: int, row_t):
        """plan_source=own_layer 下 v5 的乘子：**每层用自己的行、算自己的系数**。

        与层平均路径（`_class_aware_multipliers`）的唯一区别是行的来源，两半的
        算子完全相同：

          1. 提示侧常数 c —— **`prompt_layers` 里的层**都算，各层解各自的 c
             （`solve_prompt_scale` 作用在本层这一行上）。目标值仍由参考曲线在同一
             生成位置给出，与层无关，但当前值逐层不同，所以解出来的 c 逐层不同。
             `--prompt-layers gen` 时这批层与生成侧算子的层重合：两半作用在同一处，
             其余层原样不动（本方法对它们返回 None）。
          2. 生成侧 token 抑制 —— **只有 `flatten_layers`** 算，且**词串集合与系数
             也各层独立**（top-m 是按本层行的生成侧质量选的，交集与提示侧占比也
             都读本层这一行）。

        调用点在 patched attention 内部，`row_t` 是该层这一步的头平均行，取于
        softmax 之后、乘子之前 —— 所以计划与施加同层同步、**零滞后**，且施加在
        li < 最后一层时，本层输出的改变经残差流改写第 li+1.. 层为当前位置写入的
        K/V（跨步记忆）。

        **与分类器无关**：门读的是 32 层平均行（在最后一层才完整，见
        `_measure_and_gate`），本方法既不改它读的那一行（累加发生在乘子之前），
        也不参与特征累积。抑制的改动只通过「后续生成出什么 token」影响检测。
        """
        if not self._suppress_this_step():
            return None
        # 类别按解码步快照：层平均要跑完第 31 层才完整，门可能在本步中途改判，
        # 若不快照，同一步里最后一层会用与前面各层不同的参考曲线。
        if self._ca_step != self._step:
            self._ca_step = self._step
            self._ca_class = self._attack_class
        cls = self._ca_class
        if cls not in REFERENCE_OF:
            return None
        q = len(self._tokens) - 1
        if q < 0:
            return None
        row = row_t.cpu().numpy().astype(np.float64)[: q + 1]
        if self.dyn_perstep and not self._dyn_gate_layer(li, row):
            return None
        T, P = row.size, self._prompt_len
        gen_pos = max(0, T - 1 - P)
        raw_target = self._prompt_target(cls, gen_pos, P)
        # 提示侧只在 prompt_layers 上做：不在里面的层这一半整个跳过（c ≡ 1），
        # 若它也不是生成侧算子的层，本步就完全不碰它。目标平移量**按层锁定**
        # ——逐层路径下各层的实测 pmf 不同，借用别的层的差值会偏。
        ps, target, offset = None, raw_target, 0.0
        if li in self.prompt_layers:
            target, offset = self._offset_target(li, raw_target, row, P)
            ps = RR.solve_prompt_scale(
                row, P, target,
                sink_positions=self.sink_positions, cap=self.prompt_cap,
                direction=(EXPECTED_DIRECTION[cls]
                           if self.prompt_direction == "expected"
                           else self.prompt_direction))
        ids = self._ids_buf[: self._ids_len]
        fp = ap = None
        if li in self.flatten_layers:
            if self.gen_op == "prompt_align":
                ap = RR.solve_gen_prompt_align(
                    row, P, token_ids=ids, id_names=self._id_names,
                    top_m=self.top_m, strength=self.flatten_strength,
                    rank_by=self.topm_rank)
            else:
                fp = RR.solve_gen_flatten(
                    row, P, token_ids=ids, id_names=self._id_names,
                    top_m=self.top_m, strength=self.flatten_strength,
                    a_min=self.a_min, b_max=self.b_max,
                    group_by=self.flatten_group_by, rank_by=self.topm_rank)
        gen = fp if fp is not None else ap
        gen_active = gen is not None and gen.active
        if not (ps is not None and ps.active) and not gen_active:
            return None
        # 零滞后：行就是本步本层的行，乘子按同一长度直接构造，无需重建位置集合。
        mult = RR.build_row_multiplier(
            T, P, c=(ps.c if ps is not None else 1.0),
            flatten=fp if gen_active and fp is not None else None,
            align=ap if gen_active and ap is not None else None,
            token_ids=ids)
        n_pos = (len(fp.positions) if fp is not None and fp.active else
                 (int(np.isin(ids[P:T], np.asarray(list(ap.a_by_id))).sum())
                  if ap is not None and ap.active else 0))
        # 突发预算与 active_steps 按「解码步」计，多层施加只算一次。
        if self._supp_counted != self._step:
            self._supp_counted = self._step
            self._consume_burst()
            self._active_steps += 1
            self._n_pos = 0        # 本步重新统计（第一个被调用的层通常只做提示侧）
            if self._first_active_step is None:
                self._first_active_step = self._step
        # 逐步轨迹里的「被压位置数」取本步各层的最大值——只做提示侧的层是 0，
        # 直接用第一个被调用的层会把它记成 0。
        self._n_pos = max(self._n_pos, n_pos)
        for st_key in (f"prompt:{ps.status if ps is not None else 'off'}",
                       f"gen:{gen.status if gen is not None else 'off'}"):
            self._ca_status[st_key] = self._ca_status.get(st_key, 0) + 1
        if self._step % self.ca_hist_every == 0 and li in self.ca_hist_layers:
            self._ca_hist.append(self._ca_hist_record(
                ps, fp, ap, T=T, gen_pos=gen_pos, cls=cls, n_pos=n_pos,
                layer=li, raw_target=raw_target, offset=offset))
        return self.torch.from_numpy(mult).to(self.model.device)

    def _measure_and_gate(self, row_t) -> None:
        """用已经完整的层平均行做测量与门控，不产生乘子。

        plan_source=own_layer 时只调用这一半。分类器的特征定义在层平均上，而
        层平均要到最后一层才完整，所以本步的门状态只能对下一步生效——这是本
        方案里唯一保留的滞后。它无害：门状态只在部署检查点和每 32 token 的复
        检时才可能变化，滞后 1 步与「上一步的词集里没有这一步刚吐出的重复词」
        不是一个量级的问题。"""
        q = len(self._tokens) - 1
        if q < 0 or self._step % self.measure_every:
            return
        row = row_t.cpu().numpy().astype(np.float64)
        self._measure(row, q)
        self._gate(q)

    # ------------------------------------------------------------------- main
    def run(self, prompt: str, meta: dict[str, Any] | None = None,
            trace_every: int = 0, prefill: str = "") -> dict[str, Any]:
        torch, tok = self.torch, self.tokenizer
        st = self._state
        prompt_text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True)
        ids = tok(prompt_text, return_tensors="pt",
                  add_special_tokens=False).input_ids.to(self.model.device)
        # 预填充段（concat 子集种在 <think> 里的重复词）。生成轨迹时它接在模板
        # 之后、算作生成侧（prompt_len 止于模板末尾），检测特征也按这个口径算，
        # 所以这里不把它并进提示，而是在解码循环里逐 token 强制喂入：那些步照常
        # 记录注意力行、更新检测统计、过门控，只是不采样。整串一起分词再按提示
        # 长度切，与生成时对 context 整串分词一致；前缀对不上时退回单独分词。
        forced: list[int] = []
        if prefill:
            head = ids[0].tolist()
            joint = tok(prompt_text + prefill, add_special_tokens=False).input_ids
            forced = (joint[len(head):] if joint[:len(head)] == head
                      else tok(prefill, add_special_tokens=False).input_ids)

        self._reset_run_state()
        for h in self._au_hooks:
            h.reset_counter()
        # Re-seed per run so a sampled run is reproducible: the same (prompt,
        # config, seed) replays token for token even at temperature > 0.
        torch.manual_seed(self.seed)
        self._trace_every = trace_every
        self._prompt_len = int(ids.shape[1])
        flat = ids[0].tolist()
        self._tokens = [tok.decode([i], skip_special_tokens=False) for i in flat]
        tokens = self._tokens
        for t in tokens:
            self._push_token_id(t)
        st.update(recording=False, row_sum=None, n_seen=0, mult=None,
                  mult_flat=None, applied_steps=0, mass_layer=0.0,
                  mass_avg=0.0)
        t0 = time.time()

        past = None
        cur = ids
        stopped = False
        with torch.no_grad():
            for step in range(self.max_new_tokens):
                self._step = step
                st["recording"] = cur.shape[1] == 1     # skip the prefill
                st["row_sum"], st["n_seen"] = None, 0
                attn_mask = torch.ones((1, len(flat)), dtype=torch.long,
                                       device=self.model.device)
                out = self.model(input_ids=cur, attention_mask=attn_mask,
                                 past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[0, -1].float()
                logits = self._filter_logits(logits, flat, step)
                if step < len(forced):
                    nxt = forced[step]
                elif self.temperature and self.temperature > 0:
                    p = torch.softmax(logits / self.temperature, dim=-1)
                    nxt = int(torch.multinomial(p, 1).item())
                else:
                    nxt = int(torch.argmax(logits).item())

                # ---- lagged regime: measure the row just recorded (query =
                # the token fed this step), applied from the next step on.
                if not self.same_step and st["row_sum"] is not None \
                        and st["n_seen"] == self.n_layers \
                        and step % self.measure_every == 0:
                    row = (st["row_sum"] / self.n_layers).cpu().numpy().astype(np.float64)
                    q = len(tokens) - 1
                    self._measure(row, q)
                    self._gate(q)

                flat.append(nxt)
                tokens.append(tok.decode([nxt], skip_special_tokens=False))
                if not self._think_closed and THINK_END in "".join(tokens[-4:]):
                    self._note_think_closed()
                self._push_token_id(tokens[-1])
                if nxt in self.eos_ids:
                    stopped = True
                    break
                if self.plan_source == "layer_avg":
                    st["mult"], st["mult_flat"] = self._class_aware_multipliers()
                cur = torch.tensor([[nxt]], device=self.model.device)

        st["mult"] = st["mult_flat"] = None
        st["recording"] = False
        if self._span_start is not None:              # still on when we stopped
            self._spans.append([self._span_start, self._step])
            self._span_start = None
        gen_ids = flat[self._prompt_len:]
        text = tok.decode(gen_ids, skip_special_tokens=True)
        thinking = text.split(THINK_CLOSE)[0]
        cons_hist = self._cons_hist
        res: dict[str, Any] = {
            **(meta or {}),
            "mode": self.mode, "layers": self.layers_spec,
            "plan_source": self.plan_source,
            "prompt_target_offset": self.prompt_target_offset,
            # 最后一段抑制锁定的曲线平移量（键 = 施加层号；layer_avg 路径为 "avg"）
            "pmf_offsets": {str(k): round(float(v), 4)
                            for k, v in self._pmf_offset.items()},
            "apply_layers": sorted(self.apply_layers),
            "suppressor": self.suppressor,
            # AUSteer 基线的可复现字段：计划文件、变体、类别、k、α、实际被引导的
            # 激活数（= 论文 Table 1 的 #Acts），以及钩子真正施加的 (层, 前向) 次数
            "au_plan": (_rel_to_repo(self.au_plan_path)
                        if self.suppressor == "austeer" else None),
            "au_variant": (next(iter(self._au_plans.values())).variant
                           if self._au_plans else None),
            "au_class": self.au_class if self.suppressor == "austeer" else None,
            "au_alpha": (next(iter(self._au_plans.values())).alpha
                         if self._au_plans else None),
            "au_n_acts": ({c: p.n_acts for c, p in self._au_plans.items()}
                          if self._au_plans else None),
            "au_prefill": self.au_prefill if self.suppressor == "austeer" else None,
            "au_applied": self._au_applied() if self._au_hooks else None,
            "temperature": self.temperature, "seed": self.seed,
            "timing": "same_step" if self.same_step else "lagged",
            "plan_every": self.plan_every,
            "prompt": prompt, "prompt_len": self._prompt_len,
            "prefill": prefill, "prefill_tokens": len(forced),
            "gen_tokens": len(gen_ids),
            "stopped_naturally": bool(stopped),
            "hit_cap": bool(len(gen_ids) >= self.max_new_tokens),
            "thinking_tokens": int(len(tok(thinking, add_special_tokens=False).input_ids)),
            "first_active_step": self._first_active_step,
            "active_steps": self._active_steps,
            "applied_layer_steps": int(st["applied_steps"]),
            # mean pre-suppression attention mass on the targeted positions
            "supp_mass_applied_layer": (round(st["mass_layer"] / st["applied_steps"], 4)
                                        if st["applied_steps"] else None),
            # own_layer 下没有「层平均计划」这回事，而且在第 li 层时 row_sum
            # 只累到第 li 层，算出来是个部分平均，会误导——直接不记。
            "supp_mass_layer_avg": (
                round(st["mass_avg"] / st["applied_steps"], 4)
                if st["applied_steps"] and self.plan_source == "layer_avg"
                else None),
            "cons_hist": cons_hist,
            # 提示侧每 token 平均注意力比值（prompt_mean_fraction）的逐步采样。
            # 保留它是为了能在**同一批攻击提示**内部比较成环与不成环的轨迹——
            # 提示长度在这批提示里是同分布的，所以这是「提示侧注意力低是因为在
            # 循环，还是因为提示长」的受控对照，不需要再跑一轮。
            # 两侧每 token 强度之比 ρ 可由它反推：
            #   p = pmf·P/L，ρ = p·G / (P·(1−p))
            "pmf_stride": self.ca_hist_every,
            "pmf_hist": [None if not np.isfinite(x) else round(float(x), 4)
                         for x in self._pmf_arr[:: self.ca_hist_every]],
            "cons_mean": round(float(np.mean(cons_hist)), 4) if cons_hist else None,
            "cons_tail_mean": (round(float(np.mean(cons_hist[-8:])), 4)
                               if cons_hist else None),
            "seconds": round(time.time() - t0, 1),
            "thinking": thinking, "full_text": text,
        }
        if self.det is not None:
            probes = self._probes
            res.update({
                "detector": Path(self.detector_path).name,
                "probe_every": self.probe_every, "tau_prob": self.tau_prob,
                "off_rule": self.off_rule, "burst_steps": self.burst_steps,
                "trigger_checkpoints": sorted(self.trigger_checkpoints),
                "n_bursts": self._bursts,
                "n_probes": len(probes),
                "n_attack_probes": sum(1 for p in probes if p["pred"] in ATTACK),
                # where the gate opened — always a deployed trigger checkpoint,
                # because that is the only grid probed while it is closed
                "gate_open_L": self._gate_open["L"] if self._gate_open else None,
                "gate_open_step": (self._gate_open["step"]
                                   if self._gate_open else None),
                "gate_open_pred": (self._gate_open["pred"]
                                   if self._gate_open else None),
                "final_pred": probes[-1]["pred"] if probes else None,
                "final_attack_prob": probes[-1]["attack_prob"] if probes else None,
                "n_release": sum(1 for e in self._events if e["action"] == "off"),
                "n_rearm": max(0, sum(1 for e in self._events
                                      if e["action"] == "on") - 1),
                "suppress_spans": self._spans,
                "suppressed_frac": (round(self._active_steps / len(gen_ids), 4)
                                    if gen_ids else None),
                "gate_events": self._events,
                "probes": probes,
            })
        if True:
            h = self._ca_hist
            # 提示侧的统计只看真正做了提示侧的那些记录（不在 prompt_layers 里的
            # 层记的是 None）
            hp = [r for r in h if r["c"] is not None]
            cs = [r["c"] for r in hp]
            aa = [r["a"] for r in h if r["a"] is not None]
            res.update({
                # AUSteer 基线不读参考曲线（注意力算子才用它），这里可以是 None
            "reference": (Path(self.reference_path).name
                          if self.reference_path else None),
                "ref_metric": self.ref_metric,
                "ref_normalize": self.ref_normalize,
                "ref_quantile": self.ref_quantile,
                "prompt_cap": self.prompt_cap,
                "prompt_direction": self.prompt_direction,
                "sink_positions": self.sink_positions,
                "flatten_layers_spec": self.flatten_spec,
                "flatten_layers": sorted(self.flatten_layers),
                "prompt_layers_spec": self.prompt_spec,
                "prompt_layers": sorted(self.prompt_layers),
                "gen_op": self.gen_op, "topm_rank": self.topm_rank,
                "stop_after_think": self.stop_after_think,
                "think_closed_step": self._think_closed_step,
                "top_m": self.top_m, "flatten_strength": self.flatten_strength,
                "a_min": self.a_min, "b_max": self.b_max,
                "flatten_group_by": self.flatten_group_by,
                "attack_class": self._attack_class,
                # 提示侧常数：中位数与两端，看干预到底有多强、是否长期贴着上限
                "c_median": round(float(np.median(cs)), 4) if cs else None,
                "c_min": round(float(np.min(cs)), 4) if cs else None,
                "c_max": round(float(np.max(cs)), 4) if cs else None,
                "a_median": round(float(np.median(aa)), 4) if aa else None,
                "prompt_share_mean": (round(float(np.mean(
                    [r["prompt_share"] for r in hp])), 4) if hp else None),
                "prompt_share_after_mean": (round(float(np.mean(
                    [r["prompt_share_after"] for r in hp])), 4) if hp else None),
                "sink_share_mean": (round(float(np.mean(
                    [r["sink_share"] for r in hp])), 4) if hp else None),
                "top_mass_mean": (round(float(np.mean(
                    [r["top_mass"] for r in h if r["top_mass"] is not None])), 4)
                    if aa else None),
                # 对齐算子特有：被压的串数（top-m 里在提示中出现过、且生成侧
                # 占比更高的）与生成侧三段的质量分布（被压 / 不动 / 回填，
                # 见 row_reshape §2b）
                "n_hit_mean": (round(float(np.mean(
                    [r["n_hit"] for r in h if "n_hit" in r])), 2)
                    if any("n_hit" in r for r in h) else None),
                "unchanged_mass_mean": (round(float(np.mean(
                    [r["unchanged_mass"] for r in h
                     if "unchanged_mass" in r])), 4)
                    if any("unchanged_mass" in r for r in h) else None),
                "rest_mass_mean": (round(float(np.mean(
                    [r["rest_mass"] for r in h if "rest_mass" in r])), 4)
                    if any("rest_mass" in r for r in h) else None),
                "ca_hist_layers": sorted(self.ca_hist_layers),
                # 逐层路径下 ca_hist 每步有多条，汇总量（c_median 等）是跨层跨步
                # 的；c 随深度怎么变要看这个
                "c_median_by_layer": ({
                    str(l): round(float(np.median(
                        [r["c"] for r in h if r.get("layer") == l])), 4)
                    for l in sorted({r["layer"] for r in h if "layer" in r})}
                    if any("layer" in r for r in h) else None),
                "ca_status": self._ca_status,
                # pmf 口径特有：目标 ≥ T/P 时顶格也够不着。ρ 口径下恒为 0。
                "prompt_infeasible_steps": self._ca_status.get(
                    "prompt:infeasible", 0),
                "ca_hist": h,
            })
            if self.dyn_rule:
                res.update({
                    "dyn_spec": self.dyn_spec,
                    "dyn_rule": self.dyn_rule,
                    "dyn_n": self.dyn_n,
                    "dyn_cands": sorted(self._dyn_hist),
                    "dyn_stride": self.dyn_stride,
                    "dyn_window": self.dyn_window,
                    "dyn_perstep": self.dyn_perstep,
                    "dyn_thresh": (None if not np.isfinite(self.dyn_thresh)
                                   else self.dyn_thresh),
                    "dyn_selected": self._dyn_selected,
                    # 逐步规则特有：每个候选层被判过多少次、其中压了多少次
                    "dyn_hit_by_layer": {str(l): self._dyn_hits.get(l, 0)
                                         for l in sorted(self._dyn_evals)},
                    "dyn_eval_by_layer": {str(l): v for l, v
                                          in sorted(self._dyn_evals.items())},
                    "dyn_hit_rate_by_layer": {
                        str(l): round(self._dyn_hits.get(l, 0) / v, 4)
                        for l, v in sorted(self._dyn_evals.items()) if v},
                    # 平均每个可施加步实际压了几层（与 top-n 的 n 直接可比）：
                    # 分母是「门开后被判定过的步数」（每个候选层每步各判一次，
                    # 所以取任一层的判定次数即可），一层都没压的步也算在内
                    "dyn_layers_per_step": (
                        round(sum(self._dyn_hits.values()) /
                              max(self._dyn_evals.values()), 3)
                        if self._dyn_evals else 0.0),
                    # 至少压了一层的步数 / 判定过的步数
                    "dyn_steps_with_any_layer": len(self._dyn_hit_steps),
                    "dyn_steps_evaluated": (max(self._dyn_evals.values())
                                            if self._dyn_evals else 0),
                    "dyn_decided_step": self._dyn_step,
                    # 决策时刻各候选层的逐层一致性均值（越低越「陷在循环里」）
                    "dyn_cons_by_layer": self._dyn_means,
                    # 逐层一致性的完整逐步轨迹（每 dyn_stride 步一个点），
                    # 用来事后做「静态深度成分 vs 样例交互成分」的方差分解
                    "dyn_cons_hist": {str(l): v
                                      for l, v in sorted(self._dyn_hist.items())},
                })
        if self._trace:
            res["trace"] = self._trace
        return res


# --------------------------------------------------------------------------- #
def _selftest_class_aware(reference: str) -> int:
    """不加载模型，检验 v5 乘子的几件事（`--selftest`）：

    **两个攻击类共用同一条合并曲线**（目标值与类别无关）、提示侧闭式解是否真的把
    目标量送到目标（主档 pmf / 封顶档各自精确命中）、长提示上两种归一化口径分别
    落在哪一档、目标平移（`--prompt-target-offset`）起点不跳且之后与曲线平行、
    两种生成侧算子（削向均匀 / 对齐提示侧占比）施加后总质量是否守恒、对齐算子
    是否逐串命中提示侧占比、位置集合是否按当前长度重建（含刚吐出的重复词）。
    """
    import types
    import torch

    def make(cls_name, *, P=60, G=200, flat=True, sink=0.65,
             prompt_mass=0.75, top_m=3, gen_op="flatten", overlap=True,
             prompt_layers=None, topm_rank="gen_mass"):
        rt = SuppressionRuntime.__new__(SuppressionRuntime)
        rt.torch, rt.model = torch, types.SimpleNamespace(device="cpu")
        rt.scheme, rt.reference_path = "class_aware", reference
        rt.reference = AttentionReference.load(reference)
        rt.ref_metric = rt.reference.metric      # 口径跟着曲线文件走
        rt.ref_normalize, rt.ref_quantile = DEFAULT_NORMALIZE, 0.5
        rt.sink_positions = 1
        rt.prompt_cap, rt.prompt_direction = 0.0, "both"
        rt.prompt_target_offset, rt._pmf_offset = "none", {}
        rt.top_m, rt.flatten_strength = top_m, 1.0
        rt.topm_rank = topm_rank
        rt.a_min, rt.b_max, rt.flatten_group_by = 0.05, 5.0, "string"
        rt.gen_op = gen_op
        rt.flatten_layers = {8, 9} if flat else set()
        rt.n_layers, rt.last_layer = 32, 31
        rt.apply_layers = set(range(32))
        # 提示侧的施加层：默认全部（方案原文），传 "gen" 则与生成侧算子同层
        rt.prompt_layers = (set(rt.flatten_layers) if prompt_layers == "gen"
                            else set(range(32)) if prompt_layers is None
                            else set(prompt_layers))
        rt.plan_source = "layer_avg"
        rt.ca_hist_layers = set(range(32))
        rt._ca_step, rt._ca_class = -1, None
        rt.mode, rt.burst_steps, rt.plan_every = "classifier", 0, 8
        # 逐层动态门：自测不走这条路，逐层乘子里会读它
        rt.dyn_perstep = False
        rt.stop_after_think, rt._think_closed = True, False
        rt.ca_hist_every = 1
        rt._active, rt._burst_left = True, 0
        rt._supp_step, rt._supp_flag, rt._supp_counted = -1, False, -1
        rt._step, rt._n_pos, rt._active_steps = 0, 0, 0
        rt._first_active_step, rt._ca_hist, rt._ca_status = None, [], {}
        rt._attack_class, rt._prompt_len = cls_name, P
        rng = np.random.default_rng(0)
        pool = ["aa", "bb", "LOOP", "LOOP", "LOOP", "cc"]
        prompt_toks = [f"p{i}" for i in range(P)]
        if overlap:                      # 提示里也出现 LOOP/aa，交集才非空
            prompt_toks[P - 3: P] = ["LOOP", "aa", "LOOP"]
        rt._tokens = prompt_toks + \
                     [pool[int(rng.integers(len(pool)))] for _ in range(G)]
        rt._vocab, rt._id_names = {}, []
        rt._ids_buf, rt._ids_len = np.empty(16, dtype=np.int64), 0
        for t in rt._tokens:
            rt._push_token_id(t)
        row = np.empty(P + G - 1)                  # 上一步的行：长度 T−1
        row[0] = sink
        row[1:P] = (prompt_mass - sink) / (P - 1)
        g = rng.random(G - 1) + 0.05
        g[[t == "LOOP" for t in rt._tokens[P:-1]]] *= 12
        row[P:] = (1.0 - prompt_mass) * g / g.sum()
        rt._last_row = row / row.sum()
        return rt

    for cls_name in REFERENCE_OF:
        if True:
            rt = make(cls_name)
            m_p, m_f = rt._class_aware_multipliers()
            h, P, T = rt._ca_hist[-1], rt._prompt_len, len(rt._tokens)
            mp = m_p.numpy().astype(np.float64)
            mf = m_f.numpy().astype(np.float64)
            assert mp.size == T == mf.size
            # 提示侧整段同一个乘子（含注意力汇，不作特殊处理），生成侧为 1
            assert np.allclose(mp[:P], mp[0]) and np.allclose(mp[P:], 1.0)
            row = rt._last_row
            new = row * mp[: row.size]
            new = new / new.sum()
            got = (new[:P].sum() / P) * row.size          # 主档口径 pmf
            got_share = new[:P].sum()
            # 落点必须与记录一致（乘子是 float32，容差按此定），且**精确命中它
            # 实际瞄的那一档目标**：主档时 pmf 命中，封顶档时占比命中。
            assert abs(got - h["ratio_after"]) / h["ratio_after"] < 1e-3, (got, h)
            if h["prompt_status"] == "ok":
                assert abs(got_share - h["target_share"]) < 1e-3, (got_share, h)
                if h["target_kind"] == "pmf":
                    assert abs(got - h["ratio_target"]) / h["ratio_target"] < 1e-3, h
            gen_before = row[P:].sum()
            gen_after = (row[P:] * mf[P: row.size]).sum()
            assert abs(gen_after - gen_before) < 2e-5, (gen_before, gen_after)
            assert "LOOP" in h["top_tokens"]
            loop_pos = [j for j in range(P, T) if rt._tokens[j] == "LOOP"]
            assert np.allclose(mf[loop_pos], mf[loop_pos[0]])
            print(f"  {cls_name:22s} c={h['c']:8.3f} "
                  f"pmf {h['ratio']:.3f}→{h['ratio_after']:.3f} "
                  f"(目标 {h['ratio_target']:.3f}/{h['target_kind']}, "
                  f"{h['prompt_status']}) "
                  f"a={h['a']:.3f} b={h['b']:.3f} 位置数={h['n_pos']} "
                  f"生成侧质量守恒 OK")

    # 合并曲线：两个攻击类拿到的**目标值必须逐位相同**（方向仍可由
    # --prompt-direction expected 分开夹，但那是另一个开关）。
    tgt = {}
    for cls_name in REFERENCE_OF:
        rt = make(cls_name, gen_op="prompt_align", top_m=3)
        rt._class_aware_multipliers()
        tgt[cls_name] = rt._ca_hist[-1]["ratio_target"]
    assert len(set(tgt.values())) == 1, tgt
    print(f"  合并曲线：两个攻击类目标相同 {list(tgt.values())[0]:.4f}"
          f"（{' = '.join(tgt)}）  OK")

    # 循环思考的真实形态：提示很长（2000 token）。**落在哪一档由归一化口径决定**：
    #   position_prompt_len —— 目标随提示长度缩小（系数 d≈−0.8），所要求的提示侧
    #     质量占比通常 < 1，走主档 pmf，精确命中曲线；
    #   position —— 目标不含长度因子，把良性短提示上的水平搬到长提示上，所要求的
    #     占比普遍 ≥ 1（无解），改瞄封顶占比 SHARE_CEILING，强度由那个常数决定。
    # 两档都必须精确命中**自己那一档**的目标。
    for nm in ("position_prompt_len", "position"):
        rt = make("repetitive_reasoning", P=2000, G=600, sink=0.45,
                  prompt_mass=0.70)
        rt.ref_normalize = nm
        rt._class_aware_multipliers()
        h = rt._ca_hist[-1]
        if h["target_kind"] == "cap":
            assert abs(h["target_share"] - RR.SHARE_CEILING) < 1e-9, h
        else:
            assert abs(h["ratio_after"] - h["ratio_target"]) < 1e-3, h
        assert abs(h["prompt_share_after"] - h["target_share"]) < 1e-3, h
        print(f"  长提示 {nm:20s} 档={h['target_kind']:4s} "
              f"目标 pmf={h['ratio_target']:7.3f} c={h['c']:8.3f} 提示侧占比 "
              f"{h['prompt_share']:.3f}→{h['prompt_share_after']:.3f}"
              f"（{'抬' if h['c'] > 1 else '压'}）  OK")
    # 长度项的作用方向：提示越长目标越低，这正是长提示能落回主档的原因
    rt = make("repetitive_reasoning", P=2000, G=600, sink=0.45, prompt_mass=0.70)
    t_len = rt.reference.target_ratio("repetitive_reasoning", 599, 2000,
                                      normalize="position_prompt_len")
    t_pos = rt.reference.target_ratio("repetitive_reasoning", 599, 2000,
                                      normalize="position")
    assert t_len < t_pos, (t_len, t_pos)
    print(f"  长度项方向：P=2000 上 目标 {t_pos:.3f}(position) → "
          f"{t_len:.3f}(position_prompt_len)  OK")

    # pmf 目标要求提示侧质量占比 ≥ 1 时改瞄封顶占比，c 有限且精确命中封顶值
    rt = make("repetitive_reasoning", P=2000, G=600, sink=0.45,
              prompt_mass=0.70)
    T, row = len(rt._tokens), rt._last_row
    bound = T / 2000
    ps = RR.solve_prompt_scale(row, 2000, 1.5 * bound, cap=0.0)
    assert ps.target_kind == "cap" and ps.status == "ok", ps
    assert abs(ps.prompt_share_after - RR.SHARE_CEILING) < 1e-9, ps
    ps2 = RR.solve_prompt_scale(row, 2000, 0.9 * bound, cap=0.0)
    assert ps2.target_kind == "pmf" and abs(ps2.ratio_after
                                            - 0.9 * bound) < 1e-9, ps2
    print(f"  pmf 上界 T/P={bound:.4f}：越界改瞄封顶占比 {RR.SHARE_CEILING} → "
          f"落点 {ps.prompt_share_after:.4f}（c={ps.c:.3f}）；界内仍命中 pmf  OK")

    # --- 目标平移：起点不跳、之后与参考曲线平行（--prompt-target-offset）-----
    for off_mode in ("trigger", "trigger_log"):
        rt = make("repetitive_reasoning", gen_op="prompt_align", top_m=3)
        rt.prompt_target_offset = off_mode
        rt._class_aware_multipliers()
        h, row, Pn = rt._ca_hist[-1], rt._last_row, rt._prompt_len
        raw0, delta = h["ratio_target_raw"], rt._pmf_offset["avg"]
        # 起点不跳：实际瞄的目标就是实测值 ⇒ c = 1，落点 = 出发点
        assert abs(h["c"] - 1.0) < 1e-6, h
        assert abs(h["ratio_target"] - h["ratio"]) < 1e-3, h
        assert abs(h["ratio_after"] - h["ratio"]) < 1e-3, h
        # 平移量整段只锁定一次，目标此后随曲线走 ⇒ 与曲线平行
        g1, o1 = rt._offset_target("avg", raw0 * 1.3, row, Pn)
        g2, o2 = rt._offset_target("avg", raw0 * 0.8, row, Pn)
        assert o1 == o2 == delta, (o1, o2, delta)
        if off_mode == "trigger":        # 线性域平行：增量与曲线逐点相同
            assert abs((g1 - g2) - raw0 * 0.5) < 1e-9, (g1, g2)
        else:                            # 对数域平行：比值与曲线逐点相同
            assert abs(np.log(g1 / g2) - np.log(1.3 / 0.8)) < 1e-12, (g1, g2)
        # 门重开 / 复检改判类别会清空 ⇒ 下一段按新的触发点重新锁定
        rt._pmf_offset.clear()
        g3, o3 = rt._offset_target("avg", raw0 * 1.3, row, Pn)
        assert o3 != delta and abs(g3 - h["ratio"]) < 1e-3, (o3, delta, g3)
        print(f"  目标平移 {off_mode:11s} Δ={delta:8.4f} 起点 c={h['c']:.3f}"
              f"（曲线 {raw0:.3f} → 目标 {h['ratio_target']:.3f} = 实测 "
              f"{h['ratio']:.3f}）之后与曲线平行  OK")

    # --- 生成侧算子二：逐串对齐提示侧占比，只压不抬（row_reshape §2b）--------
    for cls_name in REFERENCE_OF:
        rt = make(cls_name, gen_op="prompt_align", top_m=3)
        m_p, m_f = rt._class_aware_multipliers()
        h, P = rt._ca_hist[-1], rt._prompt_len
        mf = m_f.numpy().astype(np.float64)
        row = rt._last_row
        assert h["gen_op"] == "prompt_align" and h["gen_status"] == "ok", h
        # 生成侧总质量精确不变（乘子是 float32，容差按此定）
        gen_before, gen_after = row[P:].sum(), (row[P:] * mf[P: row.size]).sum()
        assert abs(gen_after - gen_before) < 2e-5, (gen_before, gen_after)
        # 被压的每个串：施加后它在生成侧的占比 = 它在提示侧的占比。这里没有
        # 任何夹子，所以必须精确命中（LOOP 要的系数小到 0.005 也照给）。
        for t, a in h["a_by_token"].items():
            assert a < 1.0, (t, a)                       # 只压不抬
            pos = [j for j in range(P, row.size) if rt._tokens[j] == t]
            assert np.allclose(mf[pos], mf[pos[0]]), t   # 同一个串同一个系数
            got = (row[pos] * mf[pos]).sum() / gen_after
            assert abs(got - h["prompt_share_by_token"][t]) < 1e-3, (t, got, h)
        # 不动段：提示里没有的串 + 占比已不高于提示侧的串，逐位不变
        for t in h["keep_tokens"] + h["below_tokens"]:
            pos = [j for j in range(P, row.size) if rt._tokens[j] == t]
            assert np.allclose(mf[pos], 1.0, atol=1e-6), (t, mf[pos][:3])
        # 只有 top-m 之外的位置走回填，且只压不抬 ⇒ b ≥ 1
        rest = [j for j in range(P, row.size)
                if rt._tokens[j] not in h["a_by_token"]
                and rt._tokens[j] not in h["keep_tokens"]
                and rt._tokens[j] not in h["below_tokens"]]
        assert rest and np.allclose(mf[rest], h["b"], atol=1e-6), h["b"]
        assert h["b"] >= 1.0, h["b"]
        assert np.allclose(mf[:P], m_p.numpy()[:P])      # 提示侧两个乘子一致
        print(f"  {cls_name:22s} 对齐提示侧 被压={list(h['a_by_token'])} "
              f"系数={ {k: round(v, 4) for k, v in h['a_by_token'].items()} } "
              f"不动={h['keep_tokens'] + h['below_tokens']} 回填 b={h['b']:.3f} "
              f"（三段质量 {h['top_mass']}/{h['unchanged_mass']}/{h['rest_mass']}）"
              f" 生成侧质量守恒 OK")

    # 提示里一个 top-m 词串都没出现 → 没有可压的对象，生成侧不动
    rt = make("repetitive_string", gen_op="prompt_align", overlap=False)
    m_p, m_f = rt._class_aware_multipliers()
    assert torch.equal(m_p, m_f), "无可压对象时不应改动生成侧"
    assert rt._ca_hist[-1]["gen_status"] == "no_target", rt._ca_hist[-1]
    print("  无可压对象（提示里没有任何 top-m 词串）→ 生成侧不动  OK")

    # --- 逐层自测自抑：每层用自己的行算自己的系数（plan_source=own_layer）------
    rt = make("repetitive_reasoning", gen_op="prompt_align", top_m=3)
    rt.plan_source = "own_layer"
    P, T = rt._prompt_len, len(rt._tokens)
    base = np.concatenate([rt._last_row, [1e-3]])      # 本步本层的行：长度 T
    base = base / base.sum()
    rng = np.random.default_rng(1)
    seen_c, mults = {}, {}
    for li in (0, 8, 9, 20, 31):                       # 8/9 是生成侧算子的层
        # 每层给一条不同的行（真实情况就是各层各不相同）
        row_l = base * rng.uniform(0.5, 1.5, T)
        row_l = row_l / row_l.sum()
        m = rt._class_aware_layer_multiplier(li, torch.tensor(row_l))
        assert m is not None and m.numel() == T, (li, None if m is None else m.numel())
        mults[li] = m.numpy().astype(np.float64)
        seen_c[li] = float(mults[li][0])               # 提示侧首位就是 c
        # 提示侧：整段同一个常数；生成侧：非算子层必须全 1
        assert np.allclose(mults[li][:P], mults[li][0]), li
        if li not in rt.flatten_layers:
            assert np.allclose(mults[li][P:], 1.0), (li, mults[li][P:][:5])
        else:
            gen_after = (row_l[P:] * mults[li][P:]).sum()
            assert abs(gen_after - row_l[P:].sum()) < 2e-5, li
    # 各层的 c 互不相同 —— 系数确实是逐层独立算出来的，不是一个常数广播
    assert len(set(round(v, 6) for v in seen_c.values())) == len(seen_c), seen_c
    # 生成侧算子层之间，被压的词串集合也各算各的
    h_by_layer = {r["layer"]: r for r in rt._ca_hist if "layer" in r}
    assert set(h_by_layer) == {0, 8, 9, 20, 31}, sorted(h_by_layer)
    assert all(h_by_layer[li]["gen_status"] == "off"
               for li in (0, 20, 31)), h_by_layer[0]
    assert h_by_layer[8]["a_by_token"] and h_by_layer[9]["a_by_token"]
    # 门与计数按解码步算：五层只计一次
    assert rt._active_steps == 1, rt._active_steps
    print(f"  逐层自测自抑 c={ {k: round(v, 3) for k, v in seen_c.items()} } "
          f"（各层独立），生成侧算子只在 {sorted(rt.flatten_layers)} 层，"
          f"质量守恒，active_steps 每步只计一次  OK")

    # --- 两半施加在同一批层 + top-m 按占比差排 ------------------------------
    # （--prompt-layers gen --topm-rank share_diff）：提示侧常数只在生成侧算子的
    # 那两层上做，其余层这一步完全不动；名额全部发给「提示里出现过、且生成侧
    # 占比更高」的串，所以不动段为空、被压串数 = top-m 实际取到的串数。
    rt = make("repetitive_reasoning", gen_op="prompt_align", top_m=3,
              prompt_layers="gen", topm_rank="share_diff")
    rt.plan_source = "own_layer"
    P, T = rt._prompt_len, len(rt._tokens)
    base = np.concatenate([rt._last_row, [1e-3]])
    base = base / base.sum()
    for li in (0, 20, 31):                             # 不在两半的任何一半里
        assert rt._class_aware_layer_multiplier(li, torch.tensor(base)) is None, li
    for li in (8, 9):
        m = rt._class_aware_layer_multiplier(li, torch.tensor(base))
        assert m is not None and m.numel() == T, li
        mm = m.numpy().astype(np.float64)
        assert np.allclose(mm[:P], mm[0]) and abs(mm[0] - 1.0) > 1e-6, mm[:3]
        assert not np.allclose(mm[P:], 1.0), li          # 生成侧确实动了
        assert abs((base[P:] * mm[P:]).sum() - base[P:].sum()) < 2e-5, li
    hl = {r["layer"]: r for r in rt._ca_hist if "layer" in r}
    assert set(hl) == {8, 9}, sorted(hl)                 # 别的层连诊断都不记
    assert all(r["unchanged_mass"] == 0.0 for r in hl.values()), hl
    assert all(r["n_hit"] == r["n_top"] > 0 for r in hl.values()), hl
    assert rt._active_steps == 1, rt._active_steps
    print(f"  提示侧层=生成侧层{sorted(rt.flatten_layers)}（其余层不动）+ "
          f"top-m 按占比差排：被压串数 {hl[8]['n_hit']}/{hl[8]['n_top']}，"
          f"不动段为空，质量守恒  OK")

    rt = make("repetitive_reasoning", flat=False)
    m_p, m_f = rt._class_aware_multipliers()
    assert torch.equal(m_p, m_f) and rt._ca_hist[-1]["gen_status"] == "off"
    rt = make("repetitive_string")
    rt._attack_class = None
    assert rt._class_aware_multipliers() == (None, None)
    print("  削峰关闭 / 无攻击类别时不干预  OK\nSELFTEST PASSED")
    return 0


def main() -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--selftest", action="store_true",
                    help="不加载模型，自测 v5 乘子（需 --reference）")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--mode", default="classifier", choices=MODES,
                    help="off = 只检测不干预的基线；classifier = 部署形态的门")
    ap.add_argument("--layers", default=None,
                    help="last | all（默认） | none | closest8 | random:<n> | "
                         "mid:<n> | 逗号列表（每项可写 a-b 区间）。实际施加面是它"
                         "与 --prompt-layers / --flatten-layers 的交")
    ap.add_argument("--reference", default=None,
                    help="attention_reference.py 生成的参考曲线 JSON，"
                         "scheme=class_aware 必需")
    ap.add_argument("--ref-normalize", default=DEFAULT_NORMALIZE,
                    choices=list(NORMALIZE_MODES),
                    help="position_prompt_len（默认）= 同时按提示长度归一——"
                         "合并曲线必须带这一项，否则两类良性各带 ±0.25~±0.41 的"
                         "系统偏置；position = 只按生成位置（对照口径）")
    ap.add_argument("--ref-quantile", type=float, default=0.5,
                    help="目标取参考轨迹残差分布的哪个分位（0.5 = 中位）")
    ap.add_argument("--sink-positions", type=int, default=1,
                    help="诊断字段 sink_share 把提示开头几位当作注意力汇；\n"
                         "**不影响求解与施加**——提示侧整段同乘一个 c")
    ap.add_argument("--prompt-cap", type=float, default=0.0,
                    help="提示侧常数 c 的夹取上限（下限取其倒数）；"
                         "0 = 不设限制（默认）")
    ap.add_argument("--prompt-direction", default="both",
                    choices=["both", "expected", "off"],
                    help="expected = 只允许方案预期的方向（循环思考只抬、"
                         "循环字符只压）；off = 不动提示侧（只留生成侧削峰）")
    ap.add_argument("--flatten-layers", default=DEFAULT_FLATTEN_LAYERS,
                    help=f"生成侧削峰施加的层，默认 {DEFAULT_FLATTEN_LAYERS}"
                         "（非汇可用质量最高的中段层带）；none 关闭削峰")
    ap.add_argument("--prompt-layers", default=DEFAULT_PROMPT_LAYERS,
                    help="提示侧常数施加的层：all（默认，方案原文口径=全部施加层）"
                         "| gen（与生成侧算子同层，两半施加面重合）| 层号写法")
    ap.add_argument("--prompt-target-offset", default="none",
                    choices=list(PROMPT_TARGET_OFFSETS),
                    help="曲线目标是否平移到开始抑制那一刻的实测水平："
                         "none（默认，逐步命中曲线的水平，起点上是一个跳变）"
                         "| trigger（线性平移：目标 = 曲线 − 触发点差值）"
                         "| trigger_log（对数域平移，目标恒为正）。"
                         "平移后起点 c=1、之后实际轨迹与参考曲线平行")
    ap.add_argument("--gen-op", default="flatten", choices=list(RR.GEN_OPS),
                    help="生成侧算子：flatten = 把 top-m 削向均匀；"
                         "prompt_align = 把「top-m ∩ 提示里出现过的词串」逐串"
                         "对齐到它们在提示侧的质量占比")
    ap.add_argument("--top-m", type=int, default=8,
                    help="生成侧参与的 token 串个数（对齐算子再与提示词串取交集，"
                         "交集外的 top-m 串不动，只有 top-m 之外的位置回填）")
    ap.add_argument("--topm-rank", default=RR.DEFAULT_TOPM_RANK,
                    choices=list(RR.TOPM_RANKS),
                    help="top-m 的名额按什么排：gen_mass = 生成侧质量（初版）；"
                         "share_diff = 两侧占比差（生成侧占比 − 提示侧占比），"
                         "名额只发给相对提示侧被过度使用的串")
    ap.add_argument("--flatten-strength", type=float, default=1.0,
                    help="1.0 = 一步到位（削到均匀水平 / 对齐到提示侧占比），0 = 不动")
    ap.add_argument("--a-min", type=float, default=0.05,
                    help="削峰算子的系数下限（对齐算子不设量程限制）")
    ap.add_argument("--b-max", type=float, default=5.0,
                    help="削峰算子的回填上限（同上）")
    ap.add_argument("--flatten-group-by", default="string",
                    choices=list(RR.GROUP_BY))
    ap.add_argument("--stop-after-think", dest="stop_after_think",
                    action="store_true", default=True,
                    help="思考段结束（</think>）之后停止抑制（默认开）")
    ap.add_argument("--suppress-after-think", dest="stop_after_think",
                    action="store_false",
                    help="即使思考段已结束也继续抑制（旧行为）")
    ap.add_argument("--ca-hist-every", type=int, default=32,
                    help="v5 逐步诊断（c/a/b/占比）每多少步记一条")
    ap.add_argument("--ca-hist-layers", default="sample",
                    help="逐层路径下诊断记哪些层：all / gen（只记生成侧算子的层）"
                         "/ sample（默认：生成侧层 + 均匀补 4 个提示侧层）/ 层号")
    ap.add_argument("--plan-source", default=None, choices=PLAN_SOURCES,
                    help="own_layer（默认）= 每个施加层用自己这一步的行解自己的"
                         "系数，零滞后，且施加层可低于末层、从而具备跨步记忆；"
                         "layer_avg = 系数算在上一步的层平均行上，滞后一步")
    ap.add_argument("--plan-every", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--trace-every", type=int, default=64)
    ap.add_argument("--detector", default=None,
                    help="classifier JSON export (required for mode=classifier*)")
    ap.add_argument("--probe-every", type=int, default=32,
                    help="run the classifier whenever total length %% this == 0")
    ap.add_argument("--tau-prob", type=float, default=DEFAULT_TAU_PROB)
    ap.add_argument("--off-rule", default="argmax", choices=OFF_RULES)
    ap.add_argument("--burst-steps", type=int, default=8,
                    help="steps of suppression bought by one loop verdict "
                         "(0 = suppress continuously while the gate is open)")
    ap.add_argument("--trigger-checkpoints", default=None,
                    help=f"comma list; default {CHECKPOINTS}")
    # --- AUSteer 对比基线（arXiv:2602.04428）---------------------------------
    ap.add_argument("--suppressor", default="row_reshape",
                    choices=list(SUPPRESSORS),
                    help="干预算子：row_reshape = 主方案（重塑注意力行）；"
                         "austeer = 对比基线（按 AU 计划缩放线性层输入维度）")
    ap.add_argument("--au-plan", default=None,
                    help="AU 计划 JSON（austeer_localize.py 产出）")
    ap.add_argument("--au-class", default=AU.UNION,
                    help=f"用计划里的哪一类 AU：{AU.UNION} | repetitive_reasoning"
                         f" | repetitive_string | {AU_CLASS_AUTO}"
                         f"（按门开那一刻的判类选，仅 mode=classifier）")
    ap.add_argument("--au-top-k", type=int, default=None,
                    help=f"截到前 k 个 AU（论文主实验上限 {AU.PAPER_TOP_K_CAP}）；"
                         f"不给则用计划里的全部")
    ap.add_argument("--au-alpha", type=float, default=None,
                    help="全局强度因子 α，覆盖计划里记录的值")
    ap.add_argument("--au-decode-only", dest="au_prefill",
                    action="store_false", default=True,
                    help="只在解码步施加，提示编码阶段放过"
                         "（默认 prefill 也施加，与论文的挂钩方式一致）")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest:
        if not a.reference:
            ap.error("--selftest 需要 --reference")
        return _selftest_class_aware(a.reference)
    if not a.model or not a.prompt:
        ap.error("--model 与 --prompt 为必填（除非 --selftest）")

    rt = SuppressionRuntime(
        a.model, gpu=a.gpu, mode=a.mode, layers=a.layers,
        plan_every=a.plan_every, max_new_tokens=a.max_new_tokens,
        temperature=a.temperature, detector=a.detector,
        plan_source=a.plan_source,
        probe_every=a.probe_every, tau_prob=a.tau_prob, off_rule=a.off_rule,
        burst_steps=a.burst_steps,
        trigger_checkpoints=([int(x) for x in a.trigger_checkpoints.split(",")]
                             if a.trigger_checkpoints else None),
        reference=a.reference,
        ref_normalize=a.ref_normalize, ref_quantile=a.ref_quantile,
        sink_positions=a.sink_positions,
        prompt_target_offset=a.prompt_target_offset,
        prompt_cap=a.prompt_cap, prompt_direction=a.prompt_direction,
        flatten_layers=a.flatten_layers, prompt_layers=a.prompt_layers,
        gen_op=a.gen_op, top_m=a.top_m, topm_rank=a.topm_rank,
        flatten_strength=a.flatten_strength, a_min=a.a_min, b_max=a.b_max,
        flatten_group_by=a.flatten_group_by,
        stop_after_think=a.stop_after_think, ca_hist_every=a.ca_hist_every,
        ca_hist_layers=a.ca_hist_layers,
        suppressor=a.suppressor, au_plan=a.au_plan, au_class=a.au_class,
        au_top_k=a.au_top_k, au_alpha=a.au_alpha, au_prefill=a.au_prefill)
    r = rt.run(a.prompt, trace_every=a.trace_every)
    print(f"\n[supp] gen_tokens={r['gen_tokens']} eos={r['stopped_naturally']} "
          f"cons_mean={r['cons_mean']} tail={r['cons_tail_mean']} "
          f"active_steps={r['active_steps']} secs={r['seconds']}")
    if r.get("attack_class") is not None:
        gen_cn = ("生成侧对齐提示侧" if r["gen_op"] == "prompt_align"
                  else "生成侧削峰")
        print(f"[supp] 类别={r['attack_class']} 提示侧常数 c 中位={r['c_median']} "
              f"[{r['c_min']}, {r['c_max']}] {gen_cn}系数 a 中位={r['a_median']} "
              f"提示侧占比 {r['prompt_share_mean']}→{r['prompt_share_after_mean']} "
              f"(其中汇 {r['sink_share_mean']}) 生成侧 top-m 占比"
              f"={r['top_mass_mean']} 状态={r['ca_status']}")
    for e in r.get("gate_events", []):
        print("    gate", e)
    for t in r.get("trace", []):
        print("   ", t)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(r, f, ensure_ascii=False, indent=1)
        print(f"[supp] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
