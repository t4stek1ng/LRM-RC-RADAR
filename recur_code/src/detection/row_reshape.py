"""v5 抑制的两个行内算子（纯 numpy，无模型无 GPU）。

一次干预把当前查询 token 的注意力行 r（softmax 之后、长度 T = 提示长 P + 生成长 G）
改写成 r ∘ m，再按行重归一。乘子 m 只有三种取值：

    m[j] = c    j 在提示侧              —— 提示侧缩放，**所有层所有头**
    m[j] = a    j 在生成侧且是重点位     —— 生成侧削峰/对齐，**部分层的所有头**
    m[j] = b    j 在生成侧其余位置       —— 生成侧回填，同上

（生成侧有两种算子：§2 削向均匀，重点位共用一个 a；§2b 对齐提示侧占比，重点位
逐串一个 a_t，且 top-m 里提示中没出现过的串保持 m[j] = 1 不动。）

## 1 提示侧缩放：两档目标，闭式，永远有解

参考量是**提示侧每 token 平均强度与全行每 token 平均强度之比**
pmf = (Σ提示 / P) ÷ (Σ全行 / T)（离线字段 `prompt_mean_fraction`）。选它是因为
**它就是检测器的特征**——`detector.py` 的第二个输入是这条序列在 [0, i] 上的累计
斜率，于是「检测判它偏了」和「抑制把它拉回来」说的是同一个量。

### 求解统一走「目标提示侧质量占比」

设这一行归一后提示侧质量占比 p、生成侧 g = 1 − p。提示侧整体乘 c 再重归一之后，
占比变成 c·p/(c·p+g)。要它等于 `target_share`，闭式解是

        c = target_share / (1 − target_share) · g / p

于是问题只剩「target_share 取多少」。

### 两档目标：为什么需要第二档

pmf 与占比是同一件事的两种读法：**pmf = 提示侧质量占比 × T/P**。所以想达到目标
pmf* 就等于要求占比达到

        share_required = pmf* · P / T

**这个数可能 ≥ 1** —— 那就是要求提示侧拿走全行 100% 以上的注意力质量，**没有任何
注意力分布能实现它**（不是夹子太紧，是这个目标本身不存在）。原因是 pmf 里带着
T/P 这个长度因子：参考曲线拟合在**良性**轨迹上（提示长 75–948），而熵下降攻击的
提示长 1901–2492，同一个生成位置上两者的 T/P 差一个数量级，良性那边的 pmf 值搬到
攻击这边就越界了。

所以分两档（`ratio_target` 是 pmf 目标，`share_ceiling` 是封顶占比）：

| 情形 | 瞄的目标 | `target_kind` |
|---|---|---|
| `share_required < 1` | 就用它，等价于精确命中 pmf* | `"pmf"` |
| `share_required ≥ 1` | 改瞄**封顶占比** `SHARE_CEILING`（默认 0.95）：目标既然是「提示侧越高越好到越界」，就把提示侧推到这个封顶为止，给生成侧留 5% | `"cap"` |

两档都是精确命中一个可达的目标（封顶档命中的是那个常数），**`status` 不再有
`infeasible` 这一档**。

> **封顶档的强度由常数决定，不由参考曲线决定**（2026-09-11 改）。此前这一档瞄的是
> 参考轨迹在同一生成位置上的提示侧质量占比（离线字段 `prompt_fraction`）——不含
> 长度因子、恒在 (0,1) 内，所以也永远可达，且强度仍来自曲线；代价是那条曲线拟合
> 质量差（R² 0.59/0.77）、方向还随施加层翻转。换成封顶常数之后这两条代价没有了，
> 换来的是新的一条：**攻击提示普遍很长 ⇒ 这一档几乎总是被触发 ⇒ 提示侧强度实际上
> 由 0.95 这一个数说了算**，参考曲线只在短提示上还起作用。
>
> 更早的历史教训（仍然有效）：最初这里对 pmf 目标解「乘完之后精确等于目标」的
> 方程，无解时退到一个兜底常数（生成侧只留 1e-6），结果 c ≈ 10⁶。封顶档与它的
> 区别在于封顶值是**占比**（有界、可解释、落点可验证），而不是直接给 c 一个数。

**不对注意力汇做特殊处理**（2026-08-27 定）：全层平均约 71% 的注意力质量压在提示
首位这个汇上（`suppression_v4_material.md` F6），但求解与施加都把它当成普通提示
token——提示侧整段同乘一个 c。曾经提供过「汇位置不动、由非汇段承担整段调整量」的
第二种施加范围，已删除：那一支在循环字符类上恒无解（那一类提示侧质量的 97.5% 就是
这个汇，光靠汇就超过目标），而且它给方案多加了一个没有实验支持的自由度。
`sink_positions` 现在只用来算诊断字段 `sink_share`。

## 2 生成侧削峰：两个系数，总质量不变

循环的共同特征是生成侧的注意力**集中在少数几个 token 上**。抑制的做法是把生成侧
按 token 串聚合后质量最高的 top-m 个串所在的**全部位置**乘以 a < 1，其余生成位置
乘以 b > 1，并要求两者之后**生成侧总质量精确不变**：

        a·M + b·(Gmass − M) = Gmass        （M = top-m 位置的质量）

于是只剩一个自由度。本模块用「向均匀分配靠拢」来定它：这些位置在**完全均匀**的
生成侧分布下应得的质量是 U = Gmass · n_S / n（n_S 是这些位置的个数，n 是生成位置
总数），取

        M' = M + λ·(U − M)      λ = strength ∈ [0,1]
        a  = M' / M             b = (Gmass − M') / (Gmass − M)

λ=1 表示把这些位置**一步压到均匀水平**，λ=0 表示不动。a 有下限 `a_min`、b 有上限
`b_max`（生成侧几乎全部质量都在 top-m 上时 b 会爆炸），任一侧被夹住之后另一侧按
质量守恒重算，所以**总质量守恒始终精确成立**，被牺牲的只是均匀化的程度。

总质量守恒不是可有可无的美学：提示侧的 c 是按「生成侧总质量不变」解出来的，两个
算子只有在这个条件下才互不干扰——组合乘子重归一后的提示侧占比，恰好等于只做提示
侧缩放时的结果。

## 2b 生成侧对齐提示侧（`gen_op="prompt_align"`）

§2 把 top-m 削向**均匀**，均匀是一个与内容无关的参照。另一种参照来自提示本身：
一个词串在提示里占多大分量，它在生成侧就该占多大分量——循环的表现正是某几个词串
在生成侧的占比远高于它们在提示里的占比。于是把生成侧 top-m 词串里**占比高于提示侧
的那些**压回去：

        目标_t = 该串在提示侧的质量占比 = 提示质量(t) / 提示侧总质量
        当前_t = 该串在生成侧的质量占比 = 生成质量(t) / 生成侧总质量
        a_t    = 目标_t / 当前_t          （λ=strength 时 a_t = 1 + λ(目标/当前 − 1)）

举例：top-m 是 ['a','b','c']，提示里只出现过 'a'、'b'；'a' 在生成侧占 25%、在提示侧
占 10%，则生成侧**每一个** 'a' 位置都乘 0.4。

**只压不抬**：a_t 只在目标 < 当前时才施加。一个串若在提示里占比更高（生成侧还没
用够），保持不动——不会被抬上去。生成侧因此切成**三段，互不重叠**：

        被压段：top-m ∩ 提示词串，且生成侧占比 > 提示侧占比    × a_t < 1
        不动段：top-m 里其余的串（提示中没有的 + 占比已不高的） × 1
        回填段：top-m 之外的位置                               × b > 1

回填只落在 top-m 之外的位置上，由「生成侧总质量精确不变」唯一解出（M = 被压段质量，
K = 不动段质量，R = 回填段质量 = Gmass − M − K）：

        b = (R + M − Σ a_t·质量_t) / R

因为只压不抬，Σ a_t·质量_t ≤ M，所以 **b ≥ 1 恒成立**，不存在无解的情形。

与 §2 的三点区别：

  - **逐串一个系数**，不是所有被选位置共用一个 a——每个串各自对齐自己的提示侧占比；
  - 目标随提示走，而不是随位置数走：一个在提示里本来就很重的词，不会被压到均匀
    水平以下；
  - **不动段的存在**意味着 top-m 里"自己长出来的"循环词、以及提示里本来就更重的词，
    既不被压、也不被回填抬高，这一段在干预前后逐位不变。

**系数不设量程限制**：a_t 就是精确对齐所需的值（循环词两侧占比能差上百倍，a_t 到
0.005 是常态），b 由守恒唯一确定。代价是回填段被不动段挤小之后 b 会很大，`rest_mass`
与 `b` 都记在结果里就是为了看这个。

因为守恒精确成立，它和 §1 的提示侧常数互不干扰。提示侧各串的占比**对 c 免疫**
（提示侧整段同乘一个常数，串间占比不变），所以两个算子的先后无关紧要。

## 2c top-m 的名额发给谁（`rank_by`）

上面两个算子都先取「top-m 个词串」，初版（`rank_by="gen_mass"`）按**生成侧质量**
排。这条排序有个副作用：生成侧质量最高的串，往往是提示里本来就很重的词（问题里的
关键名词、数字），它们在两侧的占比其实相当——

  - §2b 下它们**占着名额却不被压**（占比不高于提示侧 → 落进不动段），m=5 时经常
    只剩一两个串真的被压；
  - §2 下它们被压，但压的不是循环，而是模型本来就该看的词。

`rank_by="share_diff"` 改按**两侧占比差**排：

        差_t = 该串在生成侧的质量占比 − 它在提示侧的质量占比

这个差正是「相对提示侧被过度使用了多少」，也正是 §2b 想纠正的量。两个算子的候选
集因此不同：

  - §2b（只压不抬）只有「提示里出现过、且差 > 0」的串可压，名额就只发给它们，
    按差从大到小取 m 个 —— **m 个名额全部落在真正会被压的串上，不动段为空**；
  - §2（削向均匀）不要求该串在提示里出现过（提示里没有的串差 = 它的生成侧占比，
    自然排在前面），只要求差 > 0；没有任何串的差为正时这一步不动。

排序换掉之后 §2b 的三段退化成两段（被压 / 回填），`unchanged_mass` 恒为 0——
诊断里看到它不为 0，就说明这条臂跑的还是 `gen_mass`。

## 3 与检测特征的关系

这两个算子瞄的是**正常思考轨迹的提示侧强度**与**生成侧分布的均匀度**，不是一致性
指标本身。一致性指标（`attn_consistency.py`）在这条链路里只做两件事：当分类器的第一个
输入特征，以及当逐层动态选层的判据；它不决定乘子。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

# 提示侧对齐的目标量。pmf 与**检测特征同口径**（detector.py 的第二个特征就是
# 它的累计斜率）；当 pmf 的目标要求提示侧质量占比 ≥ 1（无解，见模块文档 §1）
# 时改瞄下面这个封顶占比。
PROMPT_METRIC = "prompt_mean_fraction"
# 封顶档的目标提示侧质量占比。config.json 不管抑制侧的参数，所以写死在这里：
# 0.95 是「尽量把提示侧拉满，但给生成侧留 5% 质量、别把它压成数值噪声」的折中，
# 在 p≈0.5 的行上解出 c≈19。**要改封顶档的干预强度就改这个数。**
SHARE_CEILING = 0.95
# 求解实际瞄的是哪个目标（PromptScale.target_kind）
TARGET_KINDS = ("pmf", "cap")
GROUP_BY = ("string", "position")
# 生成侧算子：flatten = 削向均匀（§2）；prompt_align = 逐串对齐提示侧占比（§2b）
GEN_OPS = ("flatten", "prompt_align")
# top-m 的名额按什么排：gen_mass = 生成侧质量（初版）；share_diff = 两侧占比差
# （生成侧占比 − 提示侧占比），见模块文档 §2c
TOPM_RANKS = ("gen_mass", "share_diff")
DEFAULT_TOPM_RANK = "gen_mass"
DIRECTIONS = ("both", "up", "down", "off")


# ------------------------------------------------------------------ 提示侧
def prompt_mean_fraction(row: np.ndarray, prompt_len: int) -> float:
    """一行的 pmf = (Σ提示/P) ÷ (Σ全行/T)。

    与 `solve_prompt_scale` 内部的 `ratio_now`、离线
    `compute_step_trajectory._compute_per_token_ratios` 的 `prompt_mean_fraction`
    **同式**——三处实现必须一致，任何一处都不许单方面改。
    行退化（提示侧或生成侧为空、整行质量 ≤ 0）时返回 NaN。
    """
    r = np.asarray(row, dtype=np.float64)
    T, P = r.size, int(prompt_len)
    total = float(r.sum())
    if P <= 0 or P >= T or total <= 0:
        return float("nan")
    return (float(r[:P].sum()) / P) / (total / T)


@dataclass
class PromptScale:
    """提示侧缩放常数 c 及其可解释的中间量。"""
    c: float
    ratio_now: float            # 当前的目标量（口径见 metric）
    ratio_target: float         # 参考曲线给出的目标值
    ratio_after: float          # 施加后的值（被夹住/不可达时 ≠ 目标）
    prompt_share: float         # 施加前提示侧质量占比
    prompt_share_after: float
    # 诊断用：提示首位（注意力汇）在全行里的质量占比。**只是测量**，求解与施加
    # 都不对汇做任何特殊处理（见模块文档 §1「不对注意力汇做特殊处理」）。
    sink_share: float
    status: str                 # ok | capped | no_action
    target_kind: str = "pmf"    # 实际瞄的目标：pmf（主档）| cap（封顶档）
    target_share: float = float("nan")   # 换算成「提示侧质量占比」的目标值

    @property
    def active(self) -> bool:
        return abs(self.c - 1.0) > 1e-9


def solve_prompt_scale(row: np.ndarray, prompt_len: int, ratio_target: float,
                       *, share_ceiling: float = SHARE_CEILING,
                       sink_positions: int = 1, cap: float = 0.0,
                       direction: str = "both") -> PromptScale:
    """提示侧要乘的常数 c —— 两档目标，闭式，永远有解（见模块文档 §1）。

    `row` 是当前查询的注意力行（长度 T，softmax 之后；内部会归一化，不要求和为 1）。

    **主档**：`ratio_target` 是参考曲线给的 pmf 目标。达到它所需要的提示侧质量
    占比是 `share_required = pmf* · P / T`——因为 pmf = 提示侧质量占比 × T/P。

      - `share_required < 1`：目标可达，就以它为准（`target_kind="pmf"`）。
      - `share_required ≥ 1`：这个 pmf 目标要求提示侧拿走全行 100% 以上的注意力
        质量，**没有任何注意力分布能实现**（不是夹子太紧，是目标本身不存在）。
        此时改瞄封顶占比 `share_ceiling`（默认 `SHARE_CEILING`=0.95，
        `target_kind="cap"`）：既然参考曲线要的是「提示侧越高越好、越高越好到
        越界」，就把提示侧推到这个封顶值为止，剩下 1 − `share_ceiling` 留给
        生成侧。**这一档的强度由这个常数决定，不再由参考曲线决定**，代价见
        模块文档 §1。

    两档合流之后是同一个闭式解：设归一后提示侧质量占比 p、生成侧 g = 1 − p，
    要求施加并重归一后的占比等于 `target_share`，则

        c = target_share / (1 − target_share) · g / p

    **提示侧整段同乘一个 c，注意力汇（提示第一个 token）不作任何特殊处理**——
    它和其它提示 token 一样乘 c。`sink_positions` 只用来算诊断字段 `sink_share`。

    `direction`: "up"/"down" 把 c 夹在 1 的一侧（强制「循环思考只抬、循环字符
    只压」）；"both" 纯按目标算；"off" 完全不动提示侧（消融臂）。
    `cap` 是 c 的量程夹子（夹在 [1/cap, cap]）；**cap ≤ 0 表示不设限制**（默认）。
    """
    assert direction in DIRECTIONS, f"direction must be one of {DIRECTIONS}"
    assert 0.0 < share_ceiling < 1.0, \
        f"share_ceiling 是提示侧质量占比的封顶值，必须在 (0,1)：{share_ceiling}"
    r = np.asarray(row, dtype=np.float64)
    T, P = r.size, int(prompt_len)
    G = T - P
    total = float(r.sum())
    nan = float("nan")
    if P <= 0 or G <= 0 or total <= 0:
        return PromptScale(1.0, nan, ratio_target, nan, nan, nan, nan,
                           "no_action")
    r = r / total
    p = float(r[:P].sum())
    g = 1.0 - p
    s = float(r[: min(sink_positions, P)].sum())      # 诊断用，不参与求解
    if p <= 0 or g <= 0:
        return PromptScale(1.0, nan, ratio_target, nan, p, p, s, "no_action")

    ratio_now = (p / P) * T                    # 当前 pmf
    # 达到 pmf 目标所需要的提示侧质量占比。≥ 1 就是「这个目标不存在」，
    # 改瞄封顶占比（模块文档 §1）。
    share_required = float(ratio_target) * P / T
    if share_required < 1.0:
        target_share, target_kind = share_required, "pmf"
    else:
        target_share, target_kind = float(share_ceiling), "cap"
    if not np.isfinite(target_share) or not 0.0 < target_share < 1.0:
        return PromptScale(1.0, ratio_now, ratio_target, ratio_now, p, p, s,
                           "no_action", target_kind, target_share)

    status = "ok"
    # 闭式：施加并重归一后 c·p/(c·p+g) = target_share
    c = (target_share / (1.0 - target_share)) * (g / p)
    if direction == "off":
        return PromptScale(1.0, ratio_now, ratio_target, ratio_now, p, p, s,
                           "no_action", target_kind, target_share)
    if direction == "up":
        c = max(c, 1.0)
    elif direction == "down":
        c = min(c, 1.0)
    if cap and np.isfinite(cap) and cap > 0:
        lo, hi = 1.0 / cap, cap
        if not (lo <= c <= hi):
            c = float(np.clip(c, lo, hi))
            status = "capped"

    p_after = c * p
    share_after = p_after / (p_after + g)
    # pmf 对重归一不免疫，必须用重归一后的占比算，这样 ratio_after 永远等于
    # 实际发生的事（被夹住 / 走回退目标时它自然不等于 ratio_target）。
    ratio_after = share_after * T / P
    return PromptScale(c=float(c), ratio_now=float(ratio_now),
                       ratio_target=float(ratio_target),
                       ratio_after=float(ratio_after), prompt_share=float(p),
                       prompt_share_after=float(share_after),
                       sink_share=float(s), status=status,
                       target_kind=target_kind, target_share=float(target_share))


# ------------------------------------------------------------------ 生成侧
@dataclass
class FlattenPlan:
    """生成侧削峰计划：top-m 位置乘 a，其余生成位置乘 b，总质量不变。"""
    a: float
    b: float
    tokens: list[str]           # 被选中的 token 串（group_by="string" 时）
    positions: list[int]        # 这些串在生成侧的全部位置（绝对下标）
    gen_mass: float             # 生成侧总质量（占全行）
    top_mass: float             # top-m 位置的质量（占生成侧）
    uniform_mass: float         # 完全均匀时这些位置应得的质量（占生成侧）
    top_mass_after: float
    n_gen: int
    status: str                 # ok | a_min | b_max | no_action
    contributions: dict[str, float] = field(default_factory=dict)
    # 被选中的 token 串的整数 id（运行时按它向量化地重建位置集合）
    top_ids: list[int] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return bool(self.positions) and abs(self.a - 1.0) > 1e-9


def token_ids_of(tokens: list[str]) -> np.ndarray:
    """token 串 → 稠密整数 id（按首次出现顺序）。

    在线路径每步都要按串聚合一次生成侧质量；直接用 Python 字典遍历是 O(T) 的
    解释器循环，在 16k 长度上会主导解码时间。运行时改为维护一份增量的 id 数组，
    聚合走 `np.bincount`，本函数是离线/自测用的一次性构造。"""
    vocab: dict[str, int] = {}
    out = np.empty(len(tokens), dtype=np.int64)
    for i, t in enumerate(tokens):
        out[i] = vocab.setdefault(t, len(vocab))
    return out


def solve_gen_flatten(row: np.ndarray, prompt_len: int,
                      tokens: list[str] | None = None, *,
                      token_ids: np.ndarray | None = None,
                      id_names: list[str] | None = None, top_m: int = 8, strength: float = 1.0,
                      a_min: float = 0.05, b_max: float = 5.0,
                      group_by: str = "string",
                      rank_by: str = DEFAULT_TOPM_RANK) -> FlattenPlan:
    """生成侧向均匀分布靠拢的两系数解（见模块文档 §2）。

    group_by="string"：按 token 串聚合，取质量最高的 top_m 个串，压它们的**全部**
    出现位置（循环的重复词因此被整串覆盖）；"position"：直接取质量最高的 top_m 个
    位置，不做聚合（对照用）。

    `rank_by` 决定 top-m 的名额按什么发（见模块文档 §2c）：`gen_mass` 按生成侧
    质量，`share_diff` 按「生成侧占比 − 提示侧占比」，且只收这个差为正的串——
    没有一个串在生成侧被过度使用时，本算子这一步不动（status="no_action"）。
    `group_by="position"` 下没有词串可比占比，`rank_by` 被忽略。

    `token_ids`（`token_ids_of` 的输出，或运行时增量维护的等价物）给出时走
    bincount 路径，全程 numpy；否则从 `tokens` 现算。两条路径结果相同。
    """
    assert group_by in GROUP_BY, f"group_by must be one of {GROUP_BY}"
    assert rank_by in TOPM_RANKS, f"rank_by must be one of {TOPM_RANKS}"
    r = np.asarray(row, dtype=np.float64)
    T, P = r.size, int(prompt_len)
    total = float(r.sum())
    empty = FlattenPlan(1.0, 1.0, [], [], 0.0, 0.0, 0.0, 0.0, max(T - P, 0),
                        "no_action")
    if P < 0 or T - P <= 1 or total <= 0 or top_m <= 0:
        return empty
    r = r / total
    gen = r[P:]
    n = int(gen.size)
    gmass = float(gen.sum())
    if gmass <= 0:
        return empty

    if group_by == "string":
        if token_ids is None:
            if tokens is None:
                raise ValueError("group_by='string' 需要 tokens 或 token_ids")
            token_ids = token_ids_of(tokens)
        ids = np.asarray(token_ids, dtype=np.int64)[:T]
        gen_ids = ids[P:]
        n_id = int(ids.max()) + 1 if ids.size else 0
        mass = np.bincount(gen_ids, weights=gen, minlength=n_id)
        if rank_by == "share_diff":
            # 名额发给「相对提示侧被过度使用」的串：绝对质量最高的往往是提示里
            # 本来就重的词，压它既不针对循环、又会白占名额（模块文档 §2c）。
            pmass = float(r[:P].sum())
            pshare = (np.bincount(ids[:P], weights=r[:P], minlength=n_id) / pmass
                      if pmass > 0 else np.zeros(max(n_id, 1)))
            score = mass / gmass - pshare[: mass.size]
            cand = np.nonzero((mass > 0) & (score > 0))[0]
            m = min(int(top_m), int(cand.size))
            top_ids = (cand[np.argsort(-score[cand], kind="stable")[:m]] if m
                       else np.empty(0, np.int64))
        else:
            m = min(top_m, int((mass > 0).sum()))
            top_ids = np.argpartition(-mass, m - 1)[:m] if m else np.empty(0, np.int64)
            top_ids = top_ids[np.argsort(-mass[top_ids], kind="stable")]
        positions = (np.nonzero(np.isin(gen_ids, top_ids))[0] + P).tolist()
        if id_names is not None:
            sel = [id_names[int(i)] for i in top_ids]
        elif tokens is not None:
            sel = [tokens[int(np.nonzero(ids == i)[0][0])] for i in top_ids]
        else:
            sel = [f"#{int(i)}" for i in top_ids]
        contrib = {t: round(float(mass[i]) / gmass, 6)
                   for t, i in zip(sel, top_ids)}
        top_id_list = [int(i) for i in top_ids]
    else:
        idx = (np.argsort(-gen, kind="stable")[:top_m] + P).tolist()
        sel, positions = [], sorted(int(j) for j in idx)
        contrib, top_id_list = {}, []

    n_s = len(positions)
    if n_s == 0 or n_s >= n:
        return FlattenPlan(1.0, 1.0, sel, positions, gmass, 1.0, 1.0, 1.0, n,
                           "no_action", contrib, top_id_list)
    M = float(r[np.asarray(positions, dtype=np.int64)].sum())
    U = gmass * n_s / n
    if M <= 0 or M <= U or gmass - M <= 0:
        # top-m 已经不高于均匀水平（例如 top-m 串出现次数极多），无峰可削
        return FlattenPlan(1.0, 1.0, sel, positions, gmass, M / gmass,
                           U / gmass, M / gmass, n, "no_action", contrib,
                           top_id_list)

    m_target = M + float(np.clip(strength, 0.0, 1.0)) * (U - M)
    a = m_target / M
    status = "ok"
    if a < a_min:
        a, status = a_min, "a_min"
    b = (gmass - a * M) / (gmass - M)
    if b_max and b > b_max:
        # 生成侧质量几乎全在 top-m 上：回填系数会爆炸。夹住 b，再按质量守恒
        # 反算 a（守恒精确成立，牺牲的是均匀化程度）。
        b, status = b_max, "b_max"
        a = max((gmass - b * (gmass - M)) / M, a_min)
    m_after = a * M
    return FlattenPlan(a=float(a), b=float(b), tokens=sel, positions=positions,
                       gen_mass=float(gmass), top_mass=float(M / gmass),
                       uniform_mass=float(U / gmass),
                       top_mass_after=float(m_after / gmass), n_gen=n,
                       status=status, contributions=contrib,
                       top_ids=top_id_list)


# ---------------------------------------------- 生成侧：逐串对齐提示侧占比
@dataclass
class PromptAlignPlan:
    """生成侧逐串对齐计划（见模块文档 §2b）。

    生成侧被分成三段，互不重叠：**被压段**（交集里生成侧占比高于提示侧的串，各有
    自己的系数 a_t < 1）、**不动段**（top-m 里其余的串，乘子恒为 1）、**回填段**
    （top-m 之外的位置，共用一个系数 b > 1），生成侧总质量精确不变。
    """
    a_by_id: dict[int, float]      # 被压段：token 串 id → 该串所有生成位置的系数
    b: float                       # 回填段（top-m 之外）的系数
    unchanged_ids: list[int]       # 不动段：top-m 里不被压的串（乘子恒为 1）
    tokens: list[str]              # 被压的词串（按生成侧质量降序）
    ids: list[int]
    keep_tokens: list[str]         # 不动段之一：提示里没出现过的串
    below_tokens: list[str]        # 不动段之二：提示里有、但生成侧占比已不高于它
    gen_mass: float                # 生成侧总质量（占全行）
    prompt_mass: float             # 提示侧总质量（占全行）
    sel_mass: float                # 被压段占生成侧的比例（施加前）
    unchanged_mass: float          # 不动段占生成侧的比例
    rest_mass: float               # 回填段占生成侧的比例
    target_mass: float             # 对齐目标：被压的这些串在提示侧的占比之和
    sel_mass_after: float
    n_gen: int                     # 生成位置总数
    n_sel_pos: int                 # 被压段覆盖的生成位置数
    n_unchanged_pos: int           # 不动段的位置数
    n_top: int                     # top-m 实际取到几个串
    status: str                    # ok | no_target | no_action
    gen_share: dict[str, float] = field(default_factory=dict)
    prompt_share: dict[str, float] = field(default_factory=dict)
    a_by_token: dict[str, float] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return bool(self.a_by_id) and any(abs(a - 1.0) > 1e-9
                                          for a in self.a_by_id.values())


def solve_gen_prompt_align(row: np.ndarray, prompt_len: int,
                           tokens: list[str] | None = None, *,
                           token_ids: np.ndarray | None = None,
                           id_names: list[str] | None = None,
                           top_m: int = 8, strength: float = 1.0,
                           rank_by: str = DEFAULT_TOPM_RANK
                           ) -> PromptAlignPlan:
    """把生成侧占比**高于**提示侧的 top-m 词串压回到提示侧占比（见模块文档 §2b）。

    候选是「生成侧 top-m 词串」∩「提示里出现过的词串」，其中**只压不抬**：

        a_t = 提示占比(t) / 生成占比(t)      仅当该比值 < 1（λ=strength 时按比例走一半）
        a_t = 1                              该串在生成侧占比已不高于提示侧 → 不动

    top-m 里提示中没出现过的串同样不动。只有 top-m 之外的位置乘回填系数 b，使
    生成侧总质量精确不变；因为只压不抬，b ≥ 1 恒成立。系数**不设量程限制**。

    `rank_by` 决定 top-m 的名额按什么发（见模块文档 §2c）：`gen_mass` 先按生成侧
    质量取 m 个串、再取交集（名额会被压不动的串占掉）；`share_diff` 直接把名额
    发给「提示里出现过、且生成侧占比高于提示侧」的串，按两侧占比差从大到小排，
    于是 m 个名额全部落在真正会被压的串上、**不动段为空**。
    """
    assert rank_by in TOPM_RANKS, f"rank_by must be one of {TOPM_RANKS}"
    r = np.asarray(row, dtype=np.float64)
    T, P = r.size, int(prompt_len)
    G = T - P
    total = float(r.sum())
    empty = PromptAlignPlan({}, 1.0, [], [], [], [], [], 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, max(G, 0), 0, 0, 0, "no_action")
    if P <= 0 or G <= 1 or total <= 0 or top_m <= 0:
        return empty
    r = r / total
    gen = r[P:]
    gmass, pmass = float(gen.sum()), float(r[:P].sum())
    if gmass <= 0 or pmass <= 0:
        return empty

    if token_ids is None:
        if tokens is None:
            raise ValueError("需要 tokens 或 token_ids")
        token_ids = token_ids_of(tokens)
    ids = np.asarray(token_ids, dtype=np.int64)[:T]
    n_id = int(ids.max()) + 1
    gm = np.bincount(ids[P:], weights=gen, minlength=n_id)
    pm = np.bincount(ids[:P], weights=r[:P], minlength=n_id)

    if rank_by == "share_diff":
        # 只压不抬 ⇒ 可压的只有「提示里出现过、且生成侧占比更高」的串。名额直接
        # 按占比差发给它们，不再先按生成侧质量截断（模块文档 §2c）。
        diff = gm / gmass - pm / pmass
        cand = np.nonzero((gm > 0) & (pm > 0) & (diff > 0))[0]
        m = min(int(top_m), int(cand.size))
        top = (cand[np.argsort(-diff[cand], kind="stable")[:m]] if m
               else np.empty(0, dtype=np.int64))
    else:
        m = min(int(top_m), int((gm > 0).sum()))
        if m <= 0:
            return empty
        top = np.argpartition(-gm, m - 1)[:m]
        top = top[np.argsort(-gm[top], kind="stable")]
    names = (lambda i: id_names[int(i)]) if id_names is not None else \
            ((lambda i: tokens[int(np.nonzero(ids == i)[0][0])])
             if tokens is not None else (lambda i: f"#{int(i)}"))

    # 三分：被压（提示里有 + 生成侧占比更高）/ 不动（其余 top-m）/ 回填（top-m 之外）
    in_prompt = pm[top] > 0
    ratio = np.ones(top.size)
    ratio[in_prompt] = ((pm[top[in_prompt]] / pmass)
                        / (gm[top[in_prompt]] / gmass))
    down = in_prompt & (ratio < 1.0)
    sel, unchanged = top[down], top[~down]
    keep_names = [names(i) for i in top[~in_prompt]]
    below_names = [names(i) for i in top[in_prompt & ~(ratio < 1.0)]]
    unchanged_mass = float(gm[unchanged].sum())
    n_unchanged_pos = (int(np.isin(ids[P:], unchanged).sum())
                       if unchanged.size else 0)
    sel_names = [names(i) for i in sel]
    share_g = gm[sel] / gmass
    share_p = pm[sel] / pmass
    M = float(gm[sel].sum())                 # 被压段质量
    # 回填段质量（top-m 之外）。三段相减到 0 时会留下浮点残渣，按相对量判空。
    rest = gmass - M - unchanged_mass
    common = dict(
        unchanged_ids=[int(i) for i in unchanged], tokens=sel_names,
        ids=[int(i) for i in sel], keep_tokens=keep_names,
        below_tokens=below_names, gen_mass=gmass, prompt_mass=pmass,
        sel_mass=float(M / gmass), unchanged_mass=float(unchanged_mass / gmass),
        rest_mass=float(max(rest, 0.0) / gmass),
        target_mass=float(share_p.sum()), n_gen=G,
        n_sel_pos=int(np.isin(ids[P:], sel).sum()) if sel.size else 0,
        n_unchanged_pos=n_unchanged_pos, n_top=m,
        gen_share={t: round(float(v), 6) for t, v in zip(sel_names, share_g)},
        prompt_share={t: round(float(v), 6) for t, v in zip(sel_names, share_p)})
    if sel.size == 0:
        # top-m 里没有一个串的生成侧占比高于它的提示侧占比 —— 没什么可压的
        return PromptAlignPlan(a_by_id={}, b=1.0, sel_mass_after=0.0,
                               status="no_target", a_by_token={}, **common)
    if rest <= gmass * 1e-12:                # top-m 覆盖了整个生成侧，无处回填
        return PromptAlignPlan(a_by_id={}, b=1.0, sel_mass_after=float(M / gmass),
                               status="no_action", a_by_token={}, **common)

    lam = float(np.clip(strength, 0.0, 1.0))
    a = 1.0 + lam * (share_p / share_g - 1.0)          # 恒 ≤ 1
    M_after = float((a * gm[sel]).sum())
    b = (rest + M - M_after) / rest                    # 恒 ≥ 1（只压不抬）
    return PromptAlignPlan(
        a_by_id={int(i): float(v) for i, v in zip(sel, a)}, b=float(b),
        sel_mass_after=float(M_after / gmass), status="ok",
        a_by_token={t: round(float(v), 6) for t, v in zip(sel_names, a)},
        **common)


# ------------------------------------------------------------------ 组合乘子
def build_row_multiplier(length: int, prompt_len: int, *, c: float = 1.0,
                         flatten: FlattenPlan | None = None,
                         align: PromptAlignPlan | None = None,
                         token_ids: np.ndarray | None = None,
                         dtype=np.float32) -> np.ndarray:
    """长度 length 的乘子：提示侧 c、生成侧重点位 a、其余生成位 b。

    生成侧算子二选一（都为 None 表示这一层只做提示侧缩放，v5 里绝大多数层就是
    这样）：`flatten` 是削向均匀的单系数计划（§2），`align` 是逐串对齐提示侧占比
    的多系数计划（§2b）——后者按 token 串 id 落位，所以必须同时传 `token_ids`
    （长度 ≥ length 的全序列 id 数组）。
    length 可以大于计划生成时的行长（滞后一步时行又长了一个 token）——生成侧的
    新位置默认取 b，与其它生成位置同待遇；若该位置的 token 串在计划里，`align`
    路径按当前 id 数组自动覆盖到，`flatten` 路径由调用方传入重建的 positions。
    """
    P = int(prompt_len)
    L = int(length)
    m = np.ones(L, dtype=dtype)
    if abs(c - 1.0) > 1e-9 and P > 0:
        # 提示侧整段（含注意力汇）同乘 c——汇不作特殊处理，见 §1
        m[:P] = c
    if flatten is not None and flatten.active:
        if abs(flatten.b - 1.0) > 1e-9:
            m[P:] = flatten.b
        pos = [j for j in flatten.positions if j < L]
        if pos:
            m[np.asarray(pos, dtype=np.int64)] = flatten.a
    if align is not None and align.active:
        if token_ids is None:
            raise ValueError("align 计划按 token 串 id 落位，需要 token_ids")
        gen_ids = np.asarray(token_ids, dtype=np.int64)[P:L]
        # 三段互不重叠：查表一次落位（每个 token 串 id 一个系数），比按串各扫一遍
        # 生成侧快一个数量级——16k 长度、32 层、每步都要建，这里是热点。
        # 默认回填 b，不动段（top-m 里提示中没有的串、以及占比已不高于提示侧的
        # 串）填 1，被压段填各自的 a_t。
        n_id = int(gen_ids.max()) + 1 if gen_ids.size else 0
        lut = np.full(max(n_id, 1), align.b, dtype=dtype)
        keep = [t for t in align.unchanged_ids if t < n_id]
        if keep:
            lut[np.asarray(keep, dtype=np.int64)] = 1.0
        hit = [(t, a) for t, a in align.a_by_id.items() if t < n_id]
        if hit:
            lut[np.asarray([t for t, _ in hit], dtype=np.int64)] = \
                np.asarray([a for _, a in hit], dtype=dtype)
        if gen_ids.size:
            m[P:] = lut[gen_ids]
    return m


def apply_multiplier(row: np.ndarray, mult: np.ndarray,
                     renormalise: bool = True) -> np.ndarray:
    """row ∘ mult（mult 沿前置的 head/batch 维广播），可选按最后一维重归一。"""
    out = row * mult
    if renormalise:
        out = out / out.sum(axis=-1, keepdims=True)
    return out


# --------------------------------------------------------------------------- #
def _selftest() -> int:
    rng = np.random.default_rng(0)
    P, G = 6, 10
    tokens = ["p%d" % i for i in range(P)] + ["x", "&", "&", "y", "&", "&",
                                              "z", "&", "&", "&"]
    row = np.concatenate([np.array([0.40, 0.03, 0.02, 0.02, 0.02, 0.01]),
                          np.array([0.01, 0.09, 0.09, 0.01, 0.09, 0.08,
                                    0.01, 0.04, 0.04, 0.04])])
    row = row / row.sum()
    P_len, T = P, P + G

    # --- 提示侧：两档目标，都精确命中，且永远有解 --------------------------
    def pmf_of(r):
        """按主口径读一行（r 需已归一）。"""
        return (r[:P].sum() / r.sum() / P) * T

    def share_of(r):
        return r[:P].sum() / r.sum()

    pmf, share = pmf_of(row), share_of(row)
    BOUND = T / P                       # pmf 的落点上界 = 提示侧占比 1 时的值
    CEIL = 0.80                         # 封顶占比（自测里特意不取默认值）
    cases = [("抬（主档）", pmf * 1.5, "pmf"),
             ("压（主档）", pmf * 0.4, "pmf"),
             ("目标要求占比 ≥ 1 → 封顶", BOUND * 3.0, "cap"),
             ("目标恰好要求占比 = 1 → 封顶", BOUND, "cap")]
    for name, tgt, kind in cases:
        ps = solve_prompt_scale(row, P, tgt, share_ceiling=CEIL, cap=0.0)
        assert ps.target_kind == kind and ps.status == "ok", (name, ps)
        m = build_row_multiplier(T, P, c=ps.c, dtype=np.float64)
        new_row = apply_multiplier(row, m)
        # 报告的落点必须永远等于实际发生的事
        assert abs(pmf_of(new_row) - ps.ratio_after) < 1e-9, (name, ps)
        assert abs(share_of(new_row) - ps.prompt_share_after) < 1e-9, ps
        # 精确命中它实际瞄的那个目标
        assert abs(share_of(new_row) - ps.target_share) < 1e-9, (name, ps)
        if kind == "pmf":
            assert abs(pmf_of(new_row) - tgt) < 1e-9, (name, tgt)
        else:
            assert abs(share_of(new_row) - CEIL) < 1e-9, (name, ps)
        # 落点永远在 pmf 的上界之内——那是这个量的定义决定的
        assert pmf_of(new_row) < BOUND + 1e-12, (name, pmf_of(new_row))
        # 汇不作特殊处理：提示侧整段（含第 0 位）同一个乘子
        assert np.allclose(m[:P], ps.c) and np.allclose(m[P:], 1.0), name
    print("提示侧缩放: 主档精确命中 pmf、目标要求占比≥1 时改瞄封顶占比并精确"
          "命中、提示侧整段同乘（汇不特殊处理）  OK")
    # 不传 share_ceiling 时走模块默认值，越界目标照样有解
    ps_def = solve_prompt_scale(row, P, BOUND * 3.0, cap=0.0)
    assert ps_def.target_kind == "cap" and ps_def.status == "ok", ps_def
    assert abs(ps_def.target_share - SHARE_CEILING) < 1e-12, ps_def
    assert abs(share_of(apply_multiplier(row, build_row_multiplier(
        T, P, c=ps_def.c, dtype=np.float64))) - SHARE_CEILING) < 1e-9, ps_def
    # 封顶值本身必须是个合法占比
    for bad in (0.0, 1.0, 1.5, -0.1):
        try:
            solve_prompt_scale(row, P, BOUND * 3.0, share_ceiling=bad)
            raise AssertionError(f"share_ceiling={bad} 应当报错")
        except AssertionError as e:
            assert "share_ceiling" in str(e), e
    print(f"pmf 上界 T/P={BOUND:.3f}：越界目标改瞄封顶占比 "
          f"（默认 {SHARE_CEILING}，c={ps_def.c:.3g}），封顶值出界时显式报错  OK")
    print(f"pmf 与占比: pmf={pmf:.4f} = 占比 {share:.4f} × T/P {T / P:.4f}  OK")

    # --- 生成侧：总质量守恒 + 峰被削 ----------------------------------------
    fp = solve_gen_flatten(row, P, tokens, top_m=1, strength=1.0)
    assert "&" in fp.tokens and len(fp.positions) == 7, fp
    assert fp.a < 1.0 < fp.b, fp
    m = build_row_multiplier(T, P, c=1.0, flatten=fp, dtype=np.float64)
    new = row * m
    assert abs(new[P:].sum() - row[P:].sum()) < 1e-12, "生成侧总质量应精确不变"
    assert abs(new[:P].sum() - row[:P].sum()) < 1e-12, "提示侧不应被生成侧算子改动"
    top_after = new[fp.positions].sum() / new[P:].sum()
    assert abs(top_after - fp.uniform_mass) < 1e-9, (top_after, fp.uniform_mass)
    print(f"生成侧削峰: top-1 串 '&' 占比 {fp.top_mass:.3f} → {top_after:.3f} "
          f"(均匀水平 {fp.uniform_mass:.3f}), a={fp.a:.3f} b={fp.b:.3f}, "
          f"总质量守恒  OK")

    # --- 生成侧对齐提示侧：只压不抬，逐串命中提示占比，总质量守恒 -----------
    # 模块文档 §2b 的例子：top-3 = ['c','a','b']，提示里只有 'a'、'b'；
    # 'a' 在生成侧占 25%、提示侧占 10% → 每个 'a' 位置乘 0.4。
    # 'b' 在生成侧 20% < 提示侧 30% → 不动；'c' 提示里没有 → 不动；
    # 只有 top-3 之外的 'd' 走回填。
    ptok = ["a", "b", "q", "q"]
    gtok = ["a", "a", "b", "c", "c", "d", "d", "d"]
    pm_ = np.array([0.10, 0.30, 0.25, 0.35]) * 0.4           # 提示侧占比 × 提示质量
    gm_ = np.array([0.15, 0.10, 0.20, 0.20, 0.15, 0.08, 0.06, 0.06]) * 0.6
    assert abs(gm_[:2].sum() / 0.6 - 0.25) < 1e-12           # 'a' 生成侧占 25%
    row2 = np.concatenate([pm_, gm_])
    tok2, P2 = ptok + gtok, len(ptok)
    ids2 = token_ids_of(tok2)
    pos_of = lambda t: [j for j in range(P2, len(tok2)) if tok2[j] == t]  # noqa: E731
    ap = solve_gen_prompt_align(row2, P2, tok2, top_m=3)
    assert ap.status == "ok" and ap.tokens == ["a"], ap.tokens
    assert ap.keep_tokens == ["c"] and ap.below_tokens == ["b"], ap
    assert abs(ap.a_by_token["a"] - 0.4) < 1e-9, ap.a_by_token
    m4 = build_row_multiplier(len(tok2), P2, align=ap, token_ids=ids2,
                              dtype=np.float64)
    new4 = row2 * m4
    assert abs(new4[P2:].sum() - row2[P2:].sum()) < 1e-12, "生成侧总质量应精确不变"
    assert abs(new4[:P2].sum() - row2[:P2].sum()) < 1e-12, "提示侧不应被生成侧算子改动"
    got = new4[pos_of("a")].sum() / new4[P2:].sum()          # 命中提示侧占比
    assert abs(got - ap.prompt_share["a"]) < 1e-9, (got, ap.prompt_share)
    # 不动段（'b' 占比已低于提示侧、'c' 提示里没有）逐位不变；只有 'd' 回填
    for t in ("b", "c"):
        assert np.allclose(m4[pos_of(t)], 1.0), (t, m4[pos_of(t)])
        assert abs(new4[pos_of(t)].sum() - row2[pos_of(t)].sum()) < 1e-12
    assert np.allclose(m4[pos_of("d")], ap.b) and ap.b > 1.0
    print(f"生成侧对齐提示侧: 'a' 占比 {ap.gen_share['a']:.3f} → "
          f"{ap.prompt_share['a']:.3f}(=提示侧) ×{ap.a_by_token['a']}, "
          f"'b'(已低于提示侧)/'c'(提示里没有) 不动, 其余 b={ap.b:.4f}, "
          f"总质量守恒  OK")

    # 系数无量程限制：需要多小就多小（这里 'a' 要 0.02 也照给）
    pm3 = np.array([0.005, 0.30, 0.30, 0.395]) * 0.4
    row3 = np.concatenate([pm3, gm_])
    a3 = solve_gen_prompt_align(row3, P2, tok2, top_m=3)
    assert abs(a3.a_by_token["a"] - 0.005 / 0.25) < 1e-9, a3.a_by_token
    m3b = build_row_multiplier(len(tok2), P2, align=a3, token_ids=ids2,
                               dtype=np.float64)
    assert abs((row3 * m3b)[P2:].sum() - row3[P2:].sum()) < 1e-12

    # 没有一个 top-m 串的生成侧占比高于提示侧 → 无事可做；λ 是压的完成度
    pm5 = np.array([0.60, 0.35, 0.03, 0.02]) * 0.4
    a5 = solve_gen_prompt_align(np.concatenate([pm5, gm_]), P2, tok2, top_m=3)
    assert a5.status == "no_target" and not a5.active, a5
    ah = solve_gen_prompt_align(row2, P2, tok2, top_m=3, strength=0.5)
    assert abs(ah.a_by_token["a"] - (1 + 0.5 * (0.4 - 1))) < 1e-9, ah.a_by_token
    assert not solve_gen_prompt_align(row2, P2, tok2, top_m=3,
                                      strength=0.0).active
    # 提示里一个 top-m 词串都没有 → 同样无事可做
    assert solve_gen_prompt_align(row2, P2, ["z"] * P2 + gtok,
                                  top_m=3).status == "no_target"
    # top-m 覆盖了整个生成侧 ⇒ 没有回填段
    assert solve_gen_prompt_align(row2, P2, tok2, top_m=99).status == "no_action"
    print("对齐算子边界: 无可压对象 / 提示无交集 / 无回填段 / λ 完成度  OK")

    # --- top-m 按两侧占比差排（§2c）：名额不再被"压不动的串"占掉 -------------
    # 同一行数据：按生成侧质量排，top-3 是 ['c','a','b']，其中只有 'a' 真被压
    # （'c' 提示里没有、'b' 占比已低于提示侧）；按占比差排，候选只剩 'a'
    # （'b' 差为负、'c'/'d' 提示里没有 → 压不动，不发名额），不动段因此为空。
    ad = solve_gen_prompt_align(row2, P2, tok2, top_m=3, rank_by="share_diff")
    assert ad.status == "ok" and ad.tokens == ["a"], ad.tokens
    assert ad.keep_tokens == [] and ad.below_tokens == [], ad
    assert ad.n_top == len(ad.a_by_id) == 1 and ad.unchanged_mass == 0.0, ad
    assert abs(ad.a_by_token["a"] - ap.a_by_token["a"]) < 1e-12   # 系数不受排序影响
    md = build_row_multiplier(len(tok2), P2, align=ad, token_ids=ids2,
                              dtype=np.float64)
    newd = row2 * md
    assert abs(newd[P2:].sum() - row2[P2:].sum()) < 1e-12, "生成侧总质量应精确不变"
    # 'b'/'c' 从"不动段"挪进了回填段：同一个 b，逐位一致
    for t in ("b", "c", "d"):
        assert np.allclose(md[pos_of(t)], ad.b), (t, md[pos_of(t)])
    assert ad.b > 1.0 and ad.rest_mass > ap.rest_mass, (ad.rest_mass, ap.rest_mass)
    print(f"top-m 按占比差排(对齐算子): 名额 3 个全给可压的串 "
          f"(不动段 {ap.unchanged_mass:.3f} → 0)，回填段 "
          f"{ap.rest_mass:.3f} → {ad.rest_mass:.3f}，总质量守恒  OK")

    # 削峰算子同理：'k' 生成侧质量最高，但它在提示里占比更高（不是循环词），
    # 按占比差排就轮不到它，名额给了提示里没有的 'x'。
    tok3f = ["k", "k", "a"] + ["k", "k", "x", "y", "y"]
    row3f = np.array([0.30, 0.30, 0.10, 0.10, 0.08, 0.09, 0.02, 0.01])
    P3 = 3
    fk = solve_gen_flatten(row3f, P3, tok3f, top_m=1)
    fx = solve_gen_flatten(row3f, P3, tok3f, top_m=1, rank_by="share_diff")
    assert fk.tokens == ["k"] and fx.tokens == ["x"], (fk.tokens, fx.tokens)
    assert fx.a < 1.0 < fx.b, fx
    mx = build_row_multiplier(len(tok3f), P3, flatten=fx, dtype=np.float64)
    assert abs((row3f * mx)[P3:].sum() - row3f[P3:].sum()) < 1e-12
    # 一个串都不比提示侧占比高时，这一步不动
    assert solve_gen_flatten(row3f, P3, ["x", "y", "k"] + ["k"] * 5,
                             top_m=1, rank_by="share_diff").status == "no_action"
    print(f"top-m 按占比差排(削峰算子): 质量最高的 'k'(提示里更重) 让位给 "
          f"'x', a={fx.a:.3f} b={fx.b:.3f}，总质量守恒  OK")

    # --- 组合：生成侧守恒 ⇒ 两个算子互不干扰 --------------------------------
    ps = solve_prompt_scale(row, P, pmf * 1.5, cap=0.0)
    only = pmf_of(apply_multiplier(row, build_row_multiplier(
        T, P, c=ps.c, dtype=np.float64)))
    both = pmf_of(apply_multiplier(row, build_row_multiplier(
        T, P, c=ps.c, flatten=fp, dtype=np.float64)))
    assert abs(both - only) < 1e-9 and abs(both - ps.ratio_after) < 1e-9
    # 对齐算子同样守恒 ⇒ 同样不干扰提示侧（换一行数据走一遍）
    T2 = len(tok2)
    pmf2 = lambda r: (r[:P2].sum() / r.sum() / P2) * T2      # noqa: E731
    ps2 = solve_prompt_scale(row2, P2, pmf2(row2) * 1.2, cap=0.0)
    m6 = build_row_multiplier(T2, P2, c=ps2.c, align=ap, token_ids=ids2,
                              dtype=np.float64)
    only6 = pmf2(apply_multiplier(row2, build_row_multiplier(
        T2, P2, c=ps2.c, dtype=np.float64)))
    assert abs(pmf2(apply_multiplier(row2, m6)) - only6) < 1e-9
    print("组合乘子: 削峰/对齐都不改动提示侧的落点  OK")

    # --- 夹取与退化 ---------------------------------------------------------
    # 主口径下一个可达但很吃力的目标（要求占比 0.99）→ c 很大 → 被夹住
    ps = solve_prompt_scale(row, P, 0.99 * T / P, cap=20.0)
    assert ps.target_kind == "pmf" and ps.status == "capped" and ps.c == 20.0, ps
    # 封顶档同样受夹子约束；不设夹子时 c 有限、落点精确命中封顶占比
    ps = solve_prompt_scale(row, P, T / P * 100, share_ceiling=0.9, cap=0.0)
    c_unlimited = ps.c
    assert ps.status == "ok" and ps.target_kind == "cap", ps
    assert abs(ps.prompt_share_after - 0.9) < 1e-12, ps
    # 可达目标下，有没有夹子解出来的 c 必须一样
    a1 = solve_prompt_scale(row, P, pmf * 1.05, cap=20.0)
    a2 = solve_prompt_scale(row, P, pmf * 1.05, cap=0.0)
    assert a1.status == a2.status == "ok" and abs(a1.c - a2.c) < 1e-12
    ps = solve_prompt_scale(row, P, pmf * 1.2, direction="down")
    assert ps.c == 1.0 and not ps.active
    flat = np.concatenate([np.full(P, 0.5 / P), np.full(G, 0.5 / G)])
    fp2 = solve_gen_flatten(flat, P, tokens, top_m=1)
    assert fp2.status == "no_action" and not fp2.active
    # top-m 覆盖了生成侧全部 token 串 ⇒ 没有「其余位置」可回填，也退化为不动
    assert solve_gen_flatten(row, P, tokens, top_m=99).status == "no_action"
    fp3 = solve_gen_flatten(row, P, tokens, top_m=2, b_max=1.05)
    m3 = build_row_multiplier(T, P, flatten=fp3, dtype=np.float64)
    assert fp3.status == "b_max"
    assert abs((row * m3)[P:].sum() - row[P:].sum()) < 1e-12
    print(f"边界: cap / 封顶档精确命中占比目标（c={c_unlimited:.3g}）/ "
          f"direction / 无峰可削 / b_max 后仍守恒  OK")

    # --- 广播到多头 ---------------------------------------------------------
    heads = np.repeat(row[None, None, :], 4, axis=0) * rng.uniform(0.9, 1.1, (4, 1, T))
    heads = heads / heads.sum(-1, keepdims=True)
    out = apply_multiplier(heads, build_row_multiplier(T, P, c=2.0, flatten=fp, dtype=np.float64))
    assert out.shape == heads.shape and np.allclose(out.sum(-1), 1.0)
    print("多头广播  OK")
    print("SELFTEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
