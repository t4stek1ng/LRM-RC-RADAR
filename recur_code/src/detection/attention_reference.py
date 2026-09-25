"""正常思考的「提示侧注意力强度」参考曲线 —— v5 抑制方案的目标值来源。

抑制的提示侧那一半要回答的问题是：**这一步的提示侧注意力应该有多强**。答案不是
一个手调常数，而是**同一模型的正常思考轨迹在同一生成位置上的实测水平**。

## 一条曲线，不再按类别分（2026-09-13 改）

此前两类良性轨迹各拟合一条，按检测类别路由（循环思考→有效反思，循环字符→简洁
推理）。现在**两类混在一起拟合成一条** `benign` 曲线，两个攻击类共用它
（`REFERENCE_OF` 仍在，只是两边指向同一条）。`EXPECTED_DIRECTION` 保留——
`--prompt-direction expected` 仍然按类别夹方向，改掉的只是**目标值**的来源。

合并**必须带提示长度项**，否则是错的。三个模型上的实测（每条轨迹 stride=4 取点、
按轨迹分组留一交叉验证，误差统一折算到求解器真正使用的量 log 提示侧质量占比）：

| 候选 | 参数 | LOTO 误差（log 占比，Llama-8B / GLM / QwQ / Qwen3.6） | 类别残差偏置 |
|---|---|---|---|
| 两类分开 · 只按位置（旧） | 6 | 0.178 / 0.248 / 0.235 / 0.429 | —（各自拟合） |
| 合并 · 只按位置 | 3 | 0.263 / 0.391 / 0.441 / 0.497 | **±0.19 ~ ±0.41** |
| **合并 · 位置+提示长度** | 4 | **0.107 / 0.120 / 0.100 / 0.410** | ±0.001 ~ ±0.014 |
| 合并 · 位置+提示长度+类别哑元 | 5 | 0.107 / 0.121 / 0.099 / 0.410 | 0 |

读法：**两类良性的差别几乎全部是提示长度**（简洁推理提示长中位 52–69，有效反思
170–198，pmf 里带着 1/P 这个因子，两类的 pmf 之比 0.40–0.47 ≈ 长度之比的倒数）。
控制住长度之后再加类别哑元，系数只有 +0.009 / −0.007 / −0.056 / −0.107（即
0.90–1.01 倍），LOTO 一点没动——**类别本身不带额外信息**，合并因此不是妥协而是
去掉一个冗余自由度，参数从 6 个减到 4 个，泛化误差反而降到原来的一半。

> **这条结论的边界**：本批数据里类别与提示长度**几乎共线**（简洁推理 P∈[30,121]，
> 有效反思 P∈[117,446]，只在 117–121 上有一点重叠）。所以严格说，能断言的是
> 「**给定长度项之后**类别哑元无增益」，不能断言「类别与长度无关」。要分开这两件
> 事，需要一批长度同分布、类别不同的良性轨迹。

## 只拟合一档：pmf

离线三阶段流水线已经为每个生成位置算好四个口径（`compute_step_trajectory.py`），
本模块只用 `prompt_mean_fraction`（pmf）= (Σ提示/提示长) ÷ (Σ全行/总长)。理由是
**检测与抑制瞄同一个量**：`detector.py` 给分类器的第二个特征就是这条序列在
[0, i] 上的累计斜率。

**曲线管不到的那一段由封顶常数接手**：pmf = 提示侧质量占比 × T/P，所以想达到目标
pmf* 就等于要求占比达到 pmf*·P/T。这个数**可能 ≥ 1**——那是要求提示侧拿走全行
100% 以上的注意力，没有任何注意力分布能实现（推导见 `row_reshape` §1）。根源是
pmf 里带着 T/P 这个长度因子：曲线拟合在良性轨迹上（提示长 75–948），而攻击提示长
1901–2492。这种情况下 `solve_prompt_scale` 不再查任何曲线，直接把提示侧推到封顶
占比 `SHARE_CEILING`（默认 0.95）。**攻击提示普遍很长，所以这条路是常态**——本模块
的曲线实际主要作用在短提示上。

> 2026-09-11 之前，越界那一档瞄的是同一条参考轨迹的**提示侧质量占比**
> （`prompt_fraction`，不含长度因子、恒在 (0,1) 内），本模块为此多拟合一份
> `share_fits`。它的两条代价是换掉它的原因：拟合质量差（Llama-8B 上 R² 只有 0.59
> 有效反思 / 0.77 简洁推理，而 pmf 加长度项是 0.987 / 0.979），且**方向随施加层
> 翻转**（按 32 层平均读，循环思考的占比 0.666 反而高于有效反思 0.612，对齐会把
> 提示侧压下去；按第 15 层自己的行读才是抬）。旧曲线文件里残留的 `share_fits`
> 字段会被 `load` 忽略，不必重建。方向仍可用 `--prompt-direction expected` 强制。

## 但必须知道的一条：循环思考的「偏低」几乎全部来自提示长度

ρ 是**每 token** 强度比，而熵下降攻击的提示天生很长（Llama-8B 上循环思考提示中位
2184 token，有效反思只有 178）。把提示长度放进回归

        log ρ = a + b·log(生成位置) + d·log(提示长度)

在三个模型上都得到 d ≈ −0.96（R² ≈ 0.96），即提示每 token 强度几乎与提示长度成
反比；**扣掉长度后，循环思考的 ρ 反而比有效反思曲线高 11%–18%，而不是低 8 倍**。
循环字符则不受影响（它的提示长度与简洁推理同量级），扣掉长度后仍高 40%–70%，是
真实信号。

因此本模块提供两种口径，由 `--normalize` 选择：

  - `position`（默认，方案原文口径）：只按生成位置拟合，长度效应留在曲线里。
    对循环思考给出 c ≈ 8 的强干预——**它纠正的主要是「提示长」而不是「在循环」**，
    但这正是方案设计的「抬到正常思考水平」的字面含义。
  - `position_prompt_len`（对照口径）：把提示长度作为协变量一并拟合，目标值随实际
    提示长度伸缩。对循环思考给出 c ≈ 0.9（几乎不动），可用来隔离「提示长度」这个
    变量，判断强干预的收益到底来自哪里。

两种口径都写进同一个 JSON，运行时按 `--ref-normalize` 取用，因此不需要重建。

## 拟合形式

对每类正常思考轨迹，在对数域做加权最小二乘

        log ρ ≈ Σ_i coef_i · (log 生成位置)^i   [+ d · log 提示长度]

默认二次（`--degree 2`）。**直线（`--degree 1`）试过，不够**：LOTO 误差 deg=1 → 2
降 53%（Llama-8B 0.225→0.106）/ 32%（GLM）/ 31%（QwQ）/ 3%（Qwen3.6），而 deg=2 → 3
只再降 2%–5%，所以二次是拐点。

直线差在**它没法跟上加速**。log pmf 对 log 生成位置的斜率在 Llama-8B 上从 0.42
（生成位置 64）涨到 0.94（4096），直线只能取一个常数 0.513，于是残差按位置呈 S 形
系统偏离（正 = 目标偏高、会多抬）：

| 生成位置 | 0–32 | 64–512 | ≥1024 |
|---|---|---|---|
| deg=1 | −40% | +12% ~ +26% | −7% ~ −18% |
| deg=2 | −28% ~ −15% | −9% ~ −4% | −4% ~ +3% |

阶数不进默认文件名，但 `--degree` ≠ 2 时会写成 `..._deg<N>.json`，免得试阶数
覆盖掉部署在用的那份。

权重两级归一：**每条轨迹**归一到相同总权重（否则 10 万点的长轨迹会淹没短轨迹），
**每个良性类别**再归一到相同总权重（否则「混合曲线」会变成「样本多的那一类的
曲线」；本批 20+20 时这一级是恒等变换，但类别数一旦不平衡就必须有它）。

残差分位数一并保存，运行时可用 `--ref-quantile` 取目标带的下沿而不是中位（更温和
的干预）。生成位置与提示长度超出拟合范围时把自变量夹在观测范围内，避免二次项外推
发散——**攻击提示（1901–3908）比任何良性轨迹（30–446）都长 4–9 倍，所以长度项在
攻击上恒被夹在上界**，曲线实际给出的是「P = 拟合上界」那一档的目标。

拟合质量在模型之间差很多（合并 + 长度项 + 二次，加权 R² / 残差 sd）：
DeepSeek-R1-Distill-Llama-8B 0.966 / 0.143，GLM-4.7-Flash 0.957 / 0.157，QwQ-32B
0.970 / 0.136，**Qwen3.6-27B 只有 0.394 / 0.506**——它的 pmf 几乎不随生成位置增长
（2048 位置上中位 2.9，而 GLM 是 19.2），是另一种注意力形态，那条曲线的可信度低得多。

> **Llama-8B 上曲线的形状其实管不了多少事**：成环轨迹的提示侧质量占比实测已经
> 有 0.62–0.77（32 层平均；其中大部分是提示首位那个注意力汇，见
> `row_reshape` §1），再往上抬所要求的占比普遍 ≥ 1，于是 `solve_prompt_scale` 落进
> 封顶档，强度由 `SHARE_CEILING`=0.95 而不是曲线决定。实测 5 条成环轨迹在生成位置
> 256/512/1024/2048/4096 上封顶比例 4·5·5·3·1 / 5·5·5·3·1（deg=2 / deg=1），
> **两种阶数解出的 c 逐位相同**。阶数在 Llama-8B 上只改变曲线作为「良性行为描述」
> 的准确度，改不了这一半干预的强度。

数据来源按顺序找两处：老的 `exp/step_trajectory_total/<模型>/`（平铺成
`<类别>_<id>.json`），以及一键流程的
`exp/detection/pipeline/<模型>/train/step_metrics/`（按数据集名分子目录，
`<数据集>/<类别>_<id>.json`；该目录下的平铺旧文件也照收）。有效反思类在两处的
名字不同（`productive_reflection` / `productive_reasoning`），两个都认。

构建（base env 或 Recur env 均可，纯 numpy）::

    PYTHONPATH=. python -m recur_code.src.detection.attention_reference \\
        --model DeepSeek-R1-Distill-Llama-8B
    PYTHONPATH=. python -m recur_code.src.detection.attention_reference --all
"""

from __future__ import annotations

import glob
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
RECUR_CODE = REPO_ROOT / "recur_code"
TRAJ_ROOT = RECUR_CODE / "exp" / "step_trajectory_total"
# 一键检测流程把同样的逐位置指标写在别处，两处都找（见模块文档末尾）
PIPELINE_ROOT = RECUR_CODE / "exp" / "detection" / "pipeline"
DEFAULT_OUT_DIR = RECUR_CODE / "exp" / "detection_dataset" / "models"
sys.path.insert(0, str(REPO_ROOT))

from recur_code.src.detection.row_reshape import (  # noqa: E402
    PROMPT_METRIC, SHARE_CEILING,
)

# 曲线拟合的目标量：prompt_mean_fraction，与**检测特征同口径**。它要求的提示侧
# 质量占比 ≥ 1（无解）时求解器不再查曲线，直接改瞄封顶占比 SHARE_CEILING
# （见 row_reshape §1），所以这里只有这一档需要拟合。
METRIC = PROMPT_METRIC
# 合并后的唯一一条曲线的名字
MERGED_KIND = "benign"
# 参与合并的良性类别 → 该类在磁盘上可能用的文件名前缀（有效反思在新老流水线里
# 名字不同，两个都认；同一类别只会命中其中一个）
BENIGN_KINDS = {
    "concise_reasoning": ("concise_reasoning",),
    "productive_reflection": ("productive_reflection", "productive_reasoning"),
}
# 攻击类 → 参考曲线。**两个攻击类现在指向同一条**（见模块文档 §一条曲线）；
# 保留这张表是因为运行时仍按它判「这个类别有没有曲线可用」。
REFERENCE_OF = {
    "repetitive_reasoning": MERGED_KIND,
    "repetitive_string": MERGED_KIND,
}
# 期望的干预方向：循环思考要把提示侧抬上去，循环字符要压下来。目标值不再分类别，
# 但 `--prompt-direction expected` 仍按类别夹方向，所以这张表照旧。
EXPECTED_DIRECTION = {"repetitive_reasoning": "up", "repetitive_string": "down"}
NORMALIZE_MODES = ("position", "position_prompt_len")
# 合并曲线的默认口径：**必须带提示长度项**，否则两类良性各带 ±0.25~±0.41 的
# 系统偏置（模块文档 §一条曲线的表）。`position` 只留作对照。
DEFAULT_NORMALIZE = "position_prompt_len"
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


# --------------------------------------------------------------------- 拟合
@dataclass
class CurveFit:
    """一类正常思考轨迹的 log ρ 拟合结果（一种归一化口径）。"""
    kind: str                      # concise_reasoning / productive_reflection
    normalize: str                 # position / position_prompt_len
    degree: int
    coef: list[float]              # [c0, c1, ...] 对应 (log 生成位置)^0..^degree
    d_log_prompt_len: float        # normalize=position 时为 0
    r2: float
    resid_sd: float
    resid_q: dict[str, float]      # 残差分位数（对数域）
    n_samples: int
    n_points: int
    log_gen_range: list[float]     # 拟合覆盖的 log(生成位置+1) 范围，用于夹取
    log_plen_range: list[float]
    mean_prompt_len: float
    # 合并的代价：每个良性类别在这条共用曲线上的加权平均残差（log 域）。
    # 0 = 这一类没有被系统性地高估/低估。`position` 口径下会看到 ±0.25~±0.41，
    # 那正是「合并必须带提示长度项」的证据（模块文档 §一条曲线）。
    class_bias: dict[str, float] = field(default_factory=dict)
    n_by_kind: dict[str, int] = field(default_factory=dict)

    def predict_log(self, gen_pos: int, prompt_len: int,
                    quantile: float = 0.5) -> float:
        """对数域目标值。自变量夹在拟合范围内，避免二次项外推发散。"""
        x = float(np.clip(np.log(max(int(gen_pos), 0) + 1.0),
                          self.log_gen_range[0], self.log_gen_range[1]))
        y = float(np.polyval(list(reversed(self.coef)), x))
        if self.d_log_prompt_len:
            lp = float(np.clip(np.log(max(int(prompt_len), 1)),
                               self.log_plen_range[0], self.log_plen_range[1]))
            y += self.d_log_prompt_len * lp
        return y + self._resid_offset(quantile)

    def _resid_offset(self, quantile: float) -> float:
        """按残差分位数平移：0.5 = 拟合中位，0.25 = 目标带下沿（更温和）。"""
        if abs(quantile - 0.5) < 1e-9:
            return float(self.resid_q.get("q50", 0.0))
        qs = np.array(QUANTILES, dtype=float)
        vs = np.array([self.resid_q[f"q{int(q * 100):02d}"] for q in QUANTILES],
                      dtype=float)
        return float(np.interp(quantile, qs, vs))

    def predict(self, gen_pos: int, prompt_len: int,
                quantile: float = 0.5) -> float:
        return float(np.exp(self.predict_log(gen_pos, prompt_len, quantile)))

    def to_json(self) -> dict:
        return {"kind": self.kind, "normalize": self.normalize,
                "degree": self.degree, "coef": self.coef,
                "d_log_prompt_len": self.d_log_prompt_len,
                "r2": self.r2, "resid_sd": self.resid_sd,
                "resid_q": self.resid_q, "n_samples": self.n_samples,
                "n_points": self.n_points,
                "log_gen_range": self.log_gen_range,
                "log_plen_range": self.log_plen_range,
                "mean_prompt_len": self.mean_prompt_len,
                "class_bias": self.class_bias, "n_by_kind": self.n_by_kind}

    @classmethod
    def from_json(cls, d: dict) -> "CurveFit":
        return cls(**d)


@dataclass
class AttentionReference:
    """一个模型的全部参考曲线：{归一化口径: {类别: CurveFit}}。"""
    model: str
    metric: str
    fits: dict[str, dict[str, CurveFit]]              # pmf 目标曲线
    # 诊断用：各类别按生成位置分桶的实测分位数（不参与运行时决策）
    bins: dict = field(default_factory=dict)

    # ------------------------------------------------------------ 运行时接口
    def target_ratio(self, attack_class: str, gen_pos: int, prompt_len: int,
                     *, normalize: str = DEFAULT_NORMALIZE,
                     quantile: float = 0.5) -> float:
        """该生成位置上的 pmf 目标。

        **两个攻击类现在共用同一条曲线**（模块文档 §一条曲线），`attack_class`
        只用来确认它确实是个攻击类；方向仍可由 `--prompt-direction expected`
        按类别夹。这个目标要求的提示侧质量占比可能 ≥ 1（提示很长就会这样）——
        那时 `solve_prompt_scale` 改瞄封顶占比 `SHARE_CEILING`，与本曲线无关。"""
        return self.fit(self._kind_of(attack_class), normalize).predict(
            gen_pos, prompt_len, quantile)

    def _kind_of(self, attack_class: str) -> str:
        kind = REFERENCE_OF.get(attack_class)
        if kind is None:
            raise KeyError(f"{attack_class} 不是攻击类别；可用 {list(REFERENCE_OF)}")
        return kind

    def fit(self, kind: str = MERGED_KIND,
            normalize: str = DEFAULT_NORMALIZE) -> CurveFit:
        if normalize not in self.fits:
            raise KeyError(f"参考曲线里没有 normalize={normalize}，"
                           f"有 {list(self.fits)}")
        if kind not in self.fits[normalize]:
            raise KeyError(f"参考曲线里没有 {kind}，有 "
                           f"{list(self.fits[normalize])}")
        return self.fits[normalize][kind]

    # ------------------------------------------------------------------- io
    def to_json(self) -> dict:
        return {"model": self.model, "metric": self.metric,
                "fits": {nm: {k: f.to_json() for k, f in d.items()}
                         for nm, d in self.fits.items()},
                "reference_of": REFERENCE_OF, "bins": self.bins}

    @classmethod
    def load(cls, path: str | os.PathLike) -> "AttentionReference":
        d = json.load(open(path))
        metric = d.get("metric", "")
        if metric != METRIC:
            raise ValueError(
                f"{path} 的目标量是 {metric or '（未记录，旧版文件）'}，"
                f"现在只支持 {METRIC}。重建："
                f"python -m recur_code.src.detection.attention_reference --all")
        # 旧文件里可能还带着已废弃的 share_fits（回退档），直接忽略。
        load = lambda sub: {nm: {k: CurveFit.from_json(v) for k, v in d2.items()}  # noqa: E731
                            for nm, d2 in sub.items()}
        fits = load(d["fits"])
        missing = [nm for nm in fits if MERGED_KIND not in fits[nm]]
        if missing:
            raise ValueError(
                f"{path} 是按类别分开拟合的旧曲线（{missing} 里没有 "
                f"{MERGED_KIND!r}），现在两个攻击类共用一条合并曲线。重建："
                f"python -m recur_code.src.detection.attention_reference --all")
        return cls(model=d["model"], metric=metric, fits=fits,
                   bins=d.get("bins", {}))


# ------------------------------------------------------------------ 数据读取
def _traj_dirs(model: str) -> list[Path]:
    """这个模型的逐位置指标可能放在哪几个目录（见模块文档末尾）。"""
    return [TRAJ_ROOT / model,
            PIPELINE_ROOT / model / "train" / "step_metrics"]


def _load_kind(model: str, kind: str, stride: int,
               metric: str = METRIC) -> list[dict]:
    """一个良性类别的全部轨迹：[{kind, lg, ly, P}]，每条轨迹一项。

    `kind` 是 `BENIGN_KINDS` 里的标准名；磁盘上的文件名前缀可能是它的别名
    （有效反思在新老流水线里叫法不同），命中哪个都归到标准名下。
    """
    out = []
    for d in _traj_dirs(model):
        for alias in BENIGN_KINDS.get(kind, (kind,)):
            # 一键流程的指标现在按数据集名分了子目录
            # （`step_metrics/GSM8k/concise_reasoning_0.json`）；老目录仍是平铺的，
            # 两种都扫，去重后按文件名排序。
            hits = sorted(set(glob.glob(str(d / f"{alias}_*.json")))
                          | set(glob.glob(str(d / "*" / f"{alias}_*.json"))))
            for f in hits:
                rec = json.load(open(f))
                a = np.array([x if x is not None else np.nan for x in rec[metric]],
                             dtype=float)
                P = int(rec["prompt_len"])
                ok = np.isfinite(a) & (a > 0)
                idx = np.arange(a.size)[ok][::stride]
                if idx.size < 4:
                    continue
                out.append({"kind": kind, "P": P,
                            "lg": np.log(idx + 1.0), "ly": np.log(a[idx])})
        if out:
            break            # 一个模型只认一处，不跨目录混样本
    return out


def _fit(samples: list[dict], normalize: str, degree: int,
         kind: str = MERGED_KIND) -> CurveFit:
    """两级归一的对数域加权最小二乘（见模块文档 §拟合形式）。

    第一级按轨迹：每条轨迹总权重相同，否则 10 万点的长轨迹会淹没短轨迹。
    第二级按类别：每个良性类别总权重相同，否则「混合曲线」会变成样本多的那一类
    的曲线——类别数平衡时这一级是恒等变换，不平衡时它是合并能成立的前提。
    """
    n_by_kind: dict[str, int] = {}
    for s in samples:
        n_by_kind[s["kind"]] = n_by_kind.get(s["kind"], 0) + 1
    Xs, ys, ws, ks = [], [], [], []
    for s in samples:
        lg, P = s["lg"], s["P"]
        cols = [lg ** i for i in range(degree + 1)]
        if normalize == "position_prompt_len":
            cols.append(np.full(lg.size, np.log(max(P, 1))))
        Xs.append(np.column_stack(cols))
        ys.append(s["ly"])
        ws.append(np.full(lg.size, 1.0 / (lg.size * n_by_kind[s["kind"]])))
        ks.append(np.full(lg.size, s["kind"], dtype=object))
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    w = np.concatenate(ws)
    k = np.concatenate(ks)
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    resid = y - X @ beta
    wm = float(np.sum(w * y) / np.sum(w))
    ss_res = float(np.sum(w * resid ** 2))
    ss_tot = float(np.sum(w * (y - wm) ** 2))
    lp = np.log([max(s["P"], 1) for s in samples])
    # 合并的代价：每类的加权平均残差。position 口径下这两个数会明显不为 0。
    bias = {}
    for kk in sorted(n_by_kind):
        m = k == kk
        bias[kk] = round(float(np.sum(w[m] * resid[m]) / np.sum(w[m])), 4)
    return CurveFit(
        kind=kind, normalize=normalize, degree=degree,
        coef=[round(float(b), 6) for b in beta[: degree + 1]],
        d_log_prompt_len=(round(float(beta[-1]), 6)
                          if normalize == "position_prompt_len" else 0.0),
        r2=round(1.0 - ss_res / ss_tot, 4) if ss_tot > 0 else float("nan"),
        resid_sd=round(float(np.sqrt(ss_res / np.sum(w))), 4),
        resid_q={f"q{int(q * 100):02d}": round(float(np.quantile(resid, q)), 4)
                 for q in QUANTILES},
        n_samples=len(samples), n_points=int(y.size),
        log_gen_range=[round(float(X[:, 1].min()), 4),
                       round(float(X[:, 1].max()), 4)],
        log_plen_range=[round(float(lp.min()), 4), round(float(lp.max()), 4)],
        mean_prompt_len=round(float(np.mean([s["P"] for s in samples])), 1),
        class_bias=bias, n_by_kind=dict(sorted(n_by_kind.items())),
    )


def _bins(model: str, stride: int, metric: str = METRIC) -> dict:
    """诊断表：各类别按生成位置分桶的目标量分位数。

    列出两个良性类别各自、它们合并之后（`benign`，就是曲线拟合用的那一批），
    以及两个攻击类——合并曲线与攻击轨迹的落差就是抑制要跨的距离。
    """
    edges = [0, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    out: dict[str, list] = {}
    merged: list[dict] = []
    cats = list(BENIGN_KINDS) + [MERGED_KIND] + list(REFERENCE_OF)
    for cat in cats:
        if cat == MERGED_KIND:
            rows = merged
        elif cat in BENIGN_KINDS:
            rows = _load_kind(model, cat, stride, metric)
            merged.extend(rows)
        else:                                  # 攻击类：文件名就是类名
            rows = _load_kind(model, cat, stride, metric)
        if not rows:
            continue
        table = []
        for i, lo in enumerate(edges):
            hi = edges[i + 1] if i + 1 < len(edges) else 10 ** 9
            v = np.concatenate([np.exp(r["ly"][(np.exp(r["lg"]) - 1 >= lo)
                                               & (np.exp(r["lg"]) - 1 < hi)])
                                for r in rows])
            table.append({"lo": lo, "n": int(v.size),
                          "q25": round(float(np.quantile(v, .25)), 4) if v.size else None,
                          "q50": round(float(np.quantile(v, .50)), 4) if v.size else None,
                          "q75": round(float(np.quantile(v, .75)), 4) if v.size else None})
        out[cat] = table
    return out


def _all_models() -> list[str]:
    """两处数据目录下出现过的全部模型名（去重、有序）。"""
    names: list[str] = []
    for root in (TRAJ_ROOT, PIPELINE_ROOT):
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            if d.is_dir() and d.name not in names and \
                    any(x.is_dir() or x.suffix == ".json"
                        for x in _traj_dirs(d.name) if x.exists()):
                names.append(d.name)
    return names


def build(model: str, *, degree: int = 2, stride: int = 4
          ) -> AttentionReference:
    """拟合**一条**合并良性曲线（见模块文档 §一条曲线）。

    两个良性类别的轨迹混在一起拟合，两个攻击类共用结果。两种归一化口径都写进
    同一个文件，运行时按 `--ref-normalize` 取用；默认 `position_prompt_len`，
    因为合并不带长度项时每类会各带 ±0.25~±0.41 的系统偏置。
    """
    samples: list[dict] = []
    per_kind: dict[str, int] = {}
    for kind in sorted(BENIGN_KINDS):
        rows = _load_kind(model, kind, stride, METRIC)
        per_kind[kind] = len(rows)
        samples.extend(rows)
    if not samples:
        raise FileNotFoundError(
            f"{[str(d) for d in _traj_dirs(model)]} 下没有任何良性轨迹"
            f"（{list(BENIGN_KINDS)}），无法拟合参考曲线")
    missing = [k for k, n in per_kind.items() if n == 0]
    if missing:
        # 只有一类良性时曲线仍然能拟合，但它已经不是「混合」曲线了
        print(f"  [警告] {model}: 缺 {missing} 的轨迹，这条曲线只由 "
              f"{[k for k, n in per_kind.items() if n]} 拟合")
    fits = {nm: {MERGED_KIND: _fit(samples, nm, degree)}
            for nm in NORMALIZE_MODES}
    return AttentionReference(model=model, metric=METRIC, fits=fits,
                              bins=_bins(model, stride, METRIC))


# --------------------------------------------------------------------------- #
def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="DeepSeek-R1-Distill-Llama-8B")
    ap.add_argument("--all", action="store_true",
                    help=f"为 {TRAJ_ROOT} 下每个模型各建一份")
    ap.add_argument("--degree", type=int, default=2)
    ap.add_argument("--stride", type=int, default=4,
                    help="每条轨迹每隔多少个生成位置取一个拟合点")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    a = ap.parse_args()

    models = (_all_models() if a.all else [a.model])
    if a.all and not models:
        raise SystemExit(f"{[str(TRAJ_ROOT), str(PIPELINE_ROOT)]} 下都没有模型目录")
    for model in models:
        try:
            ref = build(model, degree=a.degree, stride=a.stride)
        except FileNotFoundError as e:
            print(f"[跳过] {model}: {e}")
            continue
        # 阶数进文件名（默认 2 不进），免得「试一下直线」把部署在用的那份覆盖掉
        tag = "" if a.degree == 2 else f"_deg{a.degree}"
        out = Path(a.out_dir) / f"attention_reference_{model}{tag}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(ref.to_json(), open(out, "w"), ensure_ascii=False, indent=1)
        print(f"\n=== {model} （目标量 {METRIC}；一条合并曲线 {MERGED_KIND}；"
              f"越界改瞄封顶占比 {SHARE_CEILING}）===")
        for nm in NORMALIZE_MODES:
            f = ref.fits[nm][MERGED_KIND]
            mark = " ←默认" if nm == DEFAULT_NORMALIZE else ""
            print(f"  {nm:20s} n={f.n_samples:3d}{f.n_by_kind} "
                  f"点={f.n_points:7d} R²={f.r2} 残差sd={f.resid_sd} "
                  f"提示长度系数={f.d_log_prompt_len:+.3f} "
                  f"提示长度均值={f.mean_prompt_len:.0f}{mark}")
            # 合并的代价：每类的加权平均残差。不带长度项时这两个数会明显不为 0，
            # 那说明这条曲线对一类整体偏高、对另一类整体偏低。
            print(f"  {'':20s} 类别残差偏置 "
                  + "  ".join(f"{k}={v:+.4f}" for k, v in f.class_bias.items()))
        print(f"  {'生成位置':>8}" + "".join(f"{k[:14]:>16}" for k in ref.bins))
        first = ref.bins[list(ref.bins)[0]]
        for i in range(len(first)):
            row = f"  {first[i]['lo']:>8}"
            for k in ref.bins:
                q50 = ref.bins[k][i]["q50"]
                cell = "-" if q50 is None else f"{q50:.2f}"
                row += f"{cell:>16}"
            print(row)
        print(f"  已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
