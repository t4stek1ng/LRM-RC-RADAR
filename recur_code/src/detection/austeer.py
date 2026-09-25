"""AUSteer —— 细粒度激活引导，作为循环抑制的**独立对比基线**。

复现 Feng et al., ICLR 2026, *Fine-Grained Activation Steering: Steering Less,
Achieving More*（arXiv:2602.04428）。这个文件只做两件事：

  1. **定位**（离线，一次性）：在对比样本对上算激活动量，选出 top-k 个原子单元
     （AU），产出一个 JSON 计划；
  2. **施加**（在线，每个前向步）：按计划对选中的标量维度做乘性缩放。

方法本身（论文 §3–4）
---------------------
线性投影 `y = Wx = Σ_i x_i · W[:, i]`。每一列 `W[:, i]` 是一个 **AU**，标量 `x_i`
是它的系数。于是「引导第 i 维激活」＝「引导第 i 个 AU」。论文的论点是 block 级
激活是**异质的**：不同 AU 支配输出 token 分布的不同方向，整块引导必然把有益与
有害方向一起推，所以只动少数有益 AU 反而更强（steering less achieves more）。

两个变体（论文 §5.1），对应两个挂载点：

  * ``ffn``  —— ``layers[l].mlp.down_proj`` 的**输入**（intermediate 维）
  * ``head`` —— ``layers[l].self_attn.o_proj`` 的**输入**（hidden 维，按头切分）

定位：激活动量（论文 §4.1）
---------------------------
给定 N 对对比样本，正例是**期望行为**、负例是**要消除的行为**：

    m_i^j = x_i^{j,pos} − x_i^{j,neg}
    r_i^pos = (1/N) Σ_j 1(m_i^j > 0)     r_i^neg = (1/N) Σ_j 1(m_i^j < 0)
    s_i = max(r_i^pos, r_i^neg)

`s_i` 是比例量，量纲与层深无关，因此可以**跨层全局排序**取前 k 个（这一点正是
论文选它而不选激活幅值的理由）。

施加：自适应强度（论文 §4.2）
-----------------------------
    x̂_i = x_i + γ_i · x_i
    γ_i = +α · r_i^pos   若 r_i^pos > r_i^neg
        = −α · r_i^neg   否则

乘性、保号，对输入自适应只体现在「乘的是 x_i 自己」；`γ_i` 本身是**离线定死的
常数**，在线不随轨迹状态变化。这也是它与我们主方案的本质差别：AUSteer 是
「离线计划 + 在线施加 + 无在线决策」。

本任务的适配（唯一改动的地方）
------------------------------
论文的正负例是「问题 + 正确答案」/「问题 + 错误答案」。映射到循环抑制：

    正例 = Prompt + **未成环**的思考/回答（期望行为）
    负例 = Prompt + **成环**的思考/回答（要消除的行为）

两个必须处理的混淆项，见 `austeer_localize.py`：正负序列**截到相同长度**（成环
轨迹天然长得多，不截会选出「长度 AU」），以及读取位置固定在截断后序列的最后
一个 token（SADI 的惯例，论文正文未写死，这里显式记在计划文件里）。

计划文件格式
------------
``{
   "model": ..., "variant": "ffn"|"head", "alpha": float, "top_k": int,
   "readout": {...}, "source": {...},            # 复现所需的全部元信息
   "plans": {
     "<class>": {"n_pairs": int,
                 "aus": [{"layer": int, "dim": int,
                          "r_pos": float, "r_neg": float, "s": float}, ...]}
   }
 }``

``<class>`` 取 ``repetitive_reasoning`` / ``repetitive_string`` / ``union``。
`alpha` 存在计划里只是记录定位时用的值，施加时可以被覆盖（超参扫描要扫它）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# 两个变体的挂载点：模块后缀 -> 说明。取的都是该线性层的**输入**，即
# `y = Wx` 里的 x，它的每一维就是一个 AU 的系数。
VARIANTS = {
    "ffn": "mlp.down_proj",
    "head": "self_attn.o_proj",
}
DEFAULT_VARIANT = "ffn"
# 论文主实验把被引导的激活数**卡在 100**（Table 1 的 #Acts 列），这是它的核心
# 主张，基线复现不放宽。
PAPER_TOP_K_CAP = 100
UNION = "union"


def module_suffix(variant: str) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {sorted(VARIANTS)}")
    return VARIANTS[variant]


def _resolve(obj, path: str):
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def get_block_modules(model, variant: str) -> list[tuple[Any, str]]:
    """按变体取出每层的目标模块，返回 `[(模块, 路径), ...]`，索引即层号。

    MoE 模型（我们这里是 GLM-4.7-Flash）的 `mlp` 只在密集层上有 `down_proj`，
    MoE 层上是 `experts.<i>.down_proj` + `shared_experts.down_proj`。路由专家的
    `down_proj` 只对被路由到它的 token 生效，那样的 AU 系数不是「每个 token 都
    有」的量，与论文的定义对不上；**共享专家**每个 token 都过，所以 MoE 层取
    `mlp.shared_experts.down_proj`。两者的维度不同（GLM：密集层 10240、共享专家
    1536），因此下游一律按**逐层维度**处理，不假设矩形。

    某层两条路径都取不到时返回 `(None, "")`，由调用方跳过并记录。
    """
    suffix = module_suffix(variant)
    fallbacks = ([suffix, "mlp.shared_experts.down_proj"]
                 if variant == "ffn" else [suffix])
    layers = model.model.layers if hasattr(model, "model") else model.layers
    out = []
    for blk in layers:
        got = (None, "")
        for path in fallbacks:
            obj = _resolve(blk, path)
            if obj is not None and hasattr(obj, "weight"):
                got = (obj, path)
                break
        out.append(got)
    return out


# --------------------------------------------------------------- 定位（离线）
class MomentumAccumulator:
    """逐对累加激活动量的符号计数（论文 Eq.2）。

    只保留每层的两个计数向量，不存原始激活：一对样本进来先各自归约成一个
    `[n_layers, d]` 的读取矩阵，相减取符号即可，内存与对数无关。
    """

    def __init__(self, dims: Sequence[int],
                 modules: Sequence[str] | None = None):
        # 逐层维度：MoE 模型的密集层与共享专家宽度不同（见 `get_block_modules`），
        # 所以不假设矩形。跳过的层记 0 维。
        self.dims = [int(d) for d in dims]
        self.n_layers = len(self.dims)
        # 每层实际挂的模块路径（MoE 层与密集层可能不同），写进计划供施加端使用
        self.modules = list(modules) if modules else [""] * self.n_layers
        self.offsets = np.cumsum([0] + self.dims)
        self.dim = int(self.offsets[-1])          # 扁平后的总 AU 数
        self.pos = np.zeros(self.dim, dtype=np.int64)
        self.neg = np.zeros(self.dim, dtype=np.int64)
        # 动量本身的和与绝对值和，只用来给 s 的并列**定序**（见 `top_k`）：
        # |Σm| / Σ|m| ∈ [0, 1] 是无量纲的，层深带来的幅值差被约掉，所以它和 s
        # 一样可以跨层比较——论文避开幅值正是因为幅值本身不可跨层比。
        self.msum = np.zeros(self.dim, dtype=np.float64)
        self.mabs = np.zeros(self.dim, dtype=np.float64)
        self.n_pairs = 0
        # `top_k` 落在截断线上的并列个数；并列太多说明样本对不够多或者太同质，
        # 选出来的 AU 有多少是「真的排进去的」需要如实报出来。
        self.n_tied_at_cut = 0

    def add_pair(self, x_pos: np.ndarray, x_neg: np.ndarray) -> None:
        """x_pos / x_neg: 扁平后的 `[Σ dims]`，同一对样本的读取值。"""
        if x_pos.shape != (self.dim,):
            raise ValueError(f"x_pos 形状 {x_pos.shape} != {(self.dim,)}")
        if x_neg.shape != x_pos.shape:
            raise ValueError("正负例读取向量形状不一致")
        m = x_pos.astype(np.float64) - x_neg.astype(np.float64)
        self.pos += (m > 0)
        self.neg += (m < 0)
        self.msum += m
        self.mabs += np.abs(m)
        self.n_pairs += 1

    def _locate(self, idx: int) -> tuple[int, int]:
        """扁平下标 → (层号, 层内维度)。"""
        li = int(np.searchsorted(self.offsets, idx, side="right") - 1)
        return li, int(idx - self.offsets[li])

    def scores(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 (r_pos, r_neg, s)，形状均为扁平的 `[Σ dims]`。"""
        if not self.n_pairs:
            raise ValueError("还没有累加任何样本对")
        r_pos = self.pos / self.n_pairs
        r_neg = self.neg / self.n_pairs
        return r_pos, r_neg, np.maximum(r_pos, r_neg)

    def top_k(self, k: int, *, tie_seed: int = 0) -> list[dict[str, Any]]:
        """跨层全局取前 k 个 AU（论文 §4.1 的「globally rank」）。

        判据分三级，前者相等才看后者：

          1. `s = max(r_pos, r_neg)` —— 论文定义的判别力；
          2. `|r_pos − r_neg|` —— 同样的一致性下，方向更明确的更可信；
          3. `snr = |Σm| / Σ|m|` —— 符号一致**且**幅值一致。

        第三级是必须的：`s` 的取值只有 N+1 档（N 是样本对数），对数一少就会出现
        成千上万个并列，纯按下标定序会让选中的 AU 全部挤在最前面的层里——这是
        实现细节造成的系统偏差，不是数据里的信号。
        """
        r_pos, r_neg, s = self.scores()
        bias = np.abs(r_pos - r_neg)
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = np.abs(self.msum) / self.mabs
        snr = np.nan_to_num(snr, nan=0.0, posinf=0.0)
        key = s * 4.0 + bias * 2.0 + snr
        # 三级判据都相等时（对数很少、或数据本身退化），按下标定序会把名额全发给
        # 最前面的层。加一个种子固定的极小抖动：结果仍然可复现，但并列内部是
        # 无偏的。量级 1e-9 远小于任何真实差距，不会改变非并列的次序。
        key = key + np.random.default_rng(tie_seed).random(key.shape) * 1e-9
        flat = np.argsort(-key, kind="stable")[:int(k)]
        if k < key.size:
            cut = np.partition(key, -int(k))[-int(k)]
            self.n_tied_at_cut = int(np.sum(np.isclose(key, cut, atol=1e-8)))
        out = []
        for idx in flat:
            li, di = self._locate(int(idx))
            out.append({"layer": li, "dim": di,
                        "module": self.modules[li],
                        "r_pos": float(r_pos[idx]),
                        "r_neg": float(r_neg[idx]),
                        "s": float(s[idx]),
                        "snr": round(float(snr[idx]), 4)})
        return out


# --------------------------------------------------------------- 计划（在线）
@dataclass
class AUPlan:
    """一条臂实际要施加的 AU 集合 —— 已经选定类别、k 与 α。"""

    variant: str
    alpha: float
    entries: list[dict[str, Any]] = field(default_factory=list)
    cls: str = UNION
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_acts(self) -> int:
        """被引导的激活数，对应论文 Table 1 的 `#Acts` 列。"""
        return len(self.entries)

    def gammas_by_layer(self) -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]:
        """归约成逐 (层, 模块路径) 的 (维度下标, γ) 两个数组，供 hook 直接用。

        γ_i = +α·r_pos（促进）或 −α·r_neg（压制），符号由哪一侧的一致性更高
        决定（论文 §4.2）。键里带模块路径是因为 MoE 模型同一变体在不同层上挂的
        模块不同（见 `get_block_modules`）。
        """
        by: dict[tuple[int, str], list[tuple[int, float]]] = {}
        for e in self.entries:
            r_pos, r_neg = float(e["r_pos"]), float(e["r_neg"])
            g = self.alpha * r_pos if r_pos > r_neg else -self.alpha * r_neg
            key = (int(e["layer"]), e.get("module") or "")
            by.setdefault(key, []).append((int(e["dim"]), g))
        out = {}
        for key, items in by.items():
            items.sort()
            out[key] = (np.array([i for i, _ in items], dtype=np.int64),
                        np.array([g for _, g in items], dtype=np.float64))
        return out

    def summary(self) -> dict[str, Any]:
        layers = sorted({int(e["layer"]) for e in self.entries})
        return {"variant": self.variant, "class": self.cls,
                "alpha": self.alpha, "n_acts": self.n_acts,
                "layers": layers,
                "s_min": (round(min(float(e["s"]) for e in self.entries), 4)
                          if self.entries else None),
                "s_max": (round(max(float(e["s"]) for e in self.entries), 4)
                          if self.entries else None)}


def load_plan_file(path: str | Path) -> dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)


def select_plan(doc: dict[str, Any], cls: str = UNION, *,
                top_k: int | None = None,
                alpha: float | None = None) -> AUPlan:
    """从计划文件里取出某个类别的 AU 集合，按需截断到 top-k 并换 α。

    类别缺失时退回 `union`；`union` 也没有就按 s 合并所有类别现搭一个（同一个
    (层, 维) 取 s 最大的那条），这样定位脚本只产出分类别计划也能跑联合臂。
    """
    plans = doc.get("plans") or {}
    if cls in plans:
        entries = list(plans[cls]["aus"])
    elif UNION in plans:
        entries = list(plans[UNION]["aus"])
    else:
        merged: dict[tuple[int, int], dict[str, Any]] = {}
        for sub in plans.values():
            for e in sub["aus"]:
                key = (int(e["layer"]), int(e["dim"]))
                if key not in merged or e["s"] > merged[key]["s"]:
                    merged[key] = e
        entries = sorted(merged.values(), key=lambda e: -e["s"])
    if top_k is not None:
        entries = entries[:int(top_k)]
    a = float(doc.get("alpha", 1.0) if alpha is None else alpha)
    return AUPlan(variant=doc.get("variant", DEFAULT_VARIANT), alpha=a,
                  entries=entries, cls=cls,
                  meta={k: doc.get(k) for k in
                        ("model", "readout", "source", "top_k")})


class AUSteerHooks:
    """把一个 `AUPlan` 挂到模型上：每个前向步对选中维度做 `x̂ = x + γx`。

    干预点是目标线性层的**输入**，所以用 `forward_pre_hook`。是否在本步施加由
    `gate()` 决定：

      * AUSteer 原样的基线 → `gate` 恒真（常开，论文的设定）；
      * 消融臂「AUSteer 算子 + 我们的门」→ `gate` 接运行时的门状态。

    `apply_prefill=False` 时只在解码步施加（输入长度为 1），prompt 编码阶段原样
    放过——见模块文档里对这个实现选择的说明。
    """

    def __init__(self, model, plan: AUPlan, *, gate=None,
                 apply_prefill: bool = True, dtype=None):
        self.model = model
        self.plan = plan
        self.gate = gate or (lambda: True)
        self.apply_prefill = bool(apply_prefill)
        self.handles: list = []
        self.n_applied = 0          # 实际施加的 (层, 步) 次数，落进结果供核对
        self._install(dtype)

    def _install(self, dtype) -> None:
        import torch

        mods = get_block_modules(self.model, self.plan.variant)
        layers = (self.model.model.layers if hasattr(self.model, "model")
                  else self.model.layers)
        for (li, path), (idx, gam) in self.plan.gammas_by_layer().items():
            if li >= len(mods):
                raise ValueError(f"计划里的层号 {li} 超出模型层数 {len(mods)}")
            # 计划里记了模块路径就按它挂（MoE 层与密集层不同），没记就按变体解析
            mod = _resolve(layers[li], path) if path else mods[li][0]
            if mod is None:
                raise ValueError(f"第 {li} 层取不到模块 {path or self.plan.variant}")
            dev = next(mod.parameters()).device
            dt = dtype or next(mod.parameters()).dtype
            idx_t = torch.as_tensor(idx, device=dev, dtype=torch.long)
            # 直接存 1+γ，施加时一次乘法搞定
            scale_t = torch.as_tensor(1.0 + gam, device=dev, dtype=dt)

            def pre_hook(module, args, _idx=idx_t, _scale=scale_t):
                if not args:
                    return None
                x = args[0]
                if x.shape[1] != 1 and not self.apply_prefill:
                    return None
                if not self.gate():
                    return None
                # 不能原地改：x 可能被上游复用（残差、checkpoint）
                x = x.clone()
                x[..., _idx] = x[..., _idx] * _scale
                self.n_applied += 1
                return (x,) + tuple(args[1:])

            self.handles.append(mod.register_forward_pre_hook(pre_hook))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def reset_counter(self) -> None:
        self.n_applied = 0


# ------------------------------------------------------------- 对比样本的读取
def truncated_ids(tokenizer, prompt: str, response: str, *,
                  max_response_tokens: int,
                  chat_template: bool = True) -> list[int]:
    """把「Prompt + 回答」编码成一条序列，回答段**截到固定长度**。

    正负例的长度必须一致，否则动量抓到的是长度而不是行为（成环轨迹天然长得
    多）。截断只作用于回答段，Prompt 原样保留。
    """
    if chat_template:
        head = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True)
    else:
        head = prompt
    head_ids = tokenizer(head, add_special_tokens=False).input_ids
    resp_ids = tokenizer(response, add_special_tokens=False).input_ids
    return head_ids + resp_ids[: int(max_response_tokens)]


def read_activations(model, modules: Sequence, input_ids, *,
                     positions: str = "last") -> np.ndarray:
    """teacher-forcing 前向一遍，取每层目标模块**输入**的读取值。

    `positions="last"` 取序列最后一个 token（SADI 惯例，论文未写死，见模块
    文档）；`positions="mean"` 取整条序列的均值，留给消融。

    `modules` 是 `get_block_modules` 的返回值；取不到模块的层贡献 0 个 AU。
    返回**扁平**的 `[Σ dims]`，顺序与 `MomentumAccumulator(dims)` 一致。
    无反向、无梯度，显存等同普通推理（论文附录 I）。
    """
    import torch

    buf: dict[int, np.ndarray] = {}
    handles = []

    def make(li):
        def hook(module, args):
            x = args[0]
            v = x[0].float().mean(dim=0) if positions == "mean" else x[0, -1].float()
            buf[li] = v.detach().cpu().numpy()
            return None
        return hook

    for li, (mod, _path) in enumerate(modules):
        if mod is not None:
            handles.append(mod.register_forward_pre_hook(make(li)))
    try:
        with torch.no_grad():
            model(input_ids=input_ids,
                  attention_mask=torch.ones_like(input_ids))
    finally:
        for h in handles:
            h.remove()
    return np.concatenate([buf[i] for i in range(len(modules)) if i in buf])


def module_dims(modules: Sequence) -> tuple[list[int], list[str]]:
    """每层目标模块的输入维度（= 该层的 AU 个数）与模块路径。

    线性层 `y = Wx` 的权重形状是 `[out, in]`，AU 是 W 的**列**，所以个数 = in。
    """
    dims, paths = [], []
    for mod, path in modules:
        if mod is None:
            dims.append(0)
            paths.append("")
        else:
            dims.append(int(mod.weight.shape[1]))
            paths.append(path)
    return dims, paths


def write_plan(path: str | Path, *, model_name: str, variant: str,
               alpha: float, top_k: int, readout: dict[str, Any],
               source: dict[str, Any],
               plans: dict[str, dict[str, Any]]) -> Path:
    doc = {"model": model_name, "variant": variant, "alpha": float(alpha),
           "top_k": int(top_k), "readout": readout, "source": source,
           "plans": plans}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)
    return path
