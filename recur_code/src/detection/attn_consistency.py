"""注意力一致性指标（`attn_consistency_independent`）的在线实现。

**这是一个检测特征，不是抑制算子**。给定当前查询 token 的注意力行、提示长度、逐位置
token 串，算出这一行的一致性：

    1. 整行归一到和为 1，取累计质量达到 `threshold`（0.9）的 key，即 top-θ 集合；
    2. 把 top-θ 里的位置按 **token 串**分组，分成提示侧（`pos < prompt_len`）与生成侧
       两半，**各自侧内归一**：
           prop_p(t) = 提示侧该串质量 / 提示侧总质量
           prop_g(t) = 生成侧该串质量 / 生成侧总质量
    3. 对两侧**共有**的串取 L1 差之和，一致性 = 1 − Σ_共有 |prop_p − prop_g|。

值域 0–1，越低表示生成侧关注的词分布越偏离它在提示里的样子（自指退化）。这与离线
三阶段流水线 `compute_step_trajectory.py` 里
`_compute_per_token_consistency_independent` 的定义**逐字相同**，所以在线量与离线量可比。

三个边界与调用方约定：

  - **行退化**（整行和 ≤ 0，或某一侧在 top-θ 里为空）→ 返回 `NaN`，与离线实现一致；
  - **两侧无共有串** → 返回 `1.0`（完全一致），不是 NaN；
  - 所以**门控必须读这个返回值本身**，不能从「有没有可压的东西」反推——「没有共有串」
    是最一致的情形，不是未定义的情形。历史上曾有过这个坑。

谁在用：

  - `detector.py` 的第一个分类特征（这条序列在 [0, i] 上的累计均值）；
  - `suppression_runtime.py` 的逐层动态选层规则（`--dyn-rule thresh/reldrop`）；
  - 离线侧同一定义在 `compute_step_trajectory.py` 里独立实现（两边都不许单方面改）。

自测：

    python recur_code/src/detection/attn_consistency.py
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

DEFAULT_THRESHOLD = 0.9


def _top_theta_sides(row_raw: np.ndarray, prompt_len: int, tokens: list[str],
                     threshold: float):
    """top-θ 选择 + 按 token 串分侧归一。

    返回 `(prop_p, prop_g)` 两个 `token 串 -> 侧内占比` 的字典；行退化或任一侧在
    top-θ 里为空时返回 `None`（指标在该查询上未定义）。"""
    row = row_raw.astype(np.float64)
    s = row.sum()
    if s <= 0:
        return None
    row = row / s
    order = np.argsort(-row, kind="stable")
    csum = np.cumsum(row[order])
    k = int(np.searchsorted(csum, threshold) + 1)
    k = max(1, min(k, len(row)))

    prompt_cum: dict[str, float] = defaultdict(float)
    gen_cum: dict[str, float] = defaultdict(float)
    for pos in order[:k].tolist():
        tok = tokens[pos]
        (prompt_cum if pos < prompt_len else gen_cum)[tok] += float(row[pos])
    p_total = sum(prompt_cum.values())
    g_total = sum(gen_cum.values())
    if p_total <= 0 or g_total <= 0:
        return None
    return ({t: v / p_total for t, v in prompt_cum.items()},
            {t: v / g_total for t, v in gen_cum.items()})


def consistency(row_raw: np.ndarray, prompt_len: int, tokens: list[str],
                threshold: float = DEFAULT_THRESHOLD) -> float:
    """一行的注意力一致性，值域 0–1；未定义时返回 NaN（见模块文档的三个边界）。"""
    sides = _top_theta_sides(row_raw, prompt_len, tokens, threshold)
    if sides is None:
        return float("nan")
    prop_p, prop_g = sides
    common = set(prop_p) & set(prop_g)
    if not common:
        return 1.0
    return 1.0 - sum(abs(prop_p[t] - prop_g[t]) for t in common)


def _selftest() -> int:
    """确定性合成用例，覆盖模块文档列的三个边界 + 一个正常值。"""
    ok = True

    def check(name: str, got, want, tol=1e-12):
        nonlocal ok
        good = (np.isnan(got) and np.isnan(want)) or abs(got - want) <= tol
        print(f"  {'PASS' if good else 'FAIL'}  {name}: got={got!r} want={want!r}")
        ok = ok and good

    # 1) 行退化：整行和为 0 → NaN
    check("全零行 → NaN",
          consistency(np.zeros(4), 2, ["a", "b", "a", "b"]), float("nan"))

    # 2) 生成侧在 top-θ 里为空（质量全在提示侧）→ NaN
    check("生成侧为空 → NaN",
          consistency(np.array([0.98, 0.01, 0.005, 0.005]), 2,
                      ["a", "b", "c", "d"]), float("nan"))

    # 3) 两侧无共有串 → 1.0
    check("无共有串 → 1.0",
          consistency(np.array([0.3, 0.3, 0.2, 0.2]), 2,
                      ["a", "b", "c", "d"]), 1.0)

    # 4) top-θ 会**截掉尾部**：row=[.3,.1,.1,.5] 按质量降序是 .5/.3/.1/.1，累计
    #    .5/.8/.9/1.0，达到 0.9 时只收了前 3 个——第 4 个位置（提示侧的 a 之外那个）
    #    落在集合外。所以提示侧只有 a:.3 b:.1 → {a:.75, b:.25}，生成侧只有 b:.5
    #    → {b:1.0}，共有 {b}，Σ|Δ| = |.25 − 1| = .75，一致性 = .25。
    #    这条用例专门钉住「θ 截断真的发生了」——把 0.9 写成 1.0 它就会变。
    check("top-θ 截断后只剩一个共有串",
          consistency(np.array([0.3, 0.1, 0.1, 0.5]), 2, ["a", "b", "a", "b"]),
          0.25)

    # 5) 两个共有串都在集合内：row=[.3,.25,.25,.2] 累计 .3/.55/.8/1.0，θ=0.9 收全 4 个。
    #    提示侧 {a:.3/.55, b:.25/.55}、生成侧 {a:.25/.45, b:.2/.45}
    p_a, p_b = 0.3 / 0.55, 0.25 / 0.55
    g_a, g_b = 0.25 / 0.45, 0.2 / 0.45
    check("两个共有串的一致性",
          consistency(np.array([0.3, 0.25, 0.25, 0.2]), 2, ["a", "b", "a", "b"]),
          1.0 - (abs(p_a - g_a) + abs(p_b - g_b)))

    # 6) 两侧分布完全相同 → 1.0
    check("两侧同分布 → 1.0",
          consistency(np.array([0.25, 0.25, 0.25, 0.25]), 2,
                      ["a", "b", "a", "b"]), 1.0)

    print("SELFTEST PASSED" if ok else "SELFTEST FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
