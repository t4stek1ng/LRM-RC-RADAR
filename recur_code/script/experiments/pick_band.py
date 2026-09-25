#!/usr/bin/env python3
"""按规则从逐层提示侧占比剖面里选出抑制的候选层带。

规则（`layer_prompt_share.py` 的产物是输入）：

    prompt_share 取**含汇**口径（`aggregate.prompt_share.overall`，位置 0 计入）
    w    = round(coef * n)      带宽，主设置 coef = 0.25
    lo   = round(0.1 * n)       排除靠前约 10% 的浅层
    hi   = n - 2                排除最后一层
    band = argmin_{s in [lo, hi-w+1]} mean(prompt_share[s : s+w])

`n` 是**剖面里实际有数据的层数**，不是模型的总层数。对 Qwen3.6-27B 这类混合注意力
模型，只有全注意力层会产出注意力矩阵（64 层里的 [3,7,...,63] 共 16 层），规则就在这
16 层上套用，输出的区间端点仍是真实层号，区间内的线性注意力层在运行时不被补丁触及。
层数齐全的模型上，这与「直接按层号套规则」的结果完全相同。

用法:
    python pick_band.py layer_prompt_share/<模型>.json            # -> 16-27
    python pick_band.py <剖面>.json --coef 0.25 --verbose
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def pick_band(share: dict[int, float], coef: float = 0.25) -> tuple[int, int, list[int], float]:
    """返回 (首层, 末层, 带内层号, 带内均值)。"""
    layers = sorted(share)
    n = len(layers)
    w = round(coef * n)
    if w < 1:
        raise SystemExit(f"带宽为 0：coef={coef} 太小（可用层数 {n}）")
    lo = round(0.1 * n)
    hi = n - 2
    starts = range(lo, hi - w + 2)
    if not starts:
        raise SystemExit(f"可用层数 {n} 放不下宽度 {w} 的窗口（lo={lo}, hi={hi}）")
    mean_of = lambda s: sum(share[layers[i]] for i in range(s, s + w)) / w
    best = min(starts, key=mean_of)
    band = [layers[i] for i in range(best, best + w)]
    return band[0], band[-1], band, mean_of(best)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("profile", type=Path, help="layer_prompt_share.py 产出的 JSON")
    ap.add_argument("--coef", type=float, default=0.25, help="带宽系数，默认 0.25")
    ap.add_argument("--variant", default="prompt_share",
                    choices=["prompt_share", "sink_share", "prompt_share_nosink",
                             "prompt_share_nonsink"],
                    help="用哪一档占比，默认含汇的 prompt_share")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    d = json.loads(a.profile.read_text())
    share = {int(k): float(v) for k, v in d["aggregate"][a.variant]["overall"].items()}
    first, last, band, mean = pick_band(share, a.coef)
    if a.verbose:
        n = len(share)
        print(f"# 模型 {d.get('model')}  剖面层数 {n}/{d.get('n_layers')}  "
              f"w={round(a.coef * n)}  lo={round(0.1 * n)}  hi={n - 2}")
        print(f"# 带内层号 {band}  带内 {a.variant} 均值 {mean:.6f}")
    print(f"{first}-{last}")


if __name__ == "__main__":
    main()
