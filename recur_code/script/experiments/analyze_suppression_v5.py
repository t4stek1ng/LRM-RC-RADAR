"""v5 评测的配对读法：成环了吗 → 检测对了吗 → 抑制打断了吗。

三张表，按提问顺序回答三个问题（只读 `run_suppression_eval.py` 写出的 JSONL，
可随时对**部分**结果运行）：

  表 1  基线（A0_off，只检测不抑制）：每条攻击提示到底成没成环
  表 2  分类器的逐检查点判断 × 轨迹最终结果（**不是**混淆矩阵，见口径 2）
  表 3  对「基线成环且被正确检出」的提示，抑制臂配对比较，判断是否打断

## 三处口径必须显式说明，否则表会被读错

**1. 「成环」不是一个布尔值。** `_common.loop_check_for_category` 要求尾部严格
周期（段落逐字节相同 + 周期恒定），实测会漏掉「更紧但不规则」的循环
（`suppression_perlayer_design.md` §13.4）。所以基线分三档：

  - **成环**：loop_check 判成环；
  - **疑似成环**：loop_check 未判成环，但跑满生成上限且没有自然结束——模型没停
    下来，只是重复得不够规整；
  - **未成环**：自然结束（EOS）。

汇总里「成环」= 前两档之和，单独列出各自数量。

**2. 判对的定义是「现在或最终是否循环」，所以提前判定算判对。** 分类器（训练标签取自
实测成环轨迹）回答的是这条轨迹**现在或最终**会不会循环，因此**最终成环的样本无论在
哪个检查点被判出来都算正确**——早于成环发生不扣分。反过来，「触发后仍自然结束」只是
误报的**上界**：若该轨迹触发时确实正在打转、随后自己走出来，那仍算判对。要把这一格
拆开需要注意力侧的**局部（滑窗）**指标，而现有特征全是从生成位置 0 起的累计量，判不了
「此刻是否在循环」——所以脚本只报上界，不擅自判它是误报。

**2b. 「检测触发」用的是部署协议，不是最后一次判断。** 门关着时分类器只在部署
检查点（256/512/1024/2048/4096 的总长）上跑，**首次确信判攻击即触发**。所以触发
与否要从 `probes` 里按「任一部署检查点上 argmax ∈ 攻击类且 p ≥ τ」重算，而不是读
`final_pred`（那是最后一次判断，可能早已被前面的触发推翻）。基线臂（mode=off）
门不会真的开，但探针记录一模一样，所以它就是「没有干预时检测会不会触发」的答案。

**3. 「打断」分三档，不能只看 loop_check 翻转。** 判定翻转 ≠ 打断（README 已
就此加注）：

  - **干净打断**：抑制臂自然结束（EOS）且未判成环——模型真的走完并停下；
  - **判定翻转**：未判成环但仍跑满上限——loop_check 说不循环了，模型却还在吐；
  - **未打断**：仍判成环。

**成环判定两段都查**（`is_looped`）：干预会把循环从思考段挤进答案段，只读
`looped`（对 repetitive_reasoning 只查思考段）会把搬家记成打断。

同时给出生成 token 数变化与三元组多样度变化，用来看第二档到底是好转还是换了种
退化方式。

    /root/miniconda3/envs/Recur/bin/python \\
        recur_code/script/experiments/analyze_suppression_v5.py \\
        --results recur_code/exp/suppression/eval/DeepSeek-R1-Distill-Llama-8B/t0.0_cap16k
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ATTACK = {"repetitive_reasoning", "repetitive_string"}
CAT_CN = {"repetitive_reasoning": "循环思考", "repetitive_string": "循环字符"}


def load_arm(d: Path, arm: str) -> dict:
    """读该条件目录下某条臂的攻击集记录。

    同时收分片子目录（多卡按样本分片时每片写自己的目录，跑完才合并），
    这样合并前也能对部分结果出表。分片按 id 互斥，不会互相覆盖。
    """
    out = {}
    for f in sorted(list(d.glob(f"{arm}.jsonl")) + list(d.glob(f"*/{arm}.jsonl"))):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("set") == "S1_attack_loop":
                out[r["id"]] = r
    return out


def is_looped(r: dict) -> bool:
    """成环判定：**思考段与答案段都算**。

    干预会把循环**搬家**：思考段被提前结束（模型吐出 `</think>`）、循环整段挪进
    答案段，而 `loop_check_for_category` 对 repetitive_reasoning 只查思考段，于是
    报 `looped=False`。按类别选段是建数据集时的正确口径，评测抑制时必须两段都查
    ——`run_suppression_eval.py` 正是为此额外算了 `looped_any`（见该文件
    `r["looped_any"] = bool(looped) or any(spans.values())` 处的注释）。本模块此前
    只读 `looped`，会把「把循环挪个地方」记成打断。老结果里没有 `looped_any`
    字段时退回 `looped`。
    """
    v = r.get("looped_any")
    return bool(r.get("looped") if v is None else v)


def loop_state(r: dict) -> str:
    """成环 / 疑似成环 / 未成环——见模块文档口径 1。"""
    if is_looped(r):
        return "成环"
    if r.get("hit_cap") and not r.get("stopped_naturally"):
        return "疑似成环"
    return "未成环"


def triggered(r: dict) -> tuple[bool, int | None, str | None, float | None]:
    """按部署协议重算「首次确信判攻击」——见模块文档口径 2。"""
    tau = r.get("tau_prob", 0.5)
    for p in r.get("probes", []):
        if p["pred"] in ATTACK and p["prob"] >= tau:
            return True, p["L"], p["pred"], p["prob"]
    return False, None, None, None


def break_state(r: dict) -> str:
    """干净打断 / 判定翻转 / 未打断——见模块文档口径 3。"""
    if is_looped(r):          # 含「循环挪到答案段」，见 is_looped
        return "未打断"
    if r.get("stopped_naturally"):
        return "干净打断"
    return "判定翻转"


def pct(n: int, d: int) -> str:
    return f"{n}/{d}" + (f" ({n / d * 100:.0f}%)" if d else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, help="条件目录")
    ap.add_argument("--base-arm", default="A0_off")
    ap.add_argument("--supp-arm", default="M1_suppress")
    ap.add_argument("--per-sample", action="store_true",
                    help="逐条打印表 1（默认只打印成环与被检出的那些）")
    a = ap.parse_args()

    d = Path(a.results)
    base, supp = load_arm(d, a.base_arm), load_arm(d, a.supp_arm)
    if not base:
        raise SystemExit(f"{d/(a.base_arm + '.jsonl')} 里没有 S1 攻击样本")
    ids = sorted(base)

    # ---------------------------------------------------------------- 表 1
    print(f"\n表 1 基线（{a.base_arm}，只检测不抑制）：{len(ids)} 条攻击提示\n")
    print(f"{'编号':>6}{'类别':>10}{'基线结果':>10}{'生成token':>10}"
          f"{'自然结束':>10}{'三元多样度':>11}{'检测触发':>10}{'触发总长':>10}"
          f"{'判成':>12}")
    rows = []
    for i in ids:
        r = base[i]
        st = loop_state(r)
        tr, L, pred, prob = triggered(r)
        rows.append((i, r.get("category"), st, tr, L, pred))
        if not a.per_sample and st == "未成环" and not tr:
            continue
        print(f"{i:>6}{CAT_CN.get(r.get('category'), '?'):>10}{st:>10}"
              f"{r['gen_tokens']:>10}{str(r['stopped_naturally']):>10}"
              f"{r.get('uniq3', float('nan')):>11.3f}"
              f"{('是' if tr else '否'):>10}{str(L or '-'):>10}"
              f"{CAT_CN.get(pred, pred or '-'):>12}")
    if not a.per_sample:
        print("  （只列出成环或被检测触发的；全部逐条见 --per-sample）")

    for cat in ("repetitive_reasoning", "repetitive_string", None):
        sel = [x for x in rows if cat is None or x[1] == cat]
        if not sel:
            continue
        name = CAT_CN.get(cat, "合计")
        looped = [x for x in sel if x[2] != "未成环"]
        print(f"\n  {name}：{len(sel)} 条 → 成环 "
              f"{sum(1 for x in sel if x[2] == '成环')} 条、疑似成环 "
              f"{sum(1 for x in sel if x[2] == '疑似成环')} 条、未成环 "
              f"{sum(1 for x in sel if x[2] == '未成环')} 条")
        print(f"    检测：最终成环的样本里触发 "
              f"{pct(sum(1 for x in looped if x[3]), len(looped))}；"
              f"最终自然结束的样本里触发 "
              f"{pct(sum(1 for x in sel if x[2] == '未成环' and x[3]), sum(1 for x in sel if x[2] == '未成环'))}")

    # ---------------------------------------------------------------- 表 2
    print(f"\n\n表 2 逐检查点判断 × 轨迹最终结果（不是混淆矩阵，见脚本文档口径 2）\n")
    tp = sum(1 for x in rows if x[2] != "未成环" and x[3])
    fn = sum(1 for x in rows if x[2] != "未成环" and not x[3])
    fp = sum(1 for x in rows if x[2] == "未成环" and x[3])
    tn = sum(1 for x in rows if x[2] == "未成环" and not x[3])
    print(f"{'':>16}{'检查点判循环':>14}{'未判循环':>10}")
    print(f"{'最终成环':>16}{tp:>14}{fn:>10}")
    print(f"{'最终自然结束':>16}{fp:>14}{tn:>10}")
    print(f"\n  最终成环且触发（提前判定也算判对）：{pct(tp, tp + fn)}    "
          f"最终自然结束却触发：{pct(fp, fp + tn)}"
          f"  —— 后者是误报的上界，见模块文档口径 2")
    cls_ok = sum(1 for x in rows if x[3] and x[5] == x[1])
    cls_tr = sum(1 for x in rows if x[3])
    print(f"  触发时类别判对（与攻击集类别一致）：{pct(cls_ok, cls_tr)}"
          f"  —— 类别决定 v5 用哪条参考曲线，判错方向就反了")

    # ---------------------------------------------------------------- 表 3
    if not supp:
        print(f"\n\n表 3：{a.supp_arm} 还没有结果。\n")
        return 0
    targets = [x for x in rows if x[2] != "未成环" and x[3] and x[0] in supp]
    print(f"\n\n表 3 抑制效果（{a.supp_arm} vs {a.base_arm}）："
          f"「基线成环且被正确检出」共 {len([x for x in rows if x[2] != '未成环' and x[3]])} 条，"
          f"其中抑制臂已完成 {len(targets)} 条\n")
    print(f"{'编号':>6}{'类别':>10}{'基线':>10}{'抑制后':>10}{'打断判定':>12}"
          f"{'生成token':>18}{'三元多样度':>16}{'开门总长':>10}{'抑制步占比':>11}"
          f"{'常数c中位':>11}")
    tally: dict[str, int] = {}
    for i, cat, st, _tr, _L, _pred in targets:
        b, s = base[i], supp[i]
        bs = break_state(s)
        tally[bs] = tally.get(bs, 0) + 1
        gen = f"{b['gen_tokens']}→{s['gen_tokens']}"
        uniq = f"{b.get('uniq3', 0):.3f}→{s.get('uniq3', 0):.3f}"
        print(f"{i:>6}{CAT_CN.get(cat, '?'):>10}{st:>10}{loop_state(s):>10}"
              f"{bs:>12}{gen:>18}{uniq:>16}"
              f"{str(s.get('gate_open_L') or '-'):>10}"
              f"{str(s.get('suppressed_frac') or '-'):>11}"
              f"{str(s.get('c_median') or '-'):>11}")
    print()
    for k in ("干净打断", "判定翻转", "未打断"):
        print(f"  {k}：{pct(tally.get(k, 0), len(targets))}")
    tot_b = sum(base[i]['gen_tokens'] for i, *_ in targets)
    tot_s = sum(supp[i]['gen_tokens'] for i, *_ in targets)
    print(f"  总生成 token：{tot_b} → {tot_s}"
          f"（{(tot_s - tot_b) / tot_b * 100:+.1f}%）" if tot_b else "")
    cap_b = sum(1 for i, *_ in targets if base[i].get('hit_cap'))
    cap_s = sum(1 for i, *_ in targets if supp[i].get('hit_cap'))
    print(f"  跑满上限：{cap_b} → {cap_s} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
