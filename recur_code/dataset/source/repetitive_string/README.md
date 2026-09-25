# repetitive_string (第 4 类 prompt) — LoopLLM 攻击的跨模型迁移测试

第 4 类攻击 prompt：用 **LoopLLM**（对抗后缀优化）在 Llama 上生成的
「重复字符串」攻击，测试其在全部 6 个目标推理模型上诱发 *不终止的重复输出循环*
的可迁移性，并记录成功成环的思考/输出轨迹。

## 文件
- `prompts.jsonl` — **第 1 批** 51 条攻击 prompt（base 指令 + 对抗后缀）。字段：
  `id, base_prompt, adv_suffix, prompt, src_best_iter, src_success_rate, src_avg_len`。
  **与 repetitive_reasoning 不同：这是一套「共享」prompt**——所有模型喂同一批 51 条，
  对抗后缀不是逐模型定制的。
  **这是该批 prompt 的唯一真源**（上游优化日志已删，见下），不可再生，勿删。
- `loop_samples_prompts.jsonl` — **第 2 批** 22 条攻击 prompt，由
  `dataset/LoopLLM/loop_samples.json` 转换而来（该批后缀是在**推理模型**上优化的，
  与第 1 批不同）。供 `script/experiments/loopsample_repstring{,_hf}.py` 使用。
- `summarize.py` — 从 `reasoning_trajectory/*/from_loopllm/*.jsonl` 重建
  `realized_trajectories.jsonl` + `stats.json`。

### 已清理的冗余文件（2026-07-20）
- `dataset/LoopLLM/res_*.json`（51 个，984 KB）— LoopLLM 后缀优化的逐轮迭代日志
  （每轮的 `current_losses / answer / success_rate / avg_len / time`，共 336 条）。
  下游从未使用；其唯一有价值的产物 `prompts.jsonl` 已物化保留，故整体删除。
  当时的提取规则：每个 res 文件取 success_rate 最优（平手比 avg_len）的迭代的
  `adv_prompt`。
- `realized_trajectories.jsonl`（32 条成环轨迹）、`stats.json`（逐模型统计）—
  纯派生物，`summarize.py` 重跑即可复现。

第 1 批中 32 次成环对应的 22 条唯一 prompt 另已收录进 `dataset/total/total.jsonl`
（带 `model` 标注，`type=repetitive_string`）。

## 登记表与提示池的分离（2026-09-09）

原先 `dataset/<模型>/repetitive_string/LoopLLM/prompts.jsonl` 是
`dataset/source/total/total.jsonl` 中 `type == "repetitive_string"` 切片的逐字节副本，
5 个模型目录下完全相同。它的粒度是**一行一个（提示，成环模型）对**——同一条提示会带
不同 `model` 出现多次。这个形状适合记录迁移测试结果，但**不适合当提示池**：
`run_model_detection_pipeline.py` 把每一行都喂给模型，一条登记了 4 个模型的提示会被跑 4 遍；
而且 `model` 列容易让人误以为这份文件是逐模型定制的（其实 5 份完全一样）。

现在两者分开：

| | 位置 | 粒度 | 字段 |
|---|---|---|---|
| 登记表 | `source/total/total.jsonl` 的 repetitive_string 切片 | 一行一个（提示，成环模型）对，98 行 | 含 `model` |
| 提示池 | `<模型>/repetitive_string/LoopLLM/prompts.jsonl` | **一行一条提示，74 条** | 无 `model` |

登记表保持原样，它是「哪个模型在哪条提示上成过环」的唯一记录。

### 两批 prompt 都已并入（`merge_loopllm_batch2_prompts.py`）

合并前登记表 50 行 / 35 条唯一提示，只收录了 2026-07-11 迁移测试中成过环的那部分。
第 1 批余下 29 条、第 2 批（`pool == "loopllm22"`）22 条都只存在于
`source/attack_resample/<模型>/repetitive_string.jsonl`，检测实验却已经在用它们生成的轨迹，
等于**数据集复现不出自己的实验**。

合并后登记表 98 行，`total.jsonl` 347 → 395 行；提示池 74 条，**第 1 批 51/51、
第 2 批 22/22 全覆盖**。构成：原有 35 + 第 1 批新增 29 + 第 2 批新增 10。

- `model` 取**实际观测到成环的模型**，来源 `exp/detection/archive_import`（去重后的全部旧循环轨迹）；
  从未观测到成环的提示仍并入，`model` 写 `null`——它们是合法的待试攻击提示，
  编一个没见过的成环模型比留空更糟。
- 判重与 `prompt_sha1` 一律用**归一化空白后的 sha1**（`assign_uids.py` 定义的全局提示键），
  否则同一条提示的两种写法会各占一行。
- 两个脚本都可重跑，备份 `.bak_preloopllm22` / `.bak_prepool`。

### 去重：74 = 98 − 24

删掉的 24 行里，21 行是同一条提示登记了多个成环模型，另外 3 行是**同一条提示的两种写法**：
base 指令与对抗后缀之间的拼接空白不同（`base + "  " + 后缀` 对 `base + " " + 后缀`）。
按项目的提示键（归一化空白）它们本就是一条。

保留哪种写法，按**轨迹里实际用过的那种**定——那才是真正喂给模型的字符串，
这样记录还能按原文对上数据集行。有 7 条提示在现有划分里两种写法都出现过，
只能保一种，于是 **13 条记录与提示池差一个空格**；按归一化提示比对则 **80 条循环字符串记录全部命中，无一落空**。

### 指针刷新

去重让行号整体前移，所以 `refresh_repetitive_string_source_pointers.py` 重算了所有
循环字符串记录的 `source_index`（提示在文件里的行号）与 `source_id`（该行的 id），
按归一化提示匹配。归档记录原来的 id 仍在 `archive_source.src_id` / `dataset_match` 里。
良性两类不动：它们索引的是数据集文件的切片（`rows[:20]` / `rows[-25:]`），`source_index` 含义不同。

## 生成方式
`script/experiments/build_repetitive_string_from_loopllm.py`（vLLM/Recur，老 3 模型）
及其 HF 孪生 `_hf.py`（Recur2，新 3 模型）：每条 prompt 喂给模型，temp=0.3、
max_tokens=8192、最多重试 3 次，用 `_common.loop_check` 对 **整段生成**（thinking+output）
判环，命中即记录首个成环轨迹。输出落在
`reasoning_trajectory/<model>/from_loopllm/repetitive_string.jsonl`。

## 结果（2026-07-11，51 prompts × 6 models = 306 次生成）

| 模型 | 成环 | 迁移率 | 环在 output / thinking |
|---|--:|--:|--|
| GLM-4.7-Flash | **15** | 0.29 | 0 / 15 |
| DeepSeek-R1-Distill-Llama-8B | 9 | 0.18 | 4 / 5 |
| QwQ-32B | 4 | 0.08 | 3 / 1 |
| DeepSeek-R1-Distill-Qwen-14B | 3 | 0.06 | 2 / 1 |
| Qwen3.6-27B | 1 | 0.02 | 0 / 1 |
| Gemma-4-31B-it | 0 | 0.00 | 0 / 0 |
| **合计** | **32** | — | 9 / 23 |

## 关键发现
1. **对抗后缀对推理模型的可迁移性远弱于 repetitive_reasoning，且高度模型相关**：
   GLM(29%) / Llama(18%) 明显，Qwen 系(6%/2%) 几乎免疫，Gemma(0%) 完全免疫。
   多数情况下模型会先「思考」消化对抗后缀再正常作答。
2. **攻击以两种形态成环（都撑到 8192 token 上限、消耗算力）**：
   - **output 内字面重复**（Llama/QwQ/Qwen-14B）：答案退化成 `* * * *` 到 ~7000–8000 token
     —— 真正的 repetitive_string 特征。
   - **thinking 内推理循环**（GLM 全部 15 条）：模型卡在「试图破译对抗后缀」的
     反思里反复循环（如 "Could it be Prosthetic?…"）到 cap —— 形态更接近 repetitive_reasoning。
   `loop_span` 字段记录了每条属于哪种，供检测 pipeline 的轴 B（思考/非思考）分析使用。
