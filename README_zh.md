# LRM-RC-RADAR

面向大型推理模型（LRM）的**资源消耗攻击**检测与抑制——这类提示会把模型推进一个不收敛的
推理循环，一直烧到 token 上限。

两个阶段读同一个信号：模型自己前向里的**逐位置注意力统计量**。检测与抑制瞄同一个量，
不需要额外的模型。

| 阶段 | 做什么 | 代码 |
|---|---|---|
| **检测** | 冻结的四类 logistic 分类器在检查点 `L ∈ {256, 512, 1024, 2048, 4096}` 各判一次，首次确信判为攻击（`p_max ≥ 0.5`）即触发。三个特征：注意力一致性均值、提示侧强度斜率、`log L`。 | `src/detection/detector.py`、`detection_runtime.py` |
| **抑制** | 触发后，在一段层带内把当前 token 的注意力行重写，使提示侧质量回到**同模型良性轨迹在同一生成位置**的水平。逐层自解自施，零滞后。 | `src/detection/{attention_reference,row_reshape,suppression_runtime}.py` |

四个类别是 `concise` / `productive`（良性）与 `repetitive_reasoning` /
`repetitive_string`（攻击）。在五个模型上评测：DeepSeek-R1-Distill-Llama-8B、
DeepSeek-R1-Distill-Qwen-14B、QwQ-32B、GLM-4.7-Flash、Qwen3.6-27B。

英文版见 [README.md](README.md)，两份内容一致。

---

## 1. 目录

```
recur_code/
  config.json                        模型路径与采样默认值
  dataset/<模型>/                    输入：攻击提示池 + 良性题目集
  src/
    analysis/extract_layer_avg_attention_v2.py   层平均注意力（分块 eager）+ 融合的逐位置指标
    analysis/step_metrics_from_attention.py      从已存的 (seq, seq) 矩阵算指标
    detection/detector.py                        检查点特征、τ 门、触发策略（也是无 sklearn 的基础模块）
    detection/replay_checkpoints.py              部署协议的离线回放
    detection/detection_runtime.py               在线：生成 → 注意力 → 分类 → 触发
    detection/attention_reference.py             良性提示侧参考曲线
    detection/row_reshape.py                     行内算子（提示侧缩放 + 生成侧对齐）
    detection/attn_consistency.py                注意力一致性指标
    detection/suppression_runtime.py             带门控与算子的解码回路
    detection/austeer.py                         AUSteer 对比算子（被上面 import）
  visualization/reasoning_chat/compute_step_trajectory.py   逐位置指标
  script/experiments/
    run_model_detection_pipeline.py    一键：生成 → 注意力 → 训练 → 打分
    build_classifier_points.py         训练取点
    score_offline_detection_metrics.py 冻结分类器的离线打分
    run_detection_online.py            在线检测 driver
    layer_prompt_share.py              逐层提示侧占比剖面（选层带用）
    pick_band.py                       在剖面上套选带规则
    layer_ls_closest.py                剖面所用的分块 eager 采集
    run_suppression_eval.py            配对抑制评测（臂之间只差干预方式）
    summarize_suppression_eval.py      逐臂表格
    analyze_suppression_v5.py          一个条件目录的配对读法
  exp/detection_dataset/models/        部署工件：分类器 JSON + 参考曲线
env/                                   三套环境的精确 pip freeze
```

产物都写到你指定的 `--out` 下；这个仓库里没有任何结果文件。

## 2. 环境

三套 conda 环境，因为没有一套能同时满足三件事。都是 Python 3.12.9，A100-80GB。

| 这里的名字 | 文件 | 关键版本 | 用途 |
|---|---|---|---|
| `rc-gen-vllm` | `env/env-gen-vllm.txt` | torch 2.7.0+cu126、transformers 4.52.3、vLLM 0.9.0、numpy 1.26.4 | Llama-8B / Qwen-14B / QwQ-32B 的生成与抑制运行时 |
| `rc-gen-hf` | `env/env-gen-hf.txt` | torch 2.10.0+cu128、transformers 5.12.1、vLLM 0.19.1、numpy 2.2.6 | GLM-4.7-Flash / Qwen3.6-27B 同上（vLLM 0.9.0 没有它们的 kernel） |
| `rc-analysis` | `env/env-analysis.txt` | 另加 scikit-learn 1.8.0、joblib 1.5.3 | 训练与打分分类器 |

```bash
conda create -n rc-gen-vllm python=3.12.9 -y && conda activate rc-gen-vllm
pip install -r env/env-gen-vllm.txt        # 另外两套同理
```

两个生成环境**故意不装 scikit-learn**。所以分类器要导出成 JSON（§4.2），运行时用纯 numpy 打分。

## 3. 模型与数据

权重放哪都行，指给代码即可：

```bash
export RC_MODELS_ROOT=/path/to/models     # 期望 <root>/<模型名>/
```

或者给 `--model` 传完整路径。`recur_code/config.json` 里的 `models` 改成同样的路径。

`recur_code/dataset/<模型>/` 里已经是输入——攻击提示池与良性题目集，每个来源一个 JSONL。
它们是输入不是结果：下面的流程从它们重新生成每一条轨迹。

下文 `$REPO` 是仓库根目录，所有命令都在其中执行。

```bash
export REPO=$PWD
PY_GEN=~/miniconda3/envs/rc-gen-vllm/bin/python      # GLM / Qwen3.6 换 rc-gen-hf
PY_ANALYSIS=~/miniconda3/envs/rc-analysis/bin/python
M=DeepSeek-R1-Distill-Llama-8B
```

### 3.1 快速自检——不用 GPU，不用权重

两个自测在合成的注意力行上跑算子与运行时的乘子逻辑，除环境外什么都不需要：

```bash
PYTHONPATH=. $PY_GEN -m recur_code.src.detection.row_reshape
#   SELFTEST PASSED

# 运行时那个要参考曲线，放在 §4.3 之后跑
PYTHONPATH=. $PY_GEN -m recur_code.src.detection.suppression_runtime \
    --selftest --reference recur_code/exp/detection_dataset/models/attention_reference_$M.json
#   SELFTEST PASSED
```

## 4. 跑流程

### 4.1 检测器——生成、提取、训练、打分

一个驱动脚本覆盖四段，每段都持久化到 `--out` 下，可用
`--stage {generate,attention,train,detect}` 断点续跑：

```bash
PYTHONPATH=. $PY_ANALYSIS recur_code/script/experiments/run_model_detection_pipeline.py \
    --model $RC_MODELS_ROOT/$M \
    --dataset-root recur_code/dataset/$M \
    --out runs/detection/$M \
    --vllm-python $PY_GEN --analysis-python $PY_ANALYSIS \
    --gpu 0
```

| 段 | 做什么 | 开销 |
|---|---|---|
| `generate` | 四类各生成 train/test 轨迹，按 `_common.loop_check_for_category` 判成环后选样 | 最贵的一段——攻击轨迹要跑满 16384 token |
| `attention` | 每条轨迹前向一次拿层平均注意力，**趁矩阵还在显存里**直接算完逐位置指标 | |
| `train` | 每类 1000 个探针（`--points-per-class`），名额按轨迹长度反比分配，三特征，按轨迹分组切分 | 秒级 |
| `detect` | 冻结分类器给 test 的全部生成位置打分 | |

产物：`runs/detection/$M/model/{model.joblib,model.json}`、
`{train,test}/{records,generations,step_metrics}/<数据集>/`、
`test/detection_summary.{json,md}`。

### 4.2 把分类器导出给生成环境

`model.json` 是同一个线性模型的纯 numpy 导出（scaler + 系数）。`--stage train` 已经写了一份；
要重做或核对两者一致：

```bash
$PY_ANALYSIS recur_code/src/detection/detector.py \
    --bundle runs/detection/$M/model/model.joblib \
    --export runs/detection/$M/model/model.json
#   [detector] numpy vs sklearn max|Δp| = 0.000e+00 over 2000 random probes
```

在线检测与抑制运行时读的都是 JSON 那份。

### 4.3 良性参考曲线

抑制的提示侧那一半需要一个目标：**这一步的提示侧注意力应该有多强**。答案由模型自己的良性
轨迹拟合而来——每个模型一条曲线，对生成位置二次，并带提示长度项（这一项是必需的，去掉它
两类良性会留下 ±0.19–0.41 的残差偏置）。

```bash
PYTHONPATH=. $PY_ANALYSIS -m recur_code.src.detection.attention_reference \
    --model $M --out-dir recur_code/exp/detection_dataset/models
# 读 runs/detection/$M/train/step_metrics/<数据集>/；--all 对找到的每个模型各做一份
```

### 4.4 抑制层带

作用在哪些层，是离线按模型定的，输入是逐层提示侧占比剖面。**train** split 里 12 条轨迹
（每类 3 条）就够；test 全程不碰。

```bash
PYTHONPATH=. $PY_GEN recur_code/script/experiments/layer_prompt_share.py \
    --model $M --gpu 0 --sink-positions 0 \
    --out runs/suppression/layer_prompt_share/$M.json

python3 recur_code/script/experiments/pick_band.py \
    runs/suppression/layer_prompt_share/$M.json --verbose
#   9-16
```

规则：设产出了注意力的层数为 `n`，宽度 `w = round(0.25·n)`，排除靠前约 10% 与最后一层，
取**含汇**提示侧占比均值最低的连续窗口。

`--sink-positions` 只影响剖面里诊断用的 `sink_share` / `nosink` 两档；选带用的
`prompt_share` 对整段提示侧求和，无论如何都含汇。实测的汇位置是 Llama `0`、GLM `0,1`、QwQ `1`。

按此得到、也是结果所用的层带：

| 模型 | 层数 | 层带 | 备注 |
|---|---|---|---|
| DeepSeek-R1-Distill-Llama-8B | 32 | 9–16 | |
| DeepSeek-R1-Distill-Qwen-14B | 48 | 21–32 | |
| QwQ-32B | 64 | 34–49 | |
| GLM-4.7-Flash | 47 | 16–27 | |
| Qwen3.6-27B | 64 | 11–23 | 混合注意力：只有 `[3,7,…,63]` 是全注意力层，所以层带是第 11/15/19/23 层，规则数的是这 16 层 |

### 4.5 检测评测

用已录轨迹离线回放部署协议（秒级，不用 GPU）：

```bash
PYTHONPATH=. $PY_ANALYSIS recur_code/src/detection/replay_checkpoints.py --help
```

在线，真生成、真注意力：

```bash
PYTHONPATH=. $PY_GEN recur_code/script/experiments/run_detection_online.py \
    --detector runs/detection/$M/model/model.json \
    --out runs/detection_online/$M --gpu 0
```

两侧共用 `detector.py`，数字才可比。

### 4.6 抑制评测

一次载入模型，同一提示上把各臂背靠背跑完，臂之间**只差干预方式**。`A0_off` 只检测不干预，
是分母；`M1_dyn_thresh_t50` 是本方法。

```bash
PYTHONPATH=. $PY_GEN recur_code/script/experiments/run_suppression_eval.py \
    --model $RC_MODELS_ROOT/$M \
    --detector runs/detection/$M/model/model.json \
    --reference recur_code/exp/detection_dataset/models/attention_reference_$M.json \
    --pipeline-test runs/detection/$M \
    --arms A0_off --dyn-arms "thresh:0.5:9-16" --off-rule latch \
    --temperature 0 --max-new-tokens 16384 --gpu 0 \
    --out-dir runs/suppression/eval/$M/t0_cap16k
```

各参数的意思：

- `--dyn-arms "thresh:0.5:<层带>"`——在 §4.4 的层带内，某层这一步的注意力一致性低于 0.5
  才压这一层。逐层独立，无滞后。输出里这条臂叫 `M1_dyn_thresh_t50`。
- `--off-rule latch`——分类器门一旦打开就锁死。默认的 `argmax` 每 `--probe-every`（32）
  个 token 复检一次，类别不再是攻击类就解除。
- `--pipeline-test` 从 §4.1 的 test split 取提示。
- 抑制在 `</think>` 处停止（`--stop-after-think`，默认开）。

结果落成 `<out-dir>/<臂>.jsonl` 加一份记录逐臂参数的 `manifest.json`。**一个条件一个目录**
（温度 + 生成上限）——跨条件的臂不可比。

```bash
# 逐臂表格
PYTHONPATH=. $PY_ANALYSIS recur_code/script/experiments/summarize_suppression_eval.py \
    --results runs/suppression/eval/$M/t0_cap16k

# 配对读法：成环了吗 → 检测对了吗 → 抑制打断了吗
PYTHONPATH=. $PY_ANALYSIS recur_code/script/experiments/analyze_suppression_v5.py \
    --results runs/suppression/eval/$M/t0_cap16k \
    --base-arm A0_off --supp-arm M1_dyn_thresh_t50
```

## 5. 先后顺序

```
dataset/<模型>/                                      （仓库自带）
   └─ 4.1 run_model_detection_pipeline.py --stage all
        ├─ model/model.json ──────────────┬─ 4.5 检测评测
        ├─ train/step_metrics/ ─ 4.3 attention_reference.py ─┐
        └─ test/records/ ────────────────────────────────────┤
      4.4 layer_prompt_share.py → pick_band.py → 层带 ───────┴─ 4.6 run_suppression_eval.py
```

## 6. 已知边界

- **选带是一条规则，不是一个定理。** 它取提示侧占比最低的窗口，逐层扫描是它的验证方式。
  换新模型要重新推导。
- **`repetitive_string` 是难的那一类。** 行内算子只在一行内部重新分配质量；当循环落在
  token 身份而不是注意力落点上时，往往翻不动 argmax。
- **两个检测特征都是从位置 0 起的累计量**，回答的是「这条轨迹是否一直在循环」而不是
  「此刻是否在循环」。滑窗版一致性指标是已知缺口。
- **Qwen3.6-27B 的参考曲线拟合很差**（R² 0.394，其余模型 0.957–0.970）：它的提示侧质量
  几乎不随生成位置增长，而攻击提示比任何良性轨迹长 4–9 倍，长度项恒被夹在拟合上界。
- 抑制在温度 0、上限 16384 下评测；换条件就是另一个实验。
