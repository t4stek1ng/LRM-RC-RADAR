# LRM-RC-RADAR

Detecting and suppressing **resource-consumption attacks** on large reasoning models
(LRMs) — prompts that drive a model into an unbounded reasoning loop until it hits the
token budget.

Both halves of the method read the same signal: **per-position attention statistics**
taken from the model's own forward pass, so detection and defence are measured on one
quantity and need no auxiliary model.

| Stage | What it does | Where |
|---|---|---|
| **Detection** | A frozen 4-class logistic classifier scores the generation at checkpoints `L ∈ {256, 512, 1024, 2048, 4096}`. The first confident attack verdict (`p_max ≥ 0.5`) triggers. Features: attention-consistency mean, prompt-side-mass slope, `log L`. | `src/detection/detector.py`, `detection_runtime.py` |
| **Suppression** | On trigger, the attention row of the current token is rewritten inside a band of layers, so the prompt-side mass returns to the level a *benign* trajectory of the same model shows at the same generation position. Per layer, self-solved, zero lag. | `src/detection/{attention_reference,row_reshape,suppression_runtime}.py` |

Chinese version: [README_zh.md](README_zh.md) — same content.

The four classes are `concise` / `productive` (benign) and `repetitive_reasoning` /
`repetitive_string` (attack). Evaluated on five models: DeepSeek-R1-Distill-Llama-8B,
DeepSeek-R1-Distill-Qwen-14B, QwQ-32B, GLM-4.7-Flash, Qwen3.6-27B.

---

## 1. Layout

```
recur_code/
  config.json                        model paths, sampling defaults
  dataset/<model>/                   inputs: attack prompt pools + benign question sets
    concise_reasoning/{GSM8k,MMLU,SimpleQA}/
    productive_reasoning/{GPQA,MMLU_Econometrics,MMLU_World_History}/
    repetitive_reasoning/{Recur,MiP}/
    repetitive_string/{LoopLLM,GCG,concat}/
  src/
    analysis/extract_layer_avg_attention_v2.py   head-mean → layer-mean attention,
                                                 chunked eager; fused per-position metrics
    analysis/step_metrics_from_attention.py      metrics from a saved (seq, seq) matrix
    detection/detector.py                        checkpoint features, τ gate, trigger policy
                                                 (also the sklearn-free base module)
    detection/replay_checkpoints.py              offline replay of the deployment protocol
    detection/detection_runtime.py               online: generate → attention → classify → trigger
    detection/attention_reference.py             benign prompt-side reference curve
    detection/row_reshape.py                     the in-row operator (prompt scale + gen align)
    detection/attn_consistency.py                the attention-consistency metric
    detection/suppression_runtime.py             decoding loop with the gate and the operator
    detection/austeer.py                         AUSteer comparison operator (imported by the above)
  visualization/reasoning_chat/compute_step_trajectory.py   per-position metrics
  script/experiments/
    run_model_detection_pipeline.py    one command: generate → attention → train → score
    build_classifier_points.py         probe sampling for training
    score_offline_detection_metrics.py offline scoring of a frozen classifier
    run_detection_online.py            online detection driver
    layer_prompt_share.py              per-layer prompt-side share profile (for band selection)
    pick_band.py                       the band rule applied to that profile
    layer_ls_closest.py                the chunked-eager capture used by the profile
    run_suppression_eval.py            paired suppression evaluation (arms differ only by intervention)
    summarize_suppression_eval.py      per-arm tables from the eval output
    analyze_suppression_v5.py          the paired read of one condition directory
  exp/detection_dataset/models/        deployment artifacts: classifier JSON + reference curves
env/                                   exact pip freezes of the three environments
```

Everything writes under a `--out` you choose; nothing in this repository is a result.

## 2. Environments

Three conda environments, because no single one satisfies all three jobs. All are
Python 3.12.9, CUDA on A100-80GB.

| Name here | File | Key versions | Used for |
|---|---|---|---|
| `rc-gen-vllm` | `env/env-gen-vllm.txt` | torch 2.7.0+cu126, transformers 4.52.3, vLLM 0.9.0, numpy 1.26.4 | generation and the suppression runtime for Llama-8B / Qwen-14B / QwQ-32B |
| `rc-gen-hf` | `env/env-gen-hf.txt` | torch 2.10.0+cu128, transformers 5.12.1, vLLM 0.19.1, numpy 2.2.6 | the same, for GLM-4.7-Flash / Qwen3.6-27B (vLLM 0.9.0 has no kernels for them) |
| `rc-analysis` | `env/env-analysis.txt` | + scikit-learn 1.8.0, joblib 1.5.3 | training and scoring the classifier |

```bash
conda create -n rc-gen-vllm python=3.12.9 -y && conda activate rc-gen-vllm
pip install -r env/env-gen-vllm.txt        # likewise for the other two
```

The generation environments deliberately have **no scikit-learn**. That is why the
classifier is exported to JSON (§4.2) and the runtime scores it with plain numpy.

## 3. Models and data

Put the HuggingFace checkpoints anywhere and point the code at them:

```bash
export RC_MODELS_ROOT=/path/to/models     # expects <root>/<model name>/
```

or pass a full path to `--model`. Edit the `models` list in `recur_code/config.json`
to the same paths.

`recur_code/dataset/<model>/` already holds the inputs — attack prompt pools and
benign question sets, one JSONL per source. They are inputs, not results: the
pipeline below regenerates every trajectory from them.

Below, `$REPO` is the repository root and all commands run from it.

```bash
export REPO=$PWD
PY_GEN=~/miniconda3/envs/rc-gen-vllm/bin/python      # or rc-gen-hf for GLM / Qwen3.6
PY_ANALYSIS=~/miniconda3/envs/rc-analysis/bin/python
M=DeepSeek-R1-Distill-Llama-8B
```

### 3.1 Quick check — no GPU, no model weights

Two self-tests exercise the operator and the runtime's multiplier logic on synthetic
rows. They need nothing but the environment:

```bash
PYTHONPATH=. $PY_GEN -m recur_code.src.detection.row_reshape
#   SELFTEST PASSED

# the runtime's version needs a reference curve, so run it after §4.3
PYTHONPATH=. $PY_GEN -m recur_code.src.detection.suppression_runtime \
    --selftest --reference recur_code/exp/detection_dataset/models/attention_reference_$M.json
#   SELFTEST PASSED
```

## 4. Running the pipeline

### 4.1 Detector — generate, extract, train, score

One driver covers four stages, each persisted under `--out` and resumable with
`--stage {generate,attention,train,detect}`:

```bash
PYTHONPATH=. $PY_ANALYSIS recur_code/script/experiments/run_model_detection_pipeline.py \
    --model $RC_MODELS_ROOT/$M \
    --dataset-root recur_code/dataset/$M \
    --out runs/detection/$M \
    --vllm-python $PY_GEN --analysis-python $PY_ANALYSIS \
    --gpu 0
```

| Stage | Does | Cost |
|---|---|---|
| `generate` | four classes × train/test trajectories; loop verdict by `_common.loop_check_for_category` decides which are kept | the expensive one — attack trajectories run to 16384 tokens |
| `attention` | one forward per trajectory for the layer-mean attention, per-position metrics computed **while the matrix is still on the GPU** | |
| `train` | 1000 probes per class (`--points-per-class`), quota inversely proportional to trajectory length, 3 features, grouped split by trajectory | seconds |
| `detect` | the frozen classifier scores every generation position of the test split | |

Products: `runs/detection/$M/model/{model.joblib,model.json}`,
`{train,test}/{records,generations,step_metrics}/<source>/`,
`test/detection_summary.{json,md}`.

### 4.2 Export the classifier for the generation environment

`model.json` is a plain-numpy export of the same linear model (scaler + coefficients).
`--stage train` writes it already; to redo it, or to check the two agree:

```bash
$PY_ANALYSIS recur_code/src/detection/detector.py \
    --bundle runs/detection/$M/model/model.joblib \
    --export runs/detection/$M/model/model.json
#   [detector] numpy vs sklearn max|Δp| = 0.000e+00 over 2000 random probes
```

The online detector and the suppression runtime both read the JSON one.

### 4.3 Benign reference curve

The prompt-side half of the intervention needs a target: *how strong should prompt-side
attention be at this generation position?* The answer is fitted from the model's own
benign trajectories — one curve per model, quadratic in generation position, with a
prompt-length term (that term is required; without it the two benign classes leave a
±0.19–0.41 residual bias).

```bash
PYTHONPATH=. $PY_ANALYSIS -m recur_code.src.detection.attention_reference \
    --model $M --out-dir recur_code/exp/detection_dataset/models
# reads runs/detection/$M/train/step_metrics/<source>/ ; --all does every model found
```

### 4.4 Suppression layer band

Which layers to act on is decided offline, per model, from a profile of per-layer
prompt-side share. Twelve trajectories from the **train** split (three per class) are
enough; the test split is never touched.

```bash
PYTHONPATH=. $PY_GEN recur_code/script/experiments/layer_prompt_share.py \
    --model $M --gpu 0 --sink-positions 0 \
    --out runs/suppression/layer_prompt_share/$M.json

python3 recur_code/script/experiments/pick_band.py \
    runs/suppression/layer_prompt_share/$M.json --verbose
#   9-16
```

The rule: with `n` layers that produced attention, width `w = round(0.25·n)`, excluding
roughly the first 10% and the last layer, take the contiguous window whose mean
prompt-side share (**sink included**) is lowest.

`--sink-positions` only affects the diagnostic `sink_share` / `nosink` variants in the
profile; the band itself is picked on `prompt_share`, which sums the whole prompt side
and therefore includes the sink either way. The measured sink positions are Llama `0`,
GLM `0,1`, QwQ `1`.

Bands obtained this way, and used for the results:

| Model | layers | band | note |
|---|---|---|---|
| DeepSeek-R1-Distill-Llama-8B | 32 | 9–16 | |
| DeepSeek-R1-Distill-Qwen-14B | 48 | 21–32 | |
| QwQ-32B | 64 | 34–49 | |
| GLM-4.7-Flash | 47 | 16–27 | |
| Qwen3.6-27B | 64 | 11–23 | hybrid attention: only layers `[3,7,…,63]` are full-attention, so the band is layers 11/15/19/23 and the rule counts those 16 |

### 4.5 Detection evaluation

Offline replay of the deployment protocol against recorded trajectories (seconds, no GPU):

```bash
PYTHONPATH=. $PY_ANALYSIS recur_code/src/detection/replay_checkpoints.py --help
```

Online, real generation and real attention:

```bash
PYTHONPATH=. $PY_GEN recur_code/script/experiments/run_detection_online.py \
    --detector runs/detection/$M/model/model.json \
    --out runs/detection_online/$M --gpu 0
```

Both sides share `detector.py`, so the numbers are comparable.

### 4.6 Suppression evaluation

Arms are loaded once and run back-to-back on the same prompt, so they differ **only** by
the intervention. `A0_off` detects without intervening and is the denominator;
`M1_dyn_thresh_t50` is the method.

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

What the flags mean:

- `--dyn-arms "thresh:0.5:<band>"` — inside the band from §4.4, a layer is suppressed at
  this step iff its attention consistency is below 0.5. Per layer, no latch, no lag.
  The arm is named `M1_dyn_thresh_t50` in the output.
- `--off-rule latch` — once the classifier gate opens it stays open. The default
  (`argmax`) re-checks every `--probe-every` (32) tokens and releases when the verdict is
  no longer an attack class.
- `--pipeline-test` takes the prompts from the test split of §4.1.
- Suppression stops at `</think>` (`--stop-after-think`, on by default).

Results land as `<out-dir>/<arm>.jsonl` plus a `manifest.json` recording every arm's
parameters. One condition (temperature + token budget) per directory — arms across
conditions are not comparable.

```bash
# per-arm tables
PYTHONPATH=. $PY_ANALYSIS recur_code/script/experiments/summarize_suppression_eval.py \
    --results runs/suppression/eval/$M/t0_cap16k

# the paired read: did it loop → did the detector catch it → did suppression break it
PYTHONPATH=. $PY_ANALYSIS recur_code/script/experiments/analyze_suppression_v5.py \
    --results runs/suppression/eval/$M/t0_cap16k \
    --base-arm A0_off --supp-arm M1_dyn_thresh_t50
```

## 5. Order of operations

```
dataset/<model>/                                     (in this repo)
   └─ 4.1 run_model_detection_pipeline.py --stage all
        ├─ model/model.json ──────────────┬─ 4.5 detection eval
        ├─ train/step_metrics/ ─ 4.3 attention_reference.py ─┐
        └─ test/records/ ────────────────────────────────────┤
      4.4 layer_prompt_share.py → pick_band.py → band ───────┴─ 4.6 run_suppression_eval.py
```

## 6. Known limits

- **The band rule is a rule, not a theory.** It picks the window of lowest prompt-side
  share; a per-model scan is what validated it. Re-derive it for any new model.
- **`repetitive_string` is the hard class.** The in-row operator redistributes mass
  within a row; where a loop lives in token identity rather than in attention placement,
  it often fails to flip the argmax.
- **The two detection features are cumulative from position 0**, so they answer "has this
  trajectory been looping" rather than "is it looping right now". A sliding-window
  consistency metric is the known gap.
- **Qwen3.6-27B's reference curve fits poorly** (R² 0.394 against 0.957–0.970 for the
  others): its prompt-side mass barely grows with generation position, and its attack
  prompts are 4–9× longer than any benign trajectory, so the length term sits pinned at
  its fitted bound.
- Suppression is evaluated at temperature 0 with a 16384-token budget; a different
  condition is a different experiment.
