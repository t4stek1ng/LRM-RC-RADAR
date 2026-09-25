"""End-to-end online detection experiment on DeepSeek-R1-Distill-Llama-8B.

For every selected trajectory of the offline detection set we take its ORIGINAL
prompt, re-run it live through `OnlineLoopDetector` (generation + checkpoint
attention + classifier), and record what the deployed policy would have done.

Why re-run instead of replaying cached features: the offline numbers are
computed on vLLM-generated, loop-truncated text; deployment sees a live HF
stream. This script is the only thing that measures the gap.

Prompt recovery (llama-8B):
  - repetitive_reasoning : reasoning_trajectory/<m>/from_total/repetitive_reasoning.jsonl
  - repetitive_string    : reasoning_trajectory/<m>/from_loopllm/repetitive_string.jsonl
  - concise / productive : the legacy per-sample token dumps in
    exp/stop_pattern/tokens/<cat>_<id>.json — the source jsonl for those two
    llama-8B classes was removed in the 2026-07 cleanup, but the dumps hold the
    exact prompt tokens, so the prompt is reconstructed by decoding
    ``tokens[:prompt_len]`` and stripping the chat template.
Every recovered prompt is re-templated and its token length is checked against
the trajectory's `prompt_len`; a mismatch aborts that sample rather than
silently evaluating a different prompt.

Ground truth is reported two ways, because an attack prompt is not guaranteed to
loop on a fresh run: the source class, and `loop_check` run post-hoc on the live
generation (repetitive_string is checked on the full text, repetitive_reasoning
on the thinking — see _common.loop_check_for_category). Attack prompts that did
not loop online are the strongest negatives available: same prompt family,
opposite label.

Run (Recur env; check nvidia-smi first — this needs one free GPU):
    PYTHONPATH=. /root/miniconda3/envs/Recur/bin/python \
        recur_code/script/experiments/run_detection_online.py \
        --gpu 1 --n-concise 10 --n-productive 10 \
        --out recur_code/exp/detection_dataset/replay/online_llama8b.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "recur_code" / "src" / "detection"))
sys.path.insert(0, str(REPO_ROOT / "recur_code" / "script" / "experiments"))

from _common import loop_check_for_category  # noqa: E402

MODEL_SHORT = "DeepSeek-R1-Distill-Llama-8B"
MODEL_PATH = str(Path(os.environ.get("RC_MODELS_ROOT", "/root/project/models")) / "DeepSeek-R1-Distill-Llama-8B")
CATEGORIES = ["concise_reasoning", "productive_reflection",
              "repetitive_reasoning", "repetitive_string"]
LABEL_OF = {"concise_reasoning": "concise", "productive_reflection": "productive",
            "repetitive_reasoning": "repetitive_reasoning",
            "repetitive_string": "repetitive_string"}

# Chat-template wrapper of the R1-distill family, stripped when reconstructing
# a prompt from a token dump.
TPL_PREFIX = "<｜begin▁of▁sentence｜><｜User｜>"
TPL_SUFFIX = "<｜Assistant｜><think>\n"


def load_prompt_table(repo: Path) -> dict[tuple[str, int], str]:
    """(category, id) -> original user prompt, for all four llama-8B classes."""
    table: dict[tuple[str, int], str] = {}
    traj = repo / "recur_code/reasoning_trajectory" / MODEL_SHORT
    for cat, path in [
        ("repetitive_reasoning", traj / "from_total/repetitive_reasoning.jsonl"),
        ("repetitive_string", traj / "from_loopllm/repetitive_string.jsonl"),
    ]:
        if not path.is_file():
            continue
        for line in open(path):
            r = json.loads(line)
            if "prompt" in r:
                table[(cat, int(r["id"]))] = r["prompt"]

    tokens_dir = repo / "recur_code/exp/stop_pattern/tokens"
    for f in sorted(tokens_dir.glob("*.json")):
        blob = json.load(open(f))
        cat = blob.get("category")
        if cat not in ("concise_reasoning", "productive_reflection"):
            continue
        text = "".join(blob["tokens"][: int(blob["prompt_len"])])
        if text.startswith(TPL_PREFIX):
            text = text[len(TPL_PREFIX):]
        if text.endswith(TPL_SUFFIX):
            text = text[: -len(TPL_SUFFIX)]
        table[(cat, int(blob["id"]))] = text
    return table


def select_samples(repo: Path, detector_path: str, per_class: dict[str, int],
                   seed: int, all_trajectories: bool) -> list[dict]:
    """Held-out trajectories of the saved detector, with prompts attached."""
    from detector import load_detector, load_trajectories

    det = load_detector(detector_path)
    trajs = [t for t in load_trajectories([str(repo / "recur_code/exp/step_trajectory_total")])
             if t["model"] == MODEL_SHORT]
    if not all_trajectories:
        held = set(det.test_group_ids)
        trajs = [t for t in trajs if t["group_id"] in held]

    table = load_prompt_table(repo)
    rng = random.Random(seed)
    out: list[dict] = []
    for cat in CATEGORIES:
        pool = [t for t in trajs if t["category"] == cat]
        missing = [t for t in pool if (cat, t["id"]) not in table]
        pool = [t for t in pool if (cat, t["id"]) in table]
        rng.shuffle(pool)
        n = per_class.get(cat, 0)
        take = pool if n <= 0 else pool[:n]
        if missing:
            print(f"[select] {cat}: {len(missing)} 条 held-out 轨迹找不到原 prompt，跳过")
        print(f"[select] {cat}: 可用 {len(pool)} 条，取 {len(take)} 条")
        for t in take:
            out.append({
                "group_id": t["group_id"], "category": cat, "id": int(t["id"]),
                "label": t["label"], "prompt": table[(cat, t["id"])],
                "offline_prompt_len": int(t["prompt_len"]),
                "offline_gen_len": int(t["pmf"].size),
            })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detector", default=str(
        REPO_ROOT / "recur_code/exp/detection_dataset/models/"
                    "loop_detector_llama8b_4class_method2.json"))
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--out", default=str(
        REPO_ROOT / "recur_code/exp/detection_dataset/replay/online_llama8b.jsonl"))
    ap.add_argument("--n-concise", type=int, default=10)
    ap.add_argument("--n-productive", type=int, default=10)
    ap.add_argument("--n-rep-reasoning", type=int, default=0, help="0 = all")
    ap.add_argument("--n-rep-string", type=int, default=0, help="0 = all")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--tau-prob", type=float, default=0.5)
    ap.add_argument("--loop-n", type=int, default=5,
                    help="loop_check block-repeat count for the post-hoc label")
    ap.add_argument("--no-early-stop", action="store_true",
                    help="keep generating past a trigger (needed to see what "
                         "the trajectory WOULD have done — costs GPU time)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-trajectories", action="store_true")
    ap.add_argument("--dump-metrics", action="store_true",
                    help="store the per-gen-token metric arrays of the last "
                         "probe (for the online-vs-offline comparison)")
    a = ap.parse_args()

    samples = select_samples(REPO_ROOT, a.detector,
                             {"concise_reasoning": a.n_concise,
                              "productive_reflection": a.n_productive,
                              "repetitive_reasoning": a.n_rep_reasoning,
                              "repetitive_string": a.n_rep_string},
                             a.seed, a.all_trajectories)
    print(f"[online] {len(samples)} samples; out={a.out}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    done = set()
    if os.path.exists(a.out):                       # resume
        for line in open(a.out):
            try:
                done.add(json.loads(line)["group_id"])
            except Exception:
                pass
        print(f"[online] resuming — {len(done)} samples already recorded")

    from detection_runtime import OnlineLoopDetector

    runtime = OnlineLoopDetector(
        MODEL_PATH, a.detector, gpu=a.gpu, tau_prob=a.tau_prob,
        max_new_tokens=a.max_new_tokens, stop_on_attack=not a.no_early_stop)

    t_start = time.time()
    for k, s in enumerate(samples, start=1):
        if s["group_id"] in done:
            continue
        print(f"\n[{k}/{len(samples)}] {s['group_id']}  (label={s['label']}, "
              f"offline prompt_len={s['offline_prompt_len']} "
              f"gen_len={s['offline_gen_len']})")
        r = runtime.run(s["prompt"], meta={k: v for k, v in s.items()
                                           if k != "prompt"},
                        dump_metrics=a.dump_metrics)
        if r["prompt_len"] != s["offline_prompt_len"]:
            print(f"    !! prompt_len {r['prompt_len']} != offline "
                  f"{s['offline_prompt_len']} — 该样本的 prompt 复原不可信，丢弃")
            continue
        try:
            looped = loop_check_for_category(
                r["full_text"], r["thinking"], s["category"], a.loop_n)
        except Exception as e:                       # loop_check can IndexError
            print(f"    loop_check failed: {e}")
            looped = None
        r["online_loop"] = looped
        r.pop("full_text", None)
        with open(a.out, "a") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"    -> verdict={r['verdict']} trigger_L={r['trigger_L']} "
              f"gen={r['gen_tokens']} online_loop={looped} "
              f"stopped={r['stopped_naturally']} secs={r['seconds']}")

    print(f"\n[online] done in {(time.time()-t_start)/60:.1f} min -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
