"""Score every valid generated-token position in offline step-metric files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src/detection"))
from detector import features_at, load_detector  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    detector = load_detector(str(args.model))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.out.open("w") as out:
        # Metrics now sit one level deeper, under the dataset they came from
        # (``step_metrics/GSM8k/concise_reasoning_0.json``); ``rglob`` also
        # still finds directories written flat before that regrouping.
        for path in sorted(args.metrics_dir.rglob("*.json")):
            if path.name == "manifest.json":
                continue
            metric = json.loads(path.read_text())
            # 轨迹 id 在每个数据集里都从 0 重新编号，所以下游按 (category, id)
            # 分组会把不同数据集的轨迹并成一条。带上目录名才唯一。
            subset = path.parent.name if path.parent != args.metrics_dir else None
            pmf = np.asarray(metric["prompt_mean_fraction"], float)
            cons = np.asarray(metric["attn_consistency_independent"], float)
            for i in range(pmf.size):
                feat = features_at(cons, pmf, int(metric["prompt_len"]), i,
                                   log_L=detector.log_L, use_L=detector.use_L)
                if feat is None:
                    continue
                probs = detector.predict_proba(feat)[0]
                out.write(json.dumps({
                    "category": metric["category"], "subset": subset, "id": metric["id"],
                    "gen_pos": i, "L": int(metric["prompt_len"]) + i + 1,
                    "features": [float(x) for x in feat],
                    "probabilities": {label: float(prob) for label, prob in
                                      zip(detector.classes, probs)},
                }, ensure_ascii=False, allow_nan=False) + "\n")
                count += 1
    print(f"wrote {count} scored positions -> {args.out}")


if __name__ == "__main__":
    main()
