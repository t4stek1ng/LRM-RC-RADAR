"""Summarize LoopLLM repetitive_string transferability across the 6 target models.

Reads each model's
    reasoning_trajectory/<model_short>/from_loopllm/repetitive_string.jsonl
and reports per-model loop yield + token stats, then materializes the successful
(loop=True) trajectories into
    dataset/repetitive_string/realized_trajectories.jsonl
plus stats.json.
"""
import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TRAJ = REPO / "recur_code/reasoning_trajectory"
OUT_DIR = REPO / "recur_code/dataset/repetitive_string"

MODELS = [
    "DeepSeek-R1-Distill-Llama-8B",
    "DeepSeek-R1-Distill-Qwen-14B",
    "QwQ-32B",
    "GLM-4.7-Flash",
    "Qwen3.6-27B",
    "Gemma-4-31B-it",
]


def load(model):
    p = TRAJ / model / "from_loopllm" / "repetitive_string.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    realized = []
    per_model = {}
    print(f"{'model':32s} {'recs':>5s} {'loop':>5s} {'rate':>6s} "
          f"{'span(out/think)':>16s} {'attempts(loop)':>14s}")
    for m in MODELS:
        rows = load(m)
        loops = [r for r in rows if r.get("loop")]
        att = [r.get("attempts", 0) for r in loops]
        # where does the repetition live: the answer (output) or the thinking span?
        in_output = [r for r in loops if r.get("output_tokens", 0) > 0]
        in_thinking = [r for r in loops if r.get("output_tokens", 0) == 0]
        med_att = round(st.mean(att), 1) if att else 0
        rate = f"{len(loops)/len(rows):.2f}" if rows else "-"
        print(f"{m:32s} {len(rows):5d} {len(loops):5d} {rate:>6s} "
              f"{str(len(in_output))+'/'+str(len(in_thinking)):>16s} {med_att:>14}")
        per_model[m] = {
            "records": len(rows),
            "loops": len(loops),
            "loop_ids": sorted(r["id"] for r in loops),
            "loop_in_output": len(in_output),
            "loop_in_thinking": len(in_thinking),
        }
        for r in loops:
            realized.append({
                "model": m, "id": r["id"], "prompt": r["prompt"],
                "base_prompt": r.get("base_prompt", ""),
                "adv_suffix": r.get("adv_suffix", ""),
                "loop_span": "output" if r.get("output_tokens", 0) > 0 else "thinking",
                "thinking": r.get("thinking", ""),
                "output": r.get("output", ""),
                "thinking_tokens": r.get("thinking_tokens", 0),
                "output_tokens": r.get("output_tokens", 0),
                "attempts": r.get("attempts", 0),
                "src_success_rate": r.get("src_success_rate"),
            })

    tot_loops = sum(v["loops"] for v in per_model.values())
    tot_recs = sum(v["records"] for v in per_model.values())
    print(f"\nTOTAL: {tot_loops} looping / {tot_recs} records across "
          f"{len([m for m in per_model if per_model[m]['records']])} models")

    with open(OUT_DIR / "realized_trajectories.jsonl", "w") as f:
        for r in realized:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"per_model": per_model, "total_loops": tot_loops, "total_records": tot_recs}
    json.dump(stats, open(OUT_DIR / "stats.json", "w"), indent=2, ensure_ascii=False)
    print(f"\nwrote {len(realized)} realized loop trajectories -> "
          f"{OUT_DIR/'realized_trajectories.jsonl'}")


if __name__ == "__main__":
    main()
