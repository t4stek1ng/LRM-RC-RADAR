"""Re-tokenize cached samples with the *legacy* transformers version that
produced the attention cache, so per-token text aligns byte-exactly with
the cached `seq_len` / `prompt_len`.

Why this exists: the cache under ``recur_code/exp/hf_attention_cache/`` was
extracted with transformers ~4.51.x. The current environment ships
transformers 5.x (vLLM 0.21 requires >=4.56), whose BPE/chat-template
output differs by a handful of tokens per sample — enough to misalign every
highlighted token after the first divergence.

Operationally: ``compute_stop_pattern.py`` spawns this script as a
subprocess with ``PYTHONPATH=/root/.cache/recur-legacy-transformers`` so
the older `transformers` is loaded only for the tokenize step, without
polluting the main env (which still needs the newer one for vLLM).

Output for each qualifying cached sample:
    <output-dir>/<category>_<id>.json
        {
          "category": str, "id": int,
          "prompt_len": int, "seq_len": int,
          "tokens": list[str],          # length == seq_len
          "transformers_version": str,
        }

If a sample's re-tokenized seq_len does NOT match the cache's seq_len, that
sample is skipped with a warning (the legacy tokenizer didn't reproduce it
either — usually because cache.seq_len hit the extractor's max_seq_len cap).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = REPO_ROOT / "recur_code/exp/hf_attention_cache"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "recur_code/exp/stop_pattern/tokens"
DEFAULT_MODEL_PATH = Path(os.environ.get("RC_MODELS_ROOT", "/root/project/models")) / "DeepSeek-R1-Distill-Llama-8B"
DEFAULT_LEGACY_PATH = Path("/root/.cache/recur-legacy-transformers")

QUALIFYING_CATEGORY_PREFIXES = (
    "concise_reasoning_",
    "productive_reflection_",
    "repetitive_reasoning_",  # only loop=True records are processed
)
MAX_THINKING_TOKENS = 8192

DEFAULT_RECORDS: dict[str, Path] = {
    "concise_reasoning": REPO_ROOT
    / "recur_code/reasoning_trajectory/DeepSeek-R1-Distill-Llama-8B/concise_reasoning.jsonl",
    "productive_reflection": REPO_ROOT
    / "recur_code/reasoning_trajectory/DeepSeek-R1-Distill-Llama-8B/productive_reflection.jsonl",
    "repetitive_reasoning": REPO_ROOT
    / "recur_code/reasoning_trajectory/DeepSeek-R1-Distill-Llama-8B/repetitive_reasoning.jsonl",
}


def _ensure_legacy_on_path(legacy_path: Path) -> None:
    # The caller is expected to set PYTHONPATH, but if invoked directly we
    # also prepend here so the imports below resolve to the legacy version.
    p = str(legacy_path)
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_records(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            r = json.loads(s)
            out[int(r["id"])] = r
    return out


def _category_from_dirname(name: str) -> str | None:
    for prefix in QUALIFYING_CATEGORY_PREFIXES:
        if name.startswith(prefix):
            return prefix.rstrip("_")
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    p.add_argument("--legacy-path", type=Path, default=DEFAULT_LEGACY_PATH,
                   help="Where the legacy transformers install lives.")
    p.add_argument("--max-thinking", type=int, default=MAX_THINKING_TOKENS)
    args = p.parse_args()

    _ensure_legacy_on_path(args.legacy_path)
    import transformers  # type: ignore  # noqa: E402
    from transformers import AutoTokenizer  # type: ignore  # noqa: E402

    if not transformers.__version__.startswith("4.51"):
        print(
            f"[retokenize-legacy] WARNING: loaded transformers "
            f"{transformers.__version__} (expected 4.51.x). "
            f"Tokens may not align with cache.",
            file=sys.stderr,
        )

    print(f"[retokenize-legacy] using transformers {transformers.__version__}")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model))

    records: dict[str, dict[int, dict]] = {
        cat: _load_records(path) for cat, path in DEFAULT_RECORDS.items()
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    n_ok = n_skip = n_mismatch = 0

    for sample_dir in sorted(args.cache_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        category = _category_from_dirname(sample_dir.name)
        if category is None:
            continue
        try:
            sample_id = int(sample_dir.name[len(category) + 1:])
        except ValueError:
            continue

        meta_path = sample_dir / "meta.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text())
        prompt_len = int(meta["prompt_len"])
        seq_len = int(meta["seq_len"])

        rec = records.get(category, {}).get(sample_id)
        if rec is None:
            print(f"[retokenize-legacy] skip {sample_dir.name}: id not in jsonl")
            n_skip += 1
            continue

        # Different qualification rules per category:
        #  - concise/productive: skip samples that hit the thinking cap (their
        #    last cached token isn't the pre-</think> token);
        #  - repetitive_reasoning: only loop=True samples are interesting for
        #    "why doesn't it stop?" analysis. Non-loop samples have multiple
        #    attempts and no meaningful single trajectory to study.
        if category == "repetitive_reasoning":
            if not rec.get("loop"):
                print(f"[retokenize-legacy] skip {sample_dir.name}: not a loop sample")
                n_skip += 1
                continue
        else:
            if rec.get("thinking_tokens") is not None and rec["thinking_tokens"] >= args.max_thinking:
                print(f"[retokenize-legacy] skip {sample_dir.name}: capped at max_thinking")
                n_skip += 1
                continue

        user_content = rec.get("question") or rec.get("prompt") or ""
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_ids = tokenizer.encode(prompt_text + (rec.get("thinking") or ""),
                                    add_special_tokens=False)
        rp = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        rs = len(full_ids)
        # When the cache was capped at --max-seq-len during extraction, the
        # retokenized full text can be a few tokens longer. Truncate to
        # cache.seq_len: tokenization is deterministic, so the first seq_len
        # tokens match the cache exactly.
        if rp != prompt_len:
            print(
                f"[retokenize-legacy] MISMATCH {sample_dir.name}: "
                f"cache.prompt_len={prompt_len} vs legacy.prompt_len={rp} — skipping"
            )
            n_mismatch += 1
            continue
        if rs < seq_len:
            print(
                f"[retokenize-legacy] MISMATCH {sample_dir.name}: "
                f"cache.seq_len={seq_len} > legacy.seq_len={rs} — skipping"
            )
            n_mismatch += 1
            continue
        if rs > seq_len:
            print(
                f"[retokenize-legacy] {sample_dir.name}: legacy seq_len={rs} "
                f"> cache seq_len={seq_len} (cache was capped); truncating tail."
            )
            full_ids = full_ids[:seq_len]

        # Per-position decoded text. ``skip_special_tokens=False`` keeps
        # the chat-template markers visible (``<｜begin▁of▁sentence｜>`` etc.).
        tokens = [tokenizer.decode([i], skip_special_tokens=False) for i in full_ids]

        out_path = args.output_dir / f"{category}_{sample_id}.json"
        out_path.write_text(json.dumps({
            "category": category,
            "id": sample_id,
            "prompt_len": prompt_len,
            "seq_len": seq_len,
            "tokens": tokens,
            "transformers_version": transformers.__version__,
        }, ensure_ascii=False))
        print(f"[retokenize-legacy] {sample_dir.name}: {len(tokens)} tokens written")
        n_ok += 1

    print(
        f"[retokenize-legacy] done: {n_ok} ok, {n_skip} skipped, "
        f"{n_mismatch} mismatched, output={args.output_dir}"
    )
    return 0 if n_mismatch == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
