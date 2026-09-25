"""Compute the trajectory of (sum_ratio, mean_ratio, prompt_fraction) over every gen token.

For each eligible cached attention sample we compute, at every generation-side
token position ``q`` ∈ ``[prompt_len, seq_len)`` (i.e. every query row that
sees some generated content), the two ratios

  - ``sum_ratio       = sum(attn[q, :prompt_len]) / sum(attn[q, prompt_len:q+1])``
  - ``mean_ratio      = (prompt_sum / prompt_len) / (gen_sum / (q+1-prompt_len))``
  - ``prompt_fraction      = sum(attn[q, :prompt_len]) / sum(attn[q, :q+1])``
  - ``prompt_mean_fraction  = (prompt_sum / prompt_len) / (total_sum / seq_len)``
    (prompt-side per-token average attention relative to the overall per-token
    average; > 1 ⇒ prompt side is over-represented at this query position).

This gives a dense per-token time series — the previous step-only sampling
showed only block-boundary points, which under-resolved long loop samples.

Eligibility:
  - concise_reasoning / productive_reflection: ``record.thinking_tokens <
    MAX_THINKING_TOKENS`` (naturally stopped, not capped).
  - repetitive_reasoning: ``record.loop == True`` only.

Output schema (per sample, parallel-array form keeps file size small)::

    {
      "category": str, "id": int,
      "is_stop_sample": bool, "is_loop_sample": bool,
      "prompt_len": int, "full_cached_seq_len": int,
      "gen_start_q": int,   # = prompt_len, first gen token
      "gen_end_q": int,     # = seq_len - 1, last gen token
    "sum_ratio": list[float],      # length = seq_len - prompt_len
    "mean_ratio": list[float],     # same length, aligned to q = prompt_len + i
    "prompt_fraction": list[float],       # same length
    "prompt_mean_fraction": list[float],  # same length
      "step_query_positions": list[int],  # subset of q's that are step-end content tokens
      "endpoint_q": int | null,           # stop samples: pre-</think> q; loop: null
      "loop_start_block": int | null, "loop_period_blocks": int | null,
      "loop_last_matching_iteration": int | null,
      "loop_first_iter_query_q": int | null,
      "loop_last_iter_query_q": int | null,
    }

Usage::

    PYTHONPATH=. python -m recur_code.visualization.reasoning_chat.compute_step_trajectory
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "recur_code" / "script" / "experiments"))

MAX_THINKING_TOKENS = 8192
LAYER_AVG_FILENAME = "attn_hf_L16-L31_avg.npy"
DEFAULT_CACHE_DIR = REPO_ROOT / "recur_code/exp/hf_attention_cache"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "recur_code/exp/step_trajectory"
DEFAULT_TOKENS_DIR = REPO_ROOT / "recur_code/exp/stop_pattern/tokens"
DEFAULT_MODEL_PATH = Path(os.environ.get("RC_MODELS_ROOT", "/root/project/models")) / "DeepSeek-R1-Distill-Llama-8B"
DEFAULT_LEGACY_PATH = Path("/root/.cache/recur-legacy-transformers")

DEFAULT_RECORDS: dict[str, Path] = {
    "concise_reasoning": REPO_ROOT
    / "recur_code/reasoning_trajectory/DeepSeek-R1-Distill-Llama-8B/concise_reasoning.jsonl",
    "productive_reflection": REPO_ROOT
    / "recur_code/reasoning_trajectory/DeepSeek-R1-Distill-Llama-8B/productive_reflection.jsonl",
    "repetitive_reasoning": REPO_ROOT
    / "recur_code/reasoning_trajectory/DeepSeek-R1-Distill-Llama-8B/repetitive_reasoning.jsonl",
}


def _retokenize_current_inproc(
    cache_dir: Path, tokens_dir: Path, model_path: Path,
    records: dict[str, dict[int, dict[str, Any]]],
    max_thinking: int,
    nonloop_rep_as_stop: bool = False,
) -> None:
    """Write ``<tokens-dir>/<category>_<id>.json`` using the *current*
    transformers install — for caches that were extracted with this same
    tokenizer version (Qwen-14B / QwQ-32B from-total experiment, and the
    3 new 2026 models), there's no need for the legacy 4.51 subprocess."""
    import transformers  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415
    import _hf_backend as B  # noqa: PLC0415

    print(f"[step-traj] retokenize (current) using transformers "
          f"{transformers.__version__}, model={model_path.name}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model_type = B.model_type_of(str(model_path))
    tokens_dir.mkdir(parents=True, exist_ok=True)

    n_ok = n_skip = n_mismatch = 0
    for sample_dir in sorted(cache_dir.iterdir()):
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
            n_skip += 1
            continue

        if category in ("repetitive_reasoning", "repetitive_string"):
            # By default only loop samples are retokenized. With
            # nonloop_rep_as_stop, non-looping repetitive samples are also
            # retokenized so they can be plotted as a third (no-loop) class.
            if not rec.get("loop") and not nonloop_rep_as_stop:
                n_skip += 1
                continue
        else:
            if (rec.get("thinking_tokens") is not None
                    and rec["thinking_tokens"] >= max_thinking):
                # capped thinking — still useful here (the cache was capped
                # at max_seq_len during extraction and meta.seq_len reflects
                # the actual cached length), so retokenize and truncate.
                pass

        user_content = rec.get("question") or rec.get("prompt") or ""
        prompt_text = B.build_prompt_text(tokenizer, user_content, model_type)
        full_ids = tokenizer.encode(
            prompt_text + (rec.get("thinking") or ""), add_special_tokens=False,
        )
        rp = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        rs = len(full_ids)
        if rp != prompt_len:
            print(f"[step-traj] MISMATCH {sample_dir.name}: "
                  f"cache.prompt_len={prompt_len} vs current.prompt_len={rp}")
            n_mismatch += 1
            continue
        if rs < seq_len:
            print(f"[step-traj] MISMATCH {sample_dir.name}: "
                  f"cache.seq_len={seq_len} > current.seq_len={rs}")
            n_mismatch += 1
            continue
        if rs > seq_len:
            full_ids = full_ids[:seq_len]
        tokens = [tokenizer.decode([i], skip_special_tokens=False) for i in full_ids]
        (tokens_dir / f"{category}_{sample_id}.json").write_text(
            json.dumps({
                "category": category, "id": sample_id,
                "prompt_len": prompt_len, "seq_len": seq_len,
                "tokens": tokens,
                "transformers_version": transformers.__version__,
            }, ensure_ascii=False)
        )
        n_ok += 1
    print(f"[step-traj] retokenize done: ok={n_ok} skip={n_skip} mismatch={n_mismatch}")


def _load_records(path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            r = json.loads(s)
            out[int(r["id"])] = r
    return out


def _category_from_dirname(name: str) -> str | None:
    for prefix in ("concise_reasoning_", "productive_reasoning_",
                   "productive_reflection_",
                   "repetitive_reasoning_", "repetitive_string_"):
        if name.startswith(prefix):
            return prefix.rstrip("_")
    return None


def _step_query_positions(tokens: list[str], prompt_len: int) -> list[int]:
    """For each token in gen whose text contains ``\\n\\n`` (a step-end
    marker), return the position of the content token JUST BEFORE it
    (matching the convention used elsewhere in the pipeline)."""
    out: list[int] = []
    for i in range(prompt_len, len(tokens)):
        if "\n\n" not in tokens[i]:
            continue
        q = i - 1
        while q > prompt_len and "\n\n" in tokens[q]:
            q -= 1
        if q > prompt_len and q not in out:
            out.append(q)
    return out


def _find_loop(blocks: list[str]) -> tuple[int | None, int | None]:
    n = len(blocks)
    while n > 0 and blocks[n - 1] == "":
        n -= 1
    for i in range(n):
        for L in range(1, (n - i) // 2 + 1):
            if blocks[i:i + L] == blocks[i + L:i + 2 * L]:
                return i, L
    return None, None


def _last_matching_iteration(
    blocks: list[str], step_count: int,
    loop_start: int, loop_period: int,
) -> int:
    loop_pattern = blocks[loop_start:loop_start + loop_period]
    best = 0
    max_k = (len(blocks) - loop_start) // loop_period
    for k in range(1, max_k + 1):
        end_block = loop_start + k * loop_period - 1
        start_block = loop_start + (k - 1) * loop_period
        if end_block >= step_count:
            break
        if blocks[start_block:end_block + 1] == loop_pattern:
            best = k
    return best


def _compute_per_token_consistency(
    attn: np.ndarray, prompt_len: int, tokens: list[str],
    threshold: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """For each gen token q, compute attention consistency between the
    prompt-side and gen-side portions of the top-``threshold`` mass.

    Definition (as specified):
      1. ``row = attn[q, :q+1] / sum``; sort by attn desc and pick the
         smallest set ``S`` whose cumulative sum ≥ threshold.
      2. Split ``S`` into ``S_p`` (positions < prompt_len) and ``S_g``
         (positions ≥ prompt_len).
      3. For each part, group positions by their decoded token *string*;
         each unique string ``t`` gets the sum of attention across all its
         occurrences within that part.  Proportions are computed relative
         to the *combined* mass of both parts: ``p(t) = cum[t] / total``
         where ``total = Σ prompt_cum + Σ gen_cum``.
      4. Take the intersection of unique strings between the two parts.
      5. ``raw_diff = Σ_{t ∈ intersection} |prop_p(t) - prop_g(t)|``;
         ``consistency = 1 - raw_diff``.
         Value ∈ [−1, 1]: 1 means perfectly consistent distributions,
         0 means completely disjoint, <0 possible when distributions
         are anti-correlated over the intersection.

    Returns (consistency, intersection_size), each of length
    ``seq_len - prompt_len`` and indexed by ``i`` where ``q = prompt_len + i``.
    Positions where one side has zero mass produce NaN for consistency
    (since the proportion isn't defined); empty intersections produce
    ``consistency = 1`` (since raw_diff = 0 for an empty sum).
    """
    from collections import defaultdict
    seq_len = attn.shape[0]
    n_gen = seq_len - prompt_len
    consistency = np.full(n_gen, np.nan, dtype=np.float64)
    intersection_size = np.zeros(n_gen, dtype=np.int32)

    for i in range(n_gen):
        q = prompt_len + i
        row = attn[q, :q + 1].astype(np.float64)
        s = row.sum()
        if s <= 0:
            continue
        row = row / s

        order = np.argsort(-row, kind="stable")
        csum = np.cumsum(row[order])
        k = int(np.searchsorted(csum, threshold) + 1)
        k = max(1, min(k, len(row)))
        top = order[:k]

        # Split by prompt/gen and group by token string within each side.
        prompt_cum: dict[str, float] = defaultdict(float)
        gen_cum: dict[str, float] = defaultdict(float)
        for pos in top.tolist():
            tok = tokens[pos]
            v = float(row[pos])
            if pos < prompt_len:
                prompt_cum[tok] += v
            else:
                gen_cum[tok] += v

        p_total = sum(prompt_cum.values())
        g_total = sum(gen_cum.values())
        total = p_total + g_total
        if total <= 0 or p_total <= 0 or g_total <= 0:
            continue  # one side has no mass in the top set — consistency undefined

        common = set(prompt_cum) & set(gen_cum)
        intersection_size[i] = len(common)

        cons = 0.0
        for t in common:
            cons += abs(prompt_cum[t] / total - gen_cum[t] / total)
        consistency[i] = 1.0 - cons

    return consistency, intersection_size


def _compute_per_token_consistency_independent(
    attn: np.ndarray, prompt_len: int, tokens: list[str],
    threshold: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Same as ``_compute_per_token_consistency`` but with independent
    normalisation: each side's proportions are computed relative to its
    own total, not the combined total.

    ``prop_p(t) = prompt_cum[t] / Σ prompt_cum``
    ``prop_g(t) = gen_cum[t]    / Σ gen_cum``

    ``consistency = 1 − Σ_{t ∈ intersection} |prop_p(t) − prop_g(t)|``
    """
    from collections import defaultdict
    seq_len = attn.shape[0]
    n_gen = seq_len - prompt_len
    consistency = np.full(n_gen, np.nan, dtype=np.float64)
    intersection_size = np.zeros(n_gen, dtype=np.int32)

    for i in range(n_gen):
        q = prompt_len + i
        row = attn[q, :q + 1].astype(np.float64)
        s = row.sum()
        if s <= 0:
            continue
        row = row / s

        order = np.argsort(-row, kind="stable")
        csum = np.cumsum(row[order])
        k = int(np.searchsorted(csum, threshold) + 1)
        k = max(1, min(k, len(row)))
        top = order[:k]

        prompt_cum: dict[str, float] = defaultdict(float)
        gen_cum: dict[str, float] = defaultdict(float)
        for pos in top.tolist():
            tok = tokens[pos]
            v = float(row[pos])
            if pos < prompt_len:
                prompt_cum[tok] += v
            else:
                gen_cum[tok] += v

        p_total = sum(prompt_cum.values())
        g_total = sum(gen_cum.values())
        if p_total <= 0 or g_total <= 0:
            continue

        common = set(prompt_cum) & set(gen_cum)
        intersection_size[i] = len(common)

        cons = 0.0
        for t in common:
            cons += abs(prompt_cum[t] / p_total - gen_cum[t] / g_total)
        consistency[i] = 1.0 - cons

    return consistency, intersection_size


def _compute_per_token_consistency_union(
    attn: np.ndarray, prompt_len: int, tokens: list[str],
    threshold: float = 0.9,
) -> np.ndarray:
    """Union variant of ``_compute_per_token_consistency_independent``.

    Same top-``threshold`` selection and independent per-side normalisation:
    ``prop_p(t) = prompt_cum[t]/Σprompt_cum`` and ``prop_g(t) = gen_cum[t]/Σgen_cum``.
    The ONLY difference: non-intersection token strings are no longer dropped —
    a prompt-only token contributes ``prop_p(t)`` and a gen-only token
    contributes ``prop_g(t)`` (i.e. ``|prop_p - prop_g|`` with the missing side
    treated as 0). So the accumulator is the full union L1 distance

        A = Σ_{t ∈ prompt∪gen} |prop_p(t) - prop_g(t)| = ‖p - g‖₁ = 2·TV(p, g)

    and the metric is ``consistency_union = 2 - A``. Since p, g are each
    probability distributions, ``0 ≤ A ≤ 2`` (A = 2 iff disjoint supports), so
    ``consistency_union ∈ [0, 2]``. This penalises gen-side loop tokens that are
    absent from the prompt (they fall outside the old intersection and were
    previously free), at the cost of also folding in the prompt-only sink mass.
    """
    from collections import defaultdict
    seq_len = attn.shape[0]
    n_gen = seq_len - prompt_len
    out = np.full(n_gen, np.nan, dtype=np.float64)

    for i in range(n_gen):
        q = prompt_len + i
        row = attn[q, :q + 1].astype(np.float64)
        s = row.sum()
        if s <= 0:
            continue
        row = row / s

        order = np.argsort(-row, kind="stable")
        csum = np.cumsum(row[order])
        k = int(np.searchsorted(csum, threshold) + 1)
        k = max(1, min(k, len(row)))
        top = order[:k]

        prompt_cum: dict[str, float] = defaultdict(float)
        gen_cum: dict[str, float] = defaultdict(float)
        for pos in top.tolist():
            tok = tokens[pos]
            v = float(row[pos])
            if pos < prompt_len:
                prompt_cum[tok] += v
            else:
                gen_cum[tok] += v

        p_total = sum(prompt_cum.values())
        g_total = sum(gen_cum.values())
        if p_total <= 0 or g_total <= 0:
            continue

        acc = 0.0
        for t in set(prompt_cum) | set(gen_cum):
            acc += abs(prompt_cum.get(t, 0.0) / p_total
                       - gen_cum.get(t, 0.0) / g_total)
        out[i] = 2.0 - acc

    return out


def _compute_per_token_consistency_common_norm(
    attn: np.ndarray, prompt_len: int, tokens: list[str],
    threshold: float = 0.9,
) -> np.ndarray:
    """Intersection-normalised variant (method 3).

    Same top-``threshold`` selection and prompt/gen split as the original, but
    the per-side normalisation denominator is the mass of the **intersection**
    only, not the full top-p mass of each side:

        common  = prompt_tokens ∩ gen_tokens
        p_total = Σ_{t∈common} prompt_cum[t]      (NOT Σ over all prompt-side)
        g_total = Σ_{t∈common} gen_cum[t]
        prop_p(t) = prompt_cum[t]/p_total,  prop_g(t) = gen_cum[t]/g_total
        consistency = 1 − Σ_{t∈common} |prop_p(t) − prop_g(t)|

    Rationale: in the original, a large gen-only loop token (or prompt-only
    sink) inflates the side total and shrinks every intersection proportion,
    driving |prop_p−prop_g|→0 and consistency→1 (the GLM verbalized-loop
    false-high). Normalising over the intersection removes that dilution: both
    prop_p and prop_g sum to 1 over the shared tokens, so the metric measures
    only whether the *shared* tokens are relatively distributed the same way.
    Range: since both are distributions over ``common``, Σ|·| ∈ [0, 2] ⇒
    consistency ∈ [−1, 1]. Undefined (NaN) when the intersection is empty.
    """
    from collections import defaultdict
    seq_len = attn.shape[0]
    n_gen = seq_len - prompt_len
    out = np.full(n_gen, np.nan, dtype=np.float64)

    for i in range(n_gen):
        q = prompt_len + i
        row = attn[q, :q + 1].astype(np.float64)
        s = row.sum()
        if s <= 0:
            continue
        row = row / s

        order = np.argsort(-row, kind="stable")
        csum = np.cumsum(row[order])
        k = int(np.searchsorted(csum, threshold) + 1)
        k = max(1, min(k, len(row)))
        top = order[:k]

        prompt_cum: dict[str, float] = defaultdict(float)
        gen_cum: dict[str, float] = defaultdict(float)
        for pos in top.tolist():
            tok = tokens[pos]
            v = float(row[pos])
            if pos < prompt_len:
                prompt_cum[tok] += v
            else:
                gen_cum[tok] += v

        common = set(prompt_cum) & set(gen_cum)
        if not common:
            continue
        p_total = sum(prompt_cum[t] for t in common)
        g_total = sum(gen_cum[t] for t in common)
        if p_total <= 0 or g_total <= 0:
            continue

        cons = 1.0 - sum(abs(prompt_cum[t] / p_total - gen_cum[t] / g_total)
                         for t in common)
        out[i] = cons

    return out


def _compute_per_token_consistency_gen_centric(
    attn: np.ndarray, prompt_len: int, tokens: list[str],
    threshold: float = 0.9,
) -> np.ndarray:
    """Gen-centric asymmetric variant (method 4).

    Same top-``threshold`` selection, prompt/gen split, and independent
    per-side normalisation as the original (``prop_p = prompt_cum/p_total``,
    ``prop_g = gen_cum/g_total``), but the accumulator iterates over the
    **gen-side** tokens only:

        A = Σ_{t ∈ gen∩prompt} |prop_g(t) − prop_p(t)|   (shared: distribution gap)
          + Σ_{t ∈ gen∖prompt} prop_g(t)                 (gen-only: full penalty)
        consistency = 1 − A

    Motivation: charge every unit of gen attention mass that lands on tokens
    absent from the prompt (the "loop on novel tokens" blind spot), plus the
    distribution mismatch on shared tokens, while ignoring prompt-only sink
    mass (which dominated the symmetric union variant). Since the gen side is a
    distribution, with G_i = Σ_isect prop_g, G_o = Σ_genonly prop_g (G_i+G_o=1)
    and P_i = Σ_isect prop_p ≤ 1:
        A ≤ (G_i + P_i) + G_o = 1 + P_i ≤ 2   ⇒   consistency ∈ [−1, 1].
    """
    from collections import defaultdict
    seq_len = attn.shape[0]
    n_gen = seq_len - prompt_len
    out = np.full(n_gen, np.nan, dtype=np.float64)

    for i in range(n_gen):
        q = prompt_len + i
        row = attn[q, :q + 1].astype(np.float64)
        s = row.sum()
        if s <= 0:
            continue
        row = row / s

        order = np.argsort(-row, kind="stable")
        csum = np.cumsum(row[order])
        k = int(np.searchsorted(csum, threshold) + 1)
        k = max(1, min(k, len(row)))
        top = order[:k]

        prompt_cum: dict[str, float] = defaultdict(float)
        gen_cum: dict[str, float] = defaultdict(float)
        for pos in top.tolist():
            tok = tokens[pos]
            v = float(row[pos])
            if pos < prompt_len:
                prompt_cum[tok] += v
            else:
                gen_cum[tok] += v

        p_total = sum(prompt_cum.values())
        g_total = sum(gen_cum.values())
        if p_total <= 0 or g_total <= 0:
            continue

        acc = 0.0
        for tok, v in gen_cum.items():
            prop_g = v / g_total
            if tok in prompt_cum:
                acc += abs(prop_g - prompt_cum[tok] / p_total)
            else:
                acc += prop_g
        out[i] = 1.0 - acc

    return out


def _compute_per_token_ratios(
    attn: np.ndarray, prompt_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (sum_ratio_arr, mean_ratio_arr, prompt_fraction_arr,
    prompt_mean_fraction_arr) of length seq_len - prompt_len.

    Vectorized: each gen row is normalized once, then we read off the prompt
    mass directly. ``gen_sum = 1 - prompt_sum`` post-normalization (modulo
    fp32 round-off), but we compute it explicitly via the causal mask to
    sidestep any drift.
    """
    seq_len = attn.shape[0]
    gen_rows = attn[prompt_len:, :].astype(np.float32, copy=False)
    # Row-normalize so each gen row sums to 1.
    row_sums = gen_rows.sum(axis=1, keepdims=True)
    # Avoid division by zero for any all-zero rows (shouldn't happen, but
    # be safe — those rows get NaN ratios).
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    gen_rows = gen_rows / row_sums

    prompt_sums = gen_rows[:, :prompt_len].sum(axis=1)
    # Gen mass for each row q is the sum over keys [prompt_len, q+1) — but
    # since the matrix is causal (upper triangle is zero), we can just sum
    # the entire [prompt_len, seq_len) slice; the zeros above the diagonal
    # don't add anything.
    gen_sums = gen_rows[:, prompt_len:].sum(axis=1)

    # gen_len_q = q + 1 - prompt_len. For row index i in gen_rows, q = prompt_len + i,
    # so gen_len = i + 1.
    n_gen = seq_len - prompt_len
    gen_lens = np.arange(1, n_gen + 1, dtype=np.float32)

    prompt_mean = prompt_sums / float(prompt_len) if prompt_len > 0 else np.zeros_like(prompt_sums)
    gen_mean = gen_sums / gen_lens

    total_sums = prompt_sums + gen_sums

    # prompt_mean_fraction: per-token prompt average / overall per-token average.
    # Math identity: prompt_mean_fraction = prompt_fraction * seq_len / prompt_len.
    total_lens = prompt_len + gen_lens  # seq_len for each query position

    with np.errstate(divide="ignore", invalid="ignore"):
        sum_ratio = np.where(gen_sums > 0, prompt_sums / gen_sums, np.inf)
        mean_ratio = np.where(gen_mean > 0, prompt_mean / gen_mean, np.inf)
        prompt_fraction = np.where(total_sums > 0, prompt_sums / total_sums, np.nan)
        prompt_mean_fraction = np.where(
            (total_sums > 0) & (prompt_len > 0),
            (prompt_sums / float(prompt_len)) / (total_sums / total_lens),
            np.nan,
        )

    return sum_ratio, mean_ratio, prompt_fraction, prompt_mean_fraction


def compute_one_sample(
    sample_dir: Path,
    record: dict[str, Any],
    tokens_dir: Path,
    max_thinking: int = MAX_THINKING_TOKENS,
    attn_filename: str = LAYER_AVG_FILENAME,
    nonloop_rep_as_stop: bool = False,
    include_capped: bool = False,
) -> dict[str, Any] | None:
    meta_path = sample_dir / "meta.json"
    attn_path = sample_dir / attn_filename
    if not meta_path.is_file() or not attn_path.is_file():
        return None
    meta = json.loads(meta_path.read_text())
    category = meta.get("category")
    sample_id = int(meta["id"])
    prompt_len = int(meta["prompt_len"])
    seq_len = int(meta["seq_len"])

    is_loop_sample = (category in ("repetitive_reasoning", "repetitive_string")
                      and bool(record.get("loop")))
    # A non-looping repetitive sample ended naturally (no loop), so it is
    # treated like a stop sample (endpoint marker at the last gen token).
    # Gated behind nonloop_rep_as_stop so the default pipeline is unchanged.
    is_nonloop_rep = (
        nonloop_rep_as_stop
        and category == "repetitive_reasoning"
        and not bool(record.get("loop"))
    )
    is_stop_sample = (
        category in ("concise_reasoning", "productive_reasoning", "productive_reflection")
        and (include_capped
             or (record.get("thinking_tokens") or 0) < max_thinking)
    ) or is_nonloop_rep
    if not (is_loop_sample or is_stop_sample):
        return None

    tokens_path = tokens_dir / f"{category}_{sample_id}.json"
    if not tokens_path.is_file():
        return None
    tok_blob = json.loads(tokens_path.read_text())
    tokens = tok_blob["tokens"]
    if len(tokens) != seq_len or int(tok_blob["prompt_len"]) != prompt_len:
        raise ValueError(f"{sample_dir.name}: legacy tokens out of sync with cache")

    attn = np.load(attn_path).astype(np.float32)
    if attn.shape[0] != seq_len or attn.shape[1] != seq_len:
        raise ValueError(f"{sample_dir.name}: attn shape mismatch with seq_len")

    sum_arr, mean_arr, pf_arr, pmf_arr = _compute_per_token_ratios(attn, prompt_len)
    sum_list = [round(float(x), 6) if np.isfinite(x) else None for x in sum_arr]
    mean_list = [round(float(x), 6) if np.isfinite(x) else None for x in mean_arr]
    pf_list = [round(float(x), 6) if np.isfinite(x) else None for x in pf_arr]
    pmf_list = [round(float(x), 6) if np.isfinite(x) else None for x in pmf_arr]

    print(f"[step-traj]   {sample_dir.name}: computing per-token consistency over {seq_len - prompt_len} gen tokens...")
    cons_arr, intersect_arr = _compute_per_token_consistency(
        attn, prompt_len, tokens, threshold=0.9,
    )
    cons_list = [round(float(x), 6) if np.isfinite(x) else None for x in cons_arr]
    intersect_list = [int(x) for x in intersect_arr]

    cons_ind_arr, _ = _compute_per_token_consistency_independent(
        attn, prompt_len, tokens, threshold=0.9,
    )
    cons_ind_list = [round(float(x), 6) if np.isfinite(x) else None for x in cons_ind_arr]

    cons_union_arr = _compute_per_token_consistency_union(
        attn, prompt_len, tokens, threshold=0.9,
    )
    cons_union_list = [round(float(x), 6) if np.isfinite(x) else None for x in cons_union_arr]

    cons_common_arr = _compute_per_token_consistency_common_norm(
        attn, prompt_len, tokens, threshold=0.9,
    )
    cons_common_list = [round(float(x), 6) if np.isfinite(x) else None for x in cons_common_arr]

    cons_gen_arr = _compute_per_token_consistency_gen_centric(
        attn, prompt_len, tokens, threshold=0.9,
    )
    cons_gen_list = [round(float(x), 6) if np.isfinite(x) else None for x in cons_gen_arr]

    step_qs = _step_query_positions(tokens, prompt_len)

    out: dict[str, Any] = {
        "category": category,
        "id": sample_id,
        "is_stop_sample": is_stop_sample,
        "is_loop_sample": is_loop_sample,
        "prompt_len": prompt_len,
        "full_cached_seq_len": seq_len,
        "gen_start_q": prompt_len,
        "gen_end_q": seq_len - 1,
        "sum_ratio": sum_list,
        "mean_ratio": mean_list,
        "prompt_fraction": pf_list,
        "prompt_mean_fraction": pmf_list,
        "attn_consistency": cons_list,
        "attn_consistency_independent": cons_ind_list,
        "attn_consistency_union": cons_union_list,
        "attn_consistency_common": cons_common_list,
        "attn_consistency_gen": cons_gen_list,
        "intersection_size": intersect_list,
        "step_query_positions": step_qs,
        "endpoint_q": (seq_len - 1) if is_stop_sample else None,
        "loop_start_block": None,
        "loop_period_blocks": None,
        "loop_last_matching_iteration": None,
        "loop_first_iter_query_q": None,
        "loop_last_iter_query_q": None,
    }

    if is_loop_sample:
        blocks = (record.get("thinking") or "").split("\n\n")
        loop_start, loop_period = _find_loop(blocks)
        if loop_start is not None:
            # We need the step-end token positions (not the query positions)
            # to count step boundaries the same way the loop-pattern script
            # did, so the iteration indices line up.
            step_end_positions = [
                i for i in range(prompt_len, len(tokens)) if "\n\n" in tokens[i]
            ]
            last_match = _last_matching_iteration(
                blocks, len(step_end_positions), loop_start, loop_period,
            )
            out["loop_start_block"] = loop_start
            out["loop_period_blocks"] = loop_period
            out["loop_last_matching_iteration"] = last_match

            def _block_idx_to_query_q(block_idx: int) -> int | None:
                if block_idx >= len(step_end_positions):
                    return None
                step_end = step_end_positions[block_idx]
                q = step_end - 1
                while q > prompt_len and "\n\n" in tokens[q]:
                    q -= 1
                return q if q > prompt_len else None

            out["loop_first_iter_query_q"] = _block_idx_to_query_q(loop_start + loop_period - 1)
            if last_match > 0:
                out["loop_last_iter_query_q"] = _block_idx_to_query_q(loop_start + last_match * loop_period - 1)

    return out


def _run_legacy_retokenize(
    cache_dir: Path, tokens_dir: Path, model_path: Path, legacy_path: Path,
) -> None:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{legacy_path}{os.pathsep}{existing}" if existing else str(legacy_path)
    cmd = [
        sys.executable, "-m",
        "recur_code.visualization.reasoning_chat.retokenize_legacy",
        "--cache-dir", str(cache_dir),
        "--output-dir", str(tokens_dir),
        "--model", str(model_path),
        "--legacy-path", str(legacy_path),
    ]
    print("[step-traj] running legacy retokenize subprocess")
    proc = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        raise SystemExit(f"legacy retokenize failed (rc={proc.returncode})")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--tokens-dir", type=Path, default=DEFAULT_TOKENS_DIR)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    p.add_argument("--legacy-path", type=Path, default=DEFAULT_LEGACY_PATH)
    p.add_argument("--max-thinking", type=int, default=MAX_THINKING_TOKENS)
    p.add_argument(
        "--ids", nargs="+", type=int, default=None,
        help="Only compute the selected sample ids; the final manifest is rebuilt from all output files.",
    )
    p.add_argument("--skip-retokenize", action="store_true")
    p.add_argument(
        "--current-tokenizer", action="store_true",
        help="Retokenize in-process with the current transformers install "
             "(use when the attention cache was extracted in the same env, "
             "e.g. the from_total Qwen/QwQ caches).",
    )
    p.add_argument(
        "--attn-filename", default=LAYER_AVG_FILENAME,
        help="Per-sample attention .npy filename inside --cache-dir/<sample>/ "
             f"(default '{LAYER_AVG_FILENAME}'; use 'attn_layer_avg.npy' "
             "for from_total all-layer-averaged caches).",
    )
    p.add_argument(
        "--records-concise", type=Path, default=None,
        help="Override concise_reasoning trajectory JSONL (defaults to Llama-8B path).",
    )
    p.add_argument(
        "--records-productive", type=Path, default=None,
        help="Override productive_reflection trajectory JSONL.",
    )
    p.add_argument(
        "--records-productive-reasoning", type=Path, default=None,
        help="Override productive_reasoning trajectory JSONL (new experiment schema).",
    )
    p.add_argument(
        "--records-repetitive", type=Path, default=None,
        help="Override repetitive_reasoning trajectory JSONL.",
    )
    p.add_argument(
        "--records-repetitive-string", type=Path, default=None,
        help="repetitive_string trajectory JSONL (LoopLLM adapted records; "
             "thinking field = full generation stream).",
    )
    p.add_argument(
        "--nonloop-repetitive-as-stop", action="store_true",
        help="Also emit non-looping repetitive_reasoning samples (loop=False), "
             "treated as stop samples. Needed for the loop-vs-no-loop "
             "comparison plot; default off keeps the standard pipeline "
             "(loop-only repetitive).",
    )
    p.add_argument(
        "--include-capped", action="store_true",
        help="Also emit concise/productive samples whose thinking_tokens hit "
             "the max_thinking cap (naturally-stopped filter off). Needed for "
             "the classifier training set where a capped sample is still a "
             "valid probe of that trajectory's attention behaviour.",
    )
    args = p.parse_args()

    if not args.cache_dir.is_dir():
        raise SystemExit(f"--cache-dir not found: {args.cache_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.tokens_dir.mkdir(parents=True, exist_ok=True)

    record_paths: dict[str, Path] = dict(DEFAULT_RECORDS)
    if args.records_concise is not None:
        record_paths["concise_reasoning"] = args.records_concise
    if args.records_productive is not None:
        record_paths["productive_reflection"] = args.records_productive
    if args.records_productive_reasoning is not None:
        record_paths["productive_reasoning"] = args.records_productive_reasoning
    if args.records_repetitive is not None:
        record_paths["repetitive_reasoning"] = args.records_repetitive
    if args.records_repetitive_string is not None:
        record_paths["repetitive_string"] = args.records_repetitive_string
    records = {
        cat: _load_records(path) for cat, path in record_paths.items()
        if path.exists()
    }

    if not args.skip_retokenize:
        if args.current_tokenizer:
            _retokenize_current_inproc(
                args.cache_dir, args.tokens_dir, args.model,
                records, args.max_thinking,
                nonloop_rep_as_stop=args.nonloop_repetitive_as_stop,
            )
        else:
            _run_legacy_retokenize(args.cache_dir, args.tokens_dir, args.model, args.legacy_path)

    manifest: list[dict[str, Any]] = []
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
        if args.ids is not None and sample_id not in set(args.ids):
            continue
        rec = records.get(category, {}).get(sample_id)
        if rec is None:
            continue
        result = compute_one_sample(
            sample_dir, rec, args.tokens_dir,
            max_thinking=args.max_thinking,
            attn_filename=args.attn_filename,
            nonloop_rep_as_stop=args.nonloop_repetitive_as_stop,
            include_capped=args.include_capped,
        )
        if result is None:
            continue
        out_path = args.output_dir / f"{category}_{sample_id}.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False))
        manifest.append({
            "category": category, "id": sample_id, "file": out_path.name,
            "is_stop_sample": result["is_stop_sample"],
            "is_loop_sample": result["is_loop_sample"],
            "gen_length": result["gen_end_q"] - result["gen_start_q"] + 1,
            "n_step_boundaries": len(result["step_query_positions"]),
        })
        kind = "loop" if result["is_loop_sample"] else "stop"
        gen_n = result["gen_end_q"] - result["gen_start_q"] + 1
        print(
            f"[step-traj] {sample_dir.name} ({kind}): "
            f"{gen_n} gen tokens, {len(result['step_query_positions'])} step boundaries"
        )

    # Rebuild from all completed outputs so disjoint --ids workers can safely
    # contribute files. A worker finishing early may write a partial snapshot;
    # the last worker always sees every file completed before it.
    complete_manifest: list[dict[str, Any]] = []
    for result_path in sorted(args.output_dir.glob("*.json")):
        if result_path.name == "manifest.json":
            continue
        result = json.loads(result_path.read_text())
        complete_manifest.append({
            "category": result["category"], "id": int(result["id"]),
            "file": result_path.name,
            "is_stop_sample": result["is_stop_sample"],
            "is_loop_sample": result["is_loop_sample"],
            "gen_length": result["gen_end_q"] - result["gen_start_q"] + 1,
            "n_step_boundaries": len(result["step_query_positions"]),
        })
    (args.output_dir / "manifest.json").write_text(
        json.dumps(complete_manifest, ensure_ascii=False, indent=2)
    )
    print(f"[step-traj] computed {len(manifest)}; manifest has "
          f"{len(complete_manifest)} trajectories in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
