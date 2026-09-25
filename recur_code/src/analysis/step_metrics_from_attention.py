"""Per-position detection metrics computed directly off an attention matrix.

`compute_step_trajectory.py` derives these same metrics from a cached
``attn_layer_avg.npy``. That round trip dominates the pipeline's attention
stage: for an 18k-token trajectory the (seq, seq) matrix is 648 MB on disk,
and simply loading it and widening it to float32 costs 191 s — against 25 s
for the metric that actually feeds the classifier. Measured on QwQ-32B:

    seq=18433   load 190.9s   ratios 58.6s   consistency 25.3s (x5 variants)
    seq= 4162   load   9.1s   ratios  4.4s   consistency  1.4s
    seq=  918   load   0.0s   ratios  0.0s   consistency  0.1s

Every one of those numbers is spent moving a matrix that was already resident
in GPU memory a moment earlier. This module lets the extractor compute the
metrics while it still holds it, so the matrix never has to be written at all.

Two things make that a rewrite rather than a port:

1. **All five consistency variants share one aggregation.** Each one re-ran
   the same per-row top-p selection and per-token-string grouping, five times
   over. They differ only in how they read `(prompt_cum, gen_cum)` afterwards.
   Aggregating once and taking five readings removes that 5x duplication
   independently of where the code runs.

2. **The per-row Python loop becomes a batched scatter.** Grouping attention
   by token *string* is a segmented sum, so mapping each position to a compact
   token id turns the dict accumulation into `scatter_add_` over a chunk of
   query rows at a time.

Numerics are held to the cached path on purpose: `attn_dtype="fp16"` rounds
the layer average through float16 exactly as `np.save`/`np.load` did, so
metrics computed here stay comparable with every step_metrics file already on
disk. Pass "fp32" only when regenerating a whole dataset.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch

CONSISTENCY_KEYS = ("combined", "independent", "union", "common", "gen")


def per_token_ratios(attn: torch.Tensor, prompt_len: int) -> dict[str, torch.Tensor]:
    """sum_ratio / mean_ratio / prompt_fraction / prompt_mean_fraction per gen row.

    Mirrors `_compute_per_token_ratios`: each generation row is normalised to
    sum 1, then prompt-side mass is read off directly. The causal mask makes
    the `[prompt_len:, prompt_len:]` slice safe to sum whole — entries above
    the diagonal are zero and contribute nothing.
    """
    seq_len = attn.shape[0]
    n_gen = seq_len - prompt_len
    gen = attn[prompt_len:, :].to(torch.float32)
    row_sums = gen.sum(dim=1, keepdim=True)
    gen = gen / torch.where(row_sums > 0, row_sums, torch.ones_like(row_sums))

    prompt_sums = gen[:, :prompt_len].sum(dim=1)
    gen_sums = gen[:, prompt_len:].sum(dim=1)
    gen_lens = torch.arange(1, n_gen + 1, device=attn.device, dtype=torch.float32)
    total_sums = prompt_sums + gen_sums
    total_lens = prompt_len + gen_lens

    inf = torch.tensor(float("inf"), device=attn.device)
    nan = torch.tensor(float("nan"), device=attn.device)
    prompt_mean = prompt_sums / prompt_len if prompt_len > 0 else torch.zeros_like(prompt_sums)
    gen_mean = gen_sums / gen_lens
    return {
        "sum_ratio": torch.where(gen_sums > 0, prompt_sums / gen_sums, inf),
        "mean_ratio": torch.where(gen_mean > 0, prompt_mean / gen_mean, inf),
        "prompt_fraction": torch.where(total_sums > 0, prompt_sums / total_sums, nan),
        "prompt_mean_fraction": torch.where(
            (total_sums > 0) & (prompt_len > 0),
            prompt_mean / (total_sums / total_lens), nan),
    }


def _token_ids(tokens: Sequence[str], device: torch.device) -> tuple[torch.Tensor, int]:
    """Map token strings to compact ids so grouping becomes a scatter_add."""
    lookup: dict[str, int] = {}
    ids = []
    for tok in tokens:
        idx = lookup.get(tok)
        if idx is None:
            idx = len(lookup)
            lookup[tok] = idx
        ids.append(idx)
    return torch.tensor(ids, dtype=torch.long, device=device), len(lookup)


def consistency_bundle(
    attn: torch.Tensor, prompt_len: int, tokens: Sequence[str],
    threshold: float = 0.9, chunk: int = 256,
) -> dict[str, torch.Tensor]:
    """All five attention-consistency variants, plus the intersection size.

    For every generation row: normalise, keep the smallest prefix of the
    descending order whose cumulative mass reaches `threshold`, then group the
    kept mass by token string into prompt-side `P` and gen-side `G`. The five
    variants are five readings of that pair — see `compute_step_trajectory` for
    the definitions each one implements.
    """
    seq_len = attn.shape[0]
    n_gen = seq_len - prompt_len
    device = attn.device
    tok_ids, n_tok = _token_ids(tokens, device)

    out = {k: torch.full((n_gen,), float("nan"), dtype=torch.float64, device=device)
           for k in CONSISTENCY_KEYS}
    inter_size = torch.zeros(n_gen, dtype=torch.int32, device=device)
    positions = torch.arange(seq_len, device=device).unsqueeze(0)

    for start in range(0, n_gen, chunk):
        stop = min(start + chunk, n_gen)
        rows = attn[prompt_len + start:prompt_len + stop, :].to(torch.float64)
        totals = rows.sum(dim=1, keepdim=True)
        live = totals.squeeze(1) > 0
        rows = rows / torch.where(totals > 0, totals, torch.ones_like(totals))

        # Smallest top-p prefix reaching `threshold`, matching numpy's
        # searchsorted(csum, threshold) + 1 on a stable descending sort.
        sorted_vals, sorted_idx = torch.sort(rows, dim=1, descending=True, stable=True)
        keep_count = (sorted_vals.cumsum(dim=1) < threshold).sum(dim=1) + 1
        keep_sorted = (positions < keep_count.unsqueeze(1)).to(rows.dtype)
        kept = rows * torch.zeros_like(rows).scatter_(1, sorted_idx, keep_sorted)

        width = stop - start
        index = tok_ids.unsqueeze(0).expand(width, -1)
        prompt_side = kept.clone(); prompt_side[:, prompt_len:] = 0
        gen_side = kept.clone(); gen_side[:, :prompt_len] = 0
        zeros = torch.zeros(width, n_tok, dtype=rows.dtype, device=device)
        P = zeros.scatter_add(1, index, prompt_side)
        G = zeros.scatter_add(1, index, gen_side)

        p_tot = P.sum(dim=1)
        g_tot = G.sum(dim=1)
        valid = live & (p_tot > 0) & (g_tot > 0)
        common = (P > 0) & (G > 0)
        inter_size[start:stop] = common.sum(dim=1).to(torch.int32)

        safe_p = torch.where(p_tot > 0, p_tot, torch.ones_like(p_tot)).unsqueeze(1)
        safe_g = torch.where(g_tot > 0, g_tot, torch.ones_like(g_tot)).unsqueeze(1)
        zero = torch.zeros((), dtype=rows.dtype, device=device)
        # |prop_p - prop_g| under independent per-side normalisation; variants
        # 2, 3 and 5 are all sums over different supports of this same term.
        diff = (P / safe_p - G / safe_g).abs()
        shared = torch.where(common, diff, zero).sum(dim=1)

        combined = 1.0 - torch.where(common, (P - G).abs(), zero).sum(dim=1) / (p_tot + g_tot)
        independent = 1.0 - shared
        union = 2.0 - diff.sum(dim=1)
        gen_only = (G > 0) & (P <= 0)
        gen_centric = 1.0 - (shared + torch.where(gen_only, G / safe_g, zero).sum(dim=1))

        # Intersection-normalised: denominators are the shared mass only.
        p_common = torch.where(common, P, zero).sum(dim=1)
        g_common = torch.where(common, G, zero).sum(dim=1)
        has_common = valid & (p_common > 0) & (g_common > 0)
        safe_pc = torch.where(p_common > 0, p_common, torch.ones_like(p_common)).unsqueeze(1)
        safe_gc = torch.where(g_common > 0, g_common, torch.ones_like(g_common)).unsqueeze(1)
        common_norm = 1.0 - torch.where(
            common, (P / safe_pc - G / safe_gc).abs(), zero).sum(dim=1)

        nan = torch.tensor(float("nan"), dtype=rows.dtype, device=device)
        for key, value, mask in (("combined", combined, valid),
                                 ("independent", independent, valid),
                                 ("union", union, valid),
                                 ("common", common_norm, has_common),
                                 ("gen", gen_centric, valid)):
            out[key][start:stop] = torch.where(mask, value, nan)

    out["intersection_size"] = inter_size
    return out


def _as_list(values: torch.Tensor, ndigits: int = 6) -> list[float | None]:
    """Finite values rounded like the cached path; NaN/inf become null."""
    finite = torch.isfinite(values)
    data = values.detach().to("cpu")
    keep = finite.detach().to("cpu")
    return [round(float(v), ndigits) if k else None
            for v, k in zip(data.tolist(), keep.tolist())]


def step_metrics(
    attn: torch.Tensor, prompt_len: int, tokens: Sequence[str],
    threshold: float = 0.9, chunk: int = 256, attn_dtype: str = "fp16",
) -> dict[str, Any]:
    """Every per-position array `compute_step_trajectory` writes, as JSON values.

    `attn_dtype="fp16"` reproduces the precision loss of the cached path, where
    the layer average was stored as float16. Keeping that default means metrics
    from this fast path can be compared against, and mixed with, step_metrics
    files produced before it existed.
    """
    if attn_dtype == "fp16":
        attn = attn.to(torch.float16).to(torch.float32)
    elif attn_dtype != "fp32":
        raise ValueError(f"attn_dtype must be fp16 or fp32, got {attn_dtype!r}")

    ratios = per_token_ratios(attn, prompt_len)
    cons = consistency_bundle(attn, prompt_len, tokens, threshold=threshold, chunk=chunk)
    return {
        "sum_ratio": _as_list(ratios["sum_ratio"]),
        "mean_ratio": _as_list(ratios["mean_ratio"]),
        "prompt_fraction": _as_list(ratios["prompt_fraction"]),
        "prompt_mean_fraction": _as_list(ratios["prompt_mean_fraction"]),
        "attn_consistency": _as_list(cons["combined"]),
        "attn_consistency_independent": _as_list(cons["independent"]),
        "attn_consistency_union": _as_list(cons["union"]),
        "attn_consistency_common": _as_list(cons["common"]),
        "attn_consistency_gen": _as_list(cons["gen"]),
        "intersection_size": [int(x) for x in cons["intersection_size"].detach().cpu().tolist()],
    }


def build_step_metrics(
    attn: torch.Tensor, prompt_len: int, tokens: Sequence[str],
    record: dict[str, Any], category: str, sample_id: int, seq_len: int,
    attn_dtype: str = "fp16", chunk: int = 256,
    max_thinking: int = 8192, include_capped: bool = False,
    nonloop_rep_as_stop: bool = False, threshold: float = 0.9,
) -> dict[str, Any] | None:
    """One sample's step_metrics record, byte-compatible with the cached path.

    Returns None for samples `compute_one_sample` would also skip, so the fast
    path selects exactly the same trajectories. The eligibility rules and the
    loop-iteration bookkeeping below are deliberately imported from
    `compute_step_trajectory` rather than restated, so the two paths cannot
    drift apart.
    """
    from recur_code.visualization.reasoning_chat.compute_step_trajectory import (
        _find_loop, _last_matching_iteration, _step_query_positions,
    )

    is_loop_sample = (category in ("repetitive_reasoning", "repetitive_string")
                      and bool(record.get("loop")))
    is_nonloop_rep = (nonloop_rep_as_stop and category == "repetitive_reasoning"
                      and not bool(record.get("loop")))
    # 补齐 test 时补进攻击子集的未成环轨迹带人工真值 `label`（concise/productive），
    # 按良性停止样本处理；指标文件名仍沿用所在 records 的类别。
    is_padding_stop = record.get("label") in ("concise", "productive")
    is_stop_sample = (
        (category in ("concise_reasoning", "productive_reasoning", "productive_reflection")
         or is_padding_stop)
        and (include_capped or (record.get("thinking_tokens") or 0) < max_thinking)
    ) or is_nonloop_rep
    if not (is_loop_sample or is_stop_sample):
        return None

    out: dict[str, Any] = {
        "category": category, "id": sample_id,
        "is_stop_sample": is_stop_sample, "is_loop_sample": is_loop_sample,
        "prompt_len": prompt_len, "full_cached_seq_len": seq_len,
        "gen_start_q": prompt_len, "gen_end_q": seq_len - 1,
    }
    out.update(step_metrics(attn, prompt_len, tokens,
                            threshold=threshold, chunk=chunk, attn_dtype=attn_dtype))
    out.update({
        "step_query_positions": _step_query_positions(list(tokens), prompt_len),
        "endpoint_q": (seq_len - 1) if is_stop_sample else None,
        "loop_start_block": None, "loop_period_blocks": None,
        "loop_last_matching_iteration": None,
        "loop_first_iter_query_q": None, "loop_last_iter_query_q": None,
    })

    if is_loop_sample:
        blocks = (record.get("thinking") or "").split("\n\n")
        loop_start, loop_period = _find_loop(blocks)
        if loop_start is not None:
            step_end_positions = [i for i in range(prompt_len, len(tokens))
                                  if "\n\n" in tokens[i]]
            last_match = _last_matching_iteration(
                blocks, len(step_end_positions), loop_start, loop_period)
            out["loop_start_block"] = loop_start
            out["loop_period_blocks"] = loop_period
            out["loop_last_matching_iteration"] = last_match

            def _block_idx_to_query_q(block_idx: int) -> int | None:
                if block_idx >= len(step_end_positions):
                    return None
                q = step_end_positions[block_idx] - 1
                while q > prompt_len and "\n\n" in tokens[q]:
                    q -= 1
                return q if q > prompt_len else None

            out["loop_first_iter_query_q"] = _block_idx_to_query_q(loop_start + loop_period - 1)
            if last_match > 0:
                out["loop_last_iter_query_q"] = _block_idx_to_query_q(
                    loop_start + last_match * loop_period - 1)
    return out
