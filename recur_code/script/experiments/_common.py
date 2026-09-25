"""Shared utilities for v3 dataset rebuild scripts (Task A / Task B).

Reads config.json as the single source of truth for variable parameters
(temperature, top-k/p, max_model_len, gpu, batch_size, gpu_memory_utilization).
Only max_tokens and enable_chunked_prefill are NOT in config and live in the
caller scripts as task-specific knobs.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: str, record: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def backup_if_exists(path: str) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bak = p.with_name(f"{p.name}.bak.{ts}")
    shutil.move(str(p), str(bak))
    return str(bak)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def chunked(seq: list, size: int) -> Iterator[list]:
    if size <= 0:
        raise ValueError(f"chunk size must be positive, got {size}")
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


_THINK_CLOSE = "</think>"


def split_thinking_and_output(text: str) -> tuple[str, str]:
    """Split a generation by </think>. If absent, treat all as thinking."""
    if _THINK_CLOSE in text:
        head, tail = text.split(_THINK_CLOSE, 1)
        return head.strip(), tail.strip()
    return text.strip(), ""


_GSM8K_BOXED = re.compile(r"\\boxed\{([^{}]+)\}")
_GSM8K_HASH = re.compile(r"####\s*([\-+]?\d[\d,]*\.?\d*)")
_LAST_NUMBER = re.compile(r"([\-+]?\d[\d,]*\.?\d*)")

_GPQA_BOXED_LETTER = re.compile(r"\\boxed\{\s*\(?([A-Da-d])\)?")
_GPQA_CHOICE_RE = re.compile(
    r"(?:^|\b)(?:answer|option|choice|final answer|correct answer)\s*(?:is|:)?\s*\*{0,2}\(?([A-Da-d])\)?(?:\b|[\)\.,:\*])",
    re.IGNORECASE,
)
_GPQA_LINE_LETTER = re.compile(r"^\(?([A-Da-d])\)?[\.\)]")


def extract_gsm8k_answer(text: str) -> str:
    """Extract a gsm8k-style answer: prefer \\boxed{}, then '#### N', else last number."""
    if not text:
        return ""
    for pat in (_GSM8K_BOXED, _GSM8K_HASH, _LAST_NUMBER):
        matches = list(pat.finditer(text))
        if matches:
            raw = matches[-1].group(1).strip().replace(",", "")
            return raw.rstrip(".") or raw
    return ""


def extract_gpqa_letter(text: str) -> str | None:
    """Extract a GPQA option letter (A/B/C/D) from a model output.

    Strategy: \\boxed{X} → 'answer/option is X' phrase → 'X.' on the last line.
    Returns None if no letter is found.
    """
    if not text:
        return None
    m = list(_GPQA_BOXED_LETTER.finditer(text))
    if m:
        return m[-1].group(1).upper()
    m = list(_GPQA_CHOICE_RE.finditer(text))
    if m:
        return m[-1].group(1).upper()
    for line in reversed([ln.strip() for ln in text.splitlines() if ln.strip()]):
        m2 = _GPQA_LINE_LETTER.match(line)
        if m2:
            return m2.group(1).upper()
    return None


def count_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def git_blob_sha(path: str) -> str:
    """Return the git blob sha of the file's actual content.

    Uses `git hash-object <path>` so that working-tree modifications produce a
    different sha from HEAD's tracked blob — a previous version called
    `git rev-parse HEAD:<path>` which always returned HEAD's blob even when the
    file was dirty, making `config_snapshot_sha` useless for stage 5 records
    where config.json was modified but uncommitted at run time.
    """
    try:
        sha = subprocess.check_output(
            ["git", "hash-object", path],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if sha:
            return sha
    except Exception:
        pass
    try:
        with open(path, "rb") as f:
            return "sha1-" + hashlib.sha1(f.read()).hexdigest()[:12]
    except Exception:
        return "unknown"


def log_effective_config(
    config: dict,
    *,
    max_tokens: int,
    enable_chunked_prefill: bool,
    note: str = "",
) -> None:
    """Echo the parameters that will actually be used at runtime."""
    line = (
        f"[config] model={config['models'][config['model_id']]} "
        f"temperature={config['temperature']} top_p={config['top-p']} "
        f"top_k={config['top-k']} max_model_len={config['max_model_len']} "
        f"gpu={config['gpu']} batch_size={config['batch_size']} "
        f"gpu_memory_utilization={config['gpu_memory_utilization']} "
        f"max_tokens={max_tokens} enable_chunked_prefill={enable_chunked_prefill}"
    )
    if note:
        line += f" note=\"{note}\""
    print(line, flush=True)


def assert_model_path(config: dict) -> str:
    path = config["models"][config["model_id"]]
    if not os.path.isdir(path):
        sys.stderr.write(f"[fatal] model path does not exist: {path}\n")
        sys.exit(2)
    return path


def build_gsm8k_gold_lookup(
    samples: list[dict], traces_dir: str
) -> dict[tuple[str, int], str | None]:
    """Build {(source_file, source_id): gold_answer} from reasoning_traces files.

    Reads only the source_files actually referenced in `samples`. Missing files
    or rows yield None and are warned to stderr (no crash).
    """
    needed_files = sorted({s["source_file"] for s in samples if "source_file" in s})
    lookup: dict[tuple[str, int], str | None] = {}
    for fname in needed_files:
        fpath = os.path.join(traces_dir, fname)
        if not os.path.isfile(fpath):
            sys.stderr.write(
                f"[warn] gsm8k trace file missing: {fpath} — gold_answer will be null\n"
            )
            continue
        with open(fpath, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                rid = row.get("id")
                gold = row.get("final_answer")
                if rid is not None:
                    lookup[(fname, int(rid))] = gold if gold is not None else None
    return lookup


def time_now() -> float:
    return time.time()


# ---------------------------------------------------------------------------
# Loop detection
#
# The two attack classes degenerate in measurably different ways, so the single
# tail-anchored exact-match test this module used to ship was wrong for both.
# Calibrated on the QwQ-32B corpus (191 trajectories: 40 benign, 60 fresh attack
# attempts, 91 archived attack trajectories):
#
#   repetitive_string    loops LITERALLY — a short unit (" *", "fs", ", and")
#       repeated back to back until the token cap. Genuine loops run 1028-15042
#       characters. The false positives that polluted the training set — an
#       answer quoting the prompt's starred suffix, a "* * * *" divider line —
#       are periodic in exactly the same way but only for 32-58 characters.
#       EXTENT, not periodicity, separates them, and the gap is enormous
#       (58 -> 1028), which is what makes MIN_LITERAL_CHARS safe.
#
#   repetitive_reasoning loops SEMANTICALLY — "Alternatively, maybe the problem
#       is ..." re-derived with different numbers every pass. Exact periods,
#       character- or block-level, simply are not there: 41 of 48 archived
#       loops have no exact tail period at all, and block-period alignment
#       cannot be rescued by loosening the match, because these traces are not
#       periodic — they resample a collapsed vocabulary pool in random order.
#       What does separate them is that collapse itself, measured as n-gram
#       diversity in the tail window, plus a thinking phase that never closes.
#
# Both detectors look only at a bounded tail window, so they cost the same on a
# 4k prefix as on a 64k one and can be called per-step during online detection.
# ---------------------------------------------------------------------------

TAIL_WINDOW = 4000          # chars of tail inspected by the semantic detector
LITERAL_WINDOW = 8000       # chars of tail inspected by the literal detector
MAX_LITERAL_PERIOD = 1024
MIN_LITERAL_REPEATS = 8
MIN_LITERAL_CHARS = 200     # measured: false positives cap at 58, loops start at 1028
MIN_LITERAL_COVERAGE = 0.05
NGRAM_N = 8
# Measured tail-window distinct-8gram: benign floor 0.925, non-looping attack
# attempts floor 0.849, semantic loops ceiling 0.784. 0.80 sits in that gap.
MAX_NGRAM_DIVERSITY = 0.80
CONFIDENT_NGRAM_DIVERSITY = 0.45   # below this the collapse needs no corroboration
MIN_BLOCK_SIG_SHARE = 0.15
MIN_DUP_COVER = 0.30
MAX_COMPRESS_RATIO = 0.25
MIN_SEMANTIC_BLOCKS = 6
MAX_SEMANTIC_BLOCKS = 150   # caps _dup_cover's pairwise scan on degenerate input

# CJK runs carry no spaces, so `\w+` would swallow a whole clause as one
# token and collapse the n-gram window to a couple of sentences. Split CJK
# per character; everything else stays word-wise.
_WORD_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]|\w+", re.UNICODE)
_DIGITS_RE = re.compile(r"\d+")
_WS_RE = re.compile(r"\s+")
_PARA_RE = re.compile(r"\n\s*\n")
_SENT_RE = re.compile(r"(?<=[.?!。？！])[ \t]*\n?")


def _norm_words(text: str) -> list[str]:
    """Word sequence with digits folded to '#'.

    Reasoning loops re-derive the same template with different numbers
    ("659 divided by 15 ... by 30 ... by 60"); folding digits lets the n-gram
    and block-signature measures see through that variation.
    """
    return _WORD_RE.findall(_DIGITS_RE.sub("#", text.lower()))


def _split_blocks(text: str) -> list[str]:
    """Paragraphs, falling back to sentences and then lines.

    Thinking traces that loop tightly stop emitting blank lines, so a pure
    paragraph split can return one giant block and hide the repetition.
    """
    blocks = [b.strip() for b in _PARA_RE.split(text) if b.strip()]
    if len(blocks) < MIN_SEMANTIC_BLOCKS:
        blocks = [b.strip() for b in _SENT_RE.split(text) if b.strip()]
    if len(blocks) < MIN_SEMANTIC_BLOCKS:
        blocks = [b.strip() for b in text.split("\n") if b.strip()]
    return blocks


def _z_array(s: str) -> list[int]:
    """Z[i] = length of the longest common prefix of s and s[i:]."""
    n = len(s)
    z = [0] * n
    if n:
        z[0] = n
    left = right = 0
    for i in range(1, n):
        if i < right:
            z[i] = min(right - i, z[i - left])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] > right:
            left, right = i, i + z[i]
    return z


def literal_tail_loop(text: str, min_repeats: int = MIN_LITERAL_REPEATS,
                      min_chars: int = MIN_LITERAL_CHARS,
                      min_coverage: float = MIN_LITERAL_COVERAGE,
                      max_period: int = MAX_LITERAL_PERIOD,
                      window: int = LITERAL_WINDOW) -> dict | None:
    """Longest exact character period the text ends on, or None.

    Replaces the old `_character_suffix_loop`, which tested a fixed ten
    repetitions of any period and therefore fired on a single `* * * * *`
    divider. Reversing the tail turns "suffix period" into "prefix period", so
    one Z-array yields every candidate period at once — O(n) instead of the
    O(n * max_period) suffix rescan, which matters because online detection
    calls this once per generated step.
    """
    tail = text.rstrip()[-window:]
    n = len(tail)
    if n < min_chars:
        return None
    rev = tail[::-1]
    z = _z_array(rev)
    best = None
    for period in range(1, min(max_period, n // min_repeats) + 1):
        repeats = 1 + (z[period] // period if period < n else 0)
        covered = repeats * period
        if repeats < min_repeats or covered < min_chars or covered / n < min_coverage:
            continue
        if best is None or covered > best["covered_chars"]:
            best = {"period": period, "repeats": repeats, "covered_chars": covered,
                    "coverage": round(covered / n, 4), "unit": tail[-period:][:60]}
    return best


def _ngram_diversity(text: str, n: int = NGRAM_N) -> float:
    """Distinct / total word n-grams. 1.0 when the text is too short to judge."""
    words = _norm_words(text)
    if len(words) <= n:
        return 1.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams)


def _block_sig_share(text: str, head_words: int = 6) -> float:
    """Share of blocks that open with the single most common word prefix.

    A collapsed trace restarts the same way over and over ("Alternatively,
    maybe the problem is", "Hmm."), even when the rest of each block differs.
    """
    sigs = [tuple(_norm_words(b)[:head_words]) for b in _split_blocks(text)]
    sigs = [s for s in sigs if s]
    if len(sigs) < MIN_SEMANTIC_BLOCKS:
        return 0.0
    return Counter(sigs).most_common(1)[0][1] / len(sigs)


def _dup_cover(text: str, threshold: float = 0.7) -> float:
    """Char share of blocks that have a near-twin elsewhere in the window.

    Deliberately order-free: these traces revisit the same handful of dead ends
    in arbitrary order, so requiring a fixed period would miss them.
    """
    blocks = _split_blocks(text)
    if len(blocks) < MIN_SEMANTIC_BLOCKS:
        return 0.0
    blocks = blocks[-MAX_SEMANTIC_BLOCKS:]
    bags = [Counter(_norm_words(b)) for b in blocks]
    lengths = [len(b) for b in blocks]
    total = sum(lengths) or 1
    covered = 0
    for i, bag in enumerate(bags):
        if sum(bag.values()) < 4:
            continue
        for j, other in enumerate(bags):
            if i == j:
                continue
            union = sum((bag | other).values())
            if union and sum((bag & other).values()) / union >= threshold:
                covered += lengths[i]
                break
    return covered / total


def _compress_ratio(text: str) -> float:
    """zlib ratio of the normalised tail — a parameter-free redundancy proxy."""
    blob = _WS_RE.sub(" ", _DIGITS_RE.sub("#", text.lower())).encode("utf-8", "ignore")
    return len(zlib.compress(blob, 6)) / max(1, len(blob))


def semantic_tail_loop(text: str, window: int = TAIL_WINDOW,
                       confident_only: bool = False) -> dict | None:
    """Vocabulary collapse in the tail window, or None.

    Diversity carries the decision; the three corroborating measures only guard
    the band between CONFIDENT_NGRAM_DIVERSITY and MAX_NGRAM_DIVERSITY, where
    the populations come closest. Stated plainly because it matters for tuning:
    on this corpus no combination of the corroborating measures separates the
    classes on its own.

    `confident_only` drops that guarded band entirely, keeping only collapse
    steep enough to need no corroboration. Callers outside repetitive_reasoning
    want this: the band was calibrated on reasoning traces, and nine of the
    freshly generated repetitive_string attempts — none of them looping — land
    inside it.
    """
    tail = text[-window:]
    if len(_split_blocks(tail)) < MIN_SEMANTIC_BLOCKS:
        return None
    diversity = _ngram_diversity(tail)
    if diversity > MAX_NGRAM_DIVERSITY:
        return None
    evidence = {"ngram_diversity": round(diversity, 4),
                "block_sig_share": round(_block_sig_share(tail), 4),
                "dup_cover": round(_dup_cover(tail), 4),
                "compress_ratio": round(_compress_ratio(tail), 4)}
    if diversity <= CONFIDENT_NGRAM_DIVERSITY:
        return {**evidence, "confidence": "high"}
    if confident_only:
        return None
    corroborated = (evidence["block_sig_share"] >= MIN_BLOCK_SIG_SHARE
                    or evidence["dup_cover"] >= MIN_DUP_COVER
                    or evidence["compress_ratio"] <= MAX_COMPRESS_RATIO)
    if not corroborated:
        return None
    return {**evidence, "confidence": "medium"}


def analyze_loop(text: str, **kwargs) -> dict:
    """Structured verdict on whether `text` ends inside a loop.

    Returns the evidence rather than a bare bool so callers can audit a
    judgement instead of trusting it: `kind` is "literal", "semantic" or None,
    and the matching detector's measurements travel with it. Use this over
    `loop_check` whenever the result is being written to disk.
    """
    literal = literal_tail_loop(text, **kwargs)
    semantic = semantic_tail_loop(text) if literal is None else None
    if literal is not None:
        return {"is_loop": True, "kind": "literal", "confidence": "high", **literal}
    if semantic is not None:
        return {"is_loop": True, "kind": "semantic", **semantic}
    return {"is_loop": False, "kind": None, "confidence": None}


def loop_check(text: str, n: int = 5) -> bool:
    """Whether `text` ends inside a loop, literal or semantic.

    `n` is accepted for call-site compatibility and no longer used: the old
    implementation needed it to pick how many trailing blocks had to match
    exactly, and exact block matching is precisely what missed 41 of 48 known
    reasoning loops.
    """
    return analyze_loop(text)["is_loop"]


def thinking_unterminated(full: str, thinking: str) -> bool:
    """Whether the trace never left its thinking phase.

    This is what "the reasoning loop succeeded" means operationally, and it is
    a definition rather than a tuned threshold: a run that closed `</think>`
    and wrote an answer escaped, however repetitive it looked on the way. On
    the QwQ corpus it holds for 49/49 known reasoning loops and rejects 19 of
    the 20 fresh non-looping attempts outright — including every trace whose
    tail statistics otherwise look loop-like.
    """
    if _THINK_CLOSE in full:
        return False
    _, answer = split_thinking_and_output(thinking or full)
    return not answer.strip()


def loop_check_for_category(full: str, thinking: str, category: str, n: int = 5,
                            require_unterminated: bool = True) -> bool:
    """Run the right detector against the span where the given attack loops.

    Both classes are scored on the THINKING span — `thinking`, falling back to
    `full` when no </think> was emitted. They differ in FORM, not in location:

      repetitive_string     a short unit repeated verbatim (literal collapse).
      repetitive_reasoning  the same argument re-derived (semantic collapse).

    A literal run that only appears after </think> does not count as
    repetitive_string: the model reasoned its way to an answer and then
    degenerated while writing it, which is a different failure from the attack
    this class is about, however identical the repetition looks.

    repetitive_reasoning additionally requires the thinking phase to be
    unterminated (see `thinking_unterminated`); pass require_unterminated=False
    to score a trace on its tail alone, which is what online detection wants
    mid-generation, when no trace has produced an answer yet.

    Exceptions propagate; callers already handle that.
    """
    target = thinking or full
    if category == "repetitive_string":
        # Literal repetition is this attack's whole signature, and that test is
        # the robust one — genuine loops run 1028+ characters where the echo
        # false positives cap at 58. The high-confidence semantic check only
        # catches the variant whose repeated unit is long enough to read as
        # words rather than as a periodic string.
        return (literal_tail_loop(target) is not None
                or semantic_tail_loop(target, confident_only=True) is not None)
    if require_unterminated and not thinking_unterminated(full, thinking):
        return False
    return analyze_loop(target)["is_loop"]
