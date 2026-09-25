#!/usr/bin/env python3
"""Build, train, and offline-test a four-class detector for one local model.

Run this script with the analysis environment (``Recur`` on this machine).
It starts vLLM through ``--vllm-python`` for generation, then uses full
completion streams (thinking *and* answer) for attention and position-wise
detection. Every stage is persisted below ``--out`` and may be rerun with
``--stage`` after an interruption.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
RC = REPO / "recur_code"
sys.path.insert(0, str(RC / "script/experiments"))
from _common import append_jsonl, loop_check_for_category, split_thinking_and_output  # noqa: E402

# Each class draws from one or more subsets of ``dataset/<model>/<category>/``.
# A class is a *label*, a subset is a *source*: the four labels are what the
# detector predicts, while the subsets say which corpora that label was measured
# on.  Listing several here makes the benign side broader without inventing new
# classes — a concise answer to a GSM8k word problem and to a SimpleQA lookup
# are the same behaviour on different material, and a detector that only ever
# saw GSM8k cannot be said to have been tested on "concise reasoning".
BENIGN = {
    "concise_reasoning": {"label": "concise", "datasets": [
        "concise_reasoning/GSM8k/gsm8k_test.jsonl",
        "concise_reasoning/MMLU/high_school_geography_test.jsonl",
        "concise_reasoning/SimpleQA/simpleqa_test.jsonl",
    ]},
    "productive_reasoning": {"label": "productive", "datasets": [
        "productive_reasoning/GPQA/gpqa_main.jsonl",
        "productive_reasoning/MMLU_Econometrics/econometrics_test.jsonl",
        "productive_reasoning/MMLU_World_History/high_school_world_history_test.jsonl",
    ]},
}
ATTACK = {
    "repetitive_reasoning": {"label": "repetitive_reasoning", "datasets": [
        "repetitive_reasoning/Recur/prompts.jsonl",
        "repetitive_reasoning/MiP/prompts.jsonl",
    ]},
    "repetitive_string": {"label": "repetitive_string", "datasets": [
        "repetitive_string/LoopLLM/prompts.jsonl",
        "repetitive_string/GCG/prompts.jsonl",
        "repetitive_string/concat/prompts.jsonl",
    ]},
}
CLASSES = ["concise", "productive", "repetitive_reasoning", "repetitive_string"]
# Subsets whose rows are a rendered context rather than a user turn.  ``concat``
# stores the complete chat-template string with the assistant turn already open
# and ``<think>`` seeded with a repeated word; the seeding IS the attack, so the
# row is continued verbatim.  Wrapping it in a user message — what every other
# subset needs — would bury the template inside a user turn, reopen the
# assistant turn after it, and leave the seeding inert.
PRE_RENDERED = {"concat"}
# Stand-in for the user content when asking a tokenizer what its own template
# puts around it.  Any byte sequence no prompt contains will do.
_TEMPLATE_PROBE = "\x00user-content\x00"
# Same wording as ``src/data/prepare_gpqa_dataset.build_question``, so every
# multiple-choice subset reaches the model in one format.
MCQ_INSTRUCTION = "Choose the correct answer from the following options."
MCQ_CLOSING = "Respond with the option letter and the answer text."


def subsets_of(category: str) -> list[str]:
    """The ``dataset/<model>/<category>/<subset>/`` directories this class draws from.

    Derived from the configured dataset paths rather than stated twice, so the
    subset a record carries can never drift from the file it was generated from.
    """
    return [p.split("/")[1] for p in (BENIGN | ATTACK)[category]["datasets"]]


def dataset_rel(category: str, subset: str) -> str:
    """The dataset file this (class, subset) pair reads, relative to --dataset-root."""
    for path in (BENIGN | ATTACK)[category]["datasets"]:
        if path.split("/")[1] == subset:
            return path
    raise KeyError(f"{category} 下没有配置数据集 {subset}")


def planned_tasks(categories: str | None, subsets: str | None) -> list[tuple[str, str]]:
    """(class, subset) pairs to run, in configuration order, after both filters."""
    want_cat = set(categories.split(",")) if categories else None
    want_sub = set(subsets.split(",")) if subsets else None
    out = []
    for category in [*BENIGN, *ATTACK]:
        if want_cat is not None and category not in want_cat:
            continue
        for subset in subsets_of(category):
            if want_sub is not None and subset not in want_sub:
                continue
            out.append((category, subset))
    return out


def artifact_dir(out: Path, split: str, kind: str, subset: str) -> Path:
    """``<out>/<split>/<kind>/<subset>`` — every artifact sits under its dataset.

    The subset a class was drawn from is part of the output path, not just a
    field inside the records: two subsets of the same class (GSM8k and MMLU
    under ``concise_reasoning``) both number their trajectories from 0 and would
    otherwise write ``concise_reasoning_0.json`` over each other.  Mirrors
    ``dataset/<model>/<category>/<subset>/`` on the input side.
    """
    return out / split / kind / subset


def records_path(out: Path, split: str, category: str, subset: str) -> Path:
    return artifact_dir(out, split, "records", subset) / f"{category}.jsonl"


def metric_files(metric_root: Path, category: str) -> list[tuple[str, Path]]:
    """``(subset, path)`` for every per-position metric file of this class.

    ``<metric_root>/<subset>/<category>_<id>.json`` is what the pipeline writes
    now; the flat ``<metric_root>/<category>_<id>.json`` is still read so
    directories produced before the regrouping (and the frozen snapshots under
    ``_old_from_total/`` and ``exp/step_trajectory_total/``) keep working.
    Ids restart at 0 within each subset, so the sort key carries the subset too.
    """
    nested = sorted(metric_root.glob(f"*/{category}_*.json"))
    found = [(p.parent.name, p) for p in nested]
    if not found:
        fallback = subsets_of(category)[0]
        found = [(fallback, p) for p in sorted(metric_root.glob(f"{category}_*.json"))]
    return sorted(found, key=lambda sp: (sp[0], int(sp[1].stem.rsplit("_", 1)[1])))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False, allow_nan=False) + "\n" for x in rows))


def prompt_of(row: dict) -> str:
    """The user turn for this dataset row, whatever shape the corpus stores it in.

    Three shapes occur across the configured subsets and they are not
    interchangeable:

    * ``prompt`` — the attack corpora, already a complete instruction.
    * ``question`` [+ ``options``] — the multiple-choice corpora.  GPQA was
      pre-rendered by ``prepare_gpqa_dataset`` and carries the choices inside
      ``question``; the MMLU exports keep them in a separate ``options`` map, so
      they are rendered here in the same wording.  Sending the bare stem instead
      would ask "which one of the following" with nothing following, and the
      trajectory would measure the model guessing, not reasoning.
    * ``problem`` — SimpleQA's short-answer field, which is neither of the above.
    """
    explicit = row.get("prompt")
    if explicit:
        return str(explicit)
    text = str(row.get("question") or row.get("problem") or "")
    options = row.get("options")
    if text and isinstance(options, dict) and options and MCQ_INSTRUCTION not in text:
        lines = [f"{label}. {options[label]}" for label in sorted(options)]
        text = (f"{text.strip()}\n\n{MCQ_INSTRUCTION}\n" + "\n".join(lines)
                + f"\n\n{MCQ_CLOSING}")
    return text


def template_parts(render) -> tuple[str, str]:
    """What this model's chat template puts before and after the user content."""
    head, _, tail = render(_TEMPLATE_PROBE).partition(_TEMPLATE_PROBE)
    return head, tail


def prepare_source(rows: list[dict], subset: str, render) -> list[dict]:
    """Normalise a subset's rows to the one shape the sweep works in.

    A pre-rendered row arrives as a single string — template, instruction,
    assistant header and the seeded ``<think>`` span.  It is split here into the
    bare instruction (``prompt``), the context to continue (``context``) and the
    seeded span (``prefill``), for two reasons:

    * Every later stage re-derives the context as ``render(record["prompt"]) +
      record["thinking"]`` (``extract_layer_avg_attention_v2._extract``).
      Storing the row as-is would render it a second time; storing the halves
      reproduces the generated context byte for byte, and ``prompt_len`` then
      falls where the template ends — so the seeded repetition is measured as
      generated text, over the same span a LoopLLM or Recur trajectory is.
    * Prompt-keyed bookkeeping (the split-exclusion sets, the distinct-prompt
      count) then compares instructions for every subset rather than comparing
      an instruction against a whole rendered context.

    The split is the tokenizer's own template, never a hardcoded marker, and it
    has to round-trip: a row this model's template cannot reproduce is a row the
    attention stage would rebuild differently, so the run stops instead of
    generating a trajectory whose features are silently off.
    """
    if subset not in PRE_RENDERED:
        return rows
    head, tail = template_parts(render)
    prepared = []
    for row in rows:
        context = prompt_of(row)
        if not (context.startswith(head) and tail in context[len(head):]):
            raise SystemExit(
                f"[generate] 预渲染样本 {row.get('id')!r} 与本模型的对话模板对不上，"
                f"无法还原指令与预填：\n  模板前缀 {head!r}\n  模板后缀 {tail!r}\n"
                f"  样本开头 {context[:120]!r}")
        instruction, _, prefill = context[len(head):].partition(tail)
        rendered = context[: len(context) - len(prefill)]
        if render(instruction) != rendered:
            raise SystemExit(
                f"[generate] 预渲染样本 {row.get('id')!r} 的指令无法用本模型模板复原："
                f"\n  文件里是 {rendered[-80:]!r}\n  复原出 {render(instruction)[-80:]!r}")
        prepared.append({**row, "prompt": instruction, "context": context, "prefill": prefill})
    return prepared


def completion_digest(row: dict) -> str:
    """Identity of a trajectory for the distinct-completion quota.

    Keyed on the trajectory as stored (``thinking``), not on the text decoded in
    this draw: two pre-rendered rows that seed the same word to different
    lengths can decode to byte-identical continuations, and those are two
    conditions, not two copies of one draw.  For every other subset ``thinking``
    *is* the decoded text, so the quota counts exactly what it did before.
    """
    return hashlib.sha1((row.get("thinking") or row.get("raw_text") or "").encode()).hexdigest()


def dataset_rows(root: Path, category: str, subset: str, split: str,
                 benign_test_rows: int = 25) -> list[dict]:
    dataset_path = root / dataset_rel(category, subset)
    # The in-repo datasets were renamed to ``prompts.jsonl``; older local or
    # archived model dataset copies may still carry the historical filename
    # typo ``promtps.jsonl`` under Recur.  Resolve it locally rather than
    # silently substituting the different MiP corpus.
    if not dataset_path.exists() and dataset_path.name == "prompts.jsonl":
        typo_path = dataset_path.with_name("promtps.jsonl")
        if typo_path.exists():
            dataset_path = typo_path
    rows = read_jsonl(dataset_path)
    if category in BENIGN:
        # Train takes the head, test the tail.  Any ``benign_test_rows`` up to
        # ``len(rows) - 20`` stays disjoint from the training head by construction.
        return rows[:20] if split == "train" else rows[-benign_test_rows:]
    return rows


def prompts_in_files(paths: str | None, category: str | None = None) -> set[str]:
    """Prompts to keep out of a sweep, read from JSONL records.

    Used to hold a test set clear of whatever a already-trained detector was
    fitted on: point this at the detector's training-prompt listing and the
    generated trajectories are prompt-disjoint from its training set by
    construction, which is the only thing that makes the split independent for
    a classifier that is not being retrained.  Rows may carry a ``category`` to
    restrict them; rows without one apply to every category.
    """
    if not paths:
        return set()
    collected: set[str] = set()
    for raw in str(paths).split(","):
        path = Path(raw.strip())
        if not raw.strip():
            continue
        if not path.exists():
            raise SystemExit(f"[generate] --exclude-prompts-from 找不到 {path}")
        for row in read_jsonl(path):
            row_category = row.get("category")
            if category is not None and row_category is not None and row_category != category:
                continue
            prompt = str(row.get("prompt", ""))
            if prompt:
                collected.add(prompt)
    return collected


def other_split_prompts(out: Path, split: str, category: str, subset: str) -> set[str]:
    """Prompts the *other* split already spent on this category.

    Attack classes draw from the whole prompt file in both splits, so nothing
    in ``generate`` keeps train and test apart on its own.  Passing
    ``--exclude-split-prompts`` makes the disjointness a property of generation
    instead of something a later resplit pass has to repair.
    """
    other = "train" if split == "test" else "test"
    records = records_path(out, other, category, subset)
    if not records.exists():
        return set()
    return {str(r.get("prompt", "")) for r in read_jsonl(records)}


class _Completion:
    """One decoded completion, in the shape the record builder expects."""

    __slots__ = ("text", "finish_reason", "token_ids")

    def __init__(self, text: str, finish_reason: str | None, token_ids: list[int]):
        self.text, self.finish_reason, self.token_ids = text, finish_reason, token_ids


class _VLLMBackend:
    """vLLM generation — the fast path, for architectures it supports."""

    name = "vllm"

    def __init__(self, args: argparse.Namespace):
        import dataclasses  # noqa: PLC0415
        import vllm as _vllm  # type: ignore # noqa: PLC0415
        import vllm.envs as _envs  # type: ignore # noqa: PLC0415
        from vllm import LLM, SamplingParams  # type: ignore # noqa: PLC0415
        from vllm.engine.arg_utils import EngineArgs  # type: ignore # noqa: PLC0415
        from vllm.inputs import TokensPrompt  # type: ignore # noqa: PLC0415
        self._params = SamplingParams
        self._tokens_prompt = TokensPrompt
        # 两个 env 的 vLLM 差了十几个版本（0.9.0 与 0.19.1），构造参数改过名也删过。
        # 按 EngineArgs 当前真实的字段来拼，而不是写死某一版的签名 —— 老 env 仍在跑
        # 已开始的任务，新 env 才有 GLM-4.7-Flash / Qwen3.6-27B 的 kernel，两边都要能用。
        fields = {x.name for x in dataclasses.fields(EngineArgs)}
        kwargs: dict[str, Any] = {
            "model": str(args.model),
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enable_chunked_prefill": True,
        }
        # `task` 在新版里改名为 `runner`
        if "task" in fields:
            kwargs["task"] = "generate"
        elif "runner" in fields:
            kwargs["runner"] = "generate"
        # `swap_space`（CPU pinned KV cache）在新版里整个删掉了，V1 引擎不用它
        if "swap_space" in fields:
            kwargs["swap_space"] = args.swap_space
        # V0 引擎只存在于老版本；新版没有 VLLM_USE_V1 这个开关，设了也只是个无用环境变量
        if hasattr(_envs, "VLLM_USE_V1"):
            os.environ.setdefault("VLLM_USE_V1", "0")
        print(f"[generate] vLLM {getattr(_vllm, '__version__', '?')} "
              f"构造参数 {sorted(kwargs)}", flush=True)
        self._llm = LLM(**kwargs)
        self.tokenizer = self._llm.get_tokenizer()

    def render(self, prompt: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)

    def generate(self, chats: list[str], temperature: float, max_tokens: int,
                 seed: int, top_p: float | None = None,
                 pre_rendered: bool = False) -> list[_Completion]:
        kwargs = {"temperature": temperature, "max_tokens": max_tokens, "seed": seed}
        if top_p is not None:
            kwargs["top_p"] = top_p
        # vLLM's string entry point tokenises with the checkpoint's default
        # `add_special_tokens` (`inputs/preprocess.py`; only its chat entry point
        # turns it off).  A pre-rendered context already carries its specials, and
        # DeepSeek-R1-Distill-Qwen-14B declares `add_bos_token=True`, so passing
        # the string would put a second <｜begin▁of▁sentence｜> in front of the
        # one in the file.  Continuing a seeded context means continuing exactly
        # the bytes that were seeded, so encode it here instead.
        prompts: Any = chats
        if pre_rendered:
            prompts = [self._tokens_prompt(
                prompt_token_ids=self.tokenizer(c, add_special_tokens=False).input_ids)
                for c in chats]
        outputs = self._llm.generate(prompts, self._params(**kwargs), use_tqdm=False)
        return [_Completion(o.outputs[0].text, getattr(o.outputs[0], "finish_reason", None),
                            list(o.outputs[0].token_ids)) for o in outputs]


class _HFBackend:
    """transformers generation, for architectures vLLM has no kernel for.

    GLM-4.7-Flash (`glm4_moe_lite`) and Qwen3.6-27B (`qwen3_5`) are not in vLLM
    0.9.0's registry at all, so the whole generate stage would otherwise be
    unavailable to them. Prompt rendering goes through `_hf_backend`, which
    passes `enable_thinking=True` to the templates that understand it — without
    it these models emit no reasoning trace and every trajectory is unusable.
    """

    name = "hf"

    def __init__(self, args: argparse.Namespace):
        import torch  # noqa: PLC0415
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import _hf_backend as B  # noqa: PLC0415
        self._torch = torch
        self._B = B
        self._model_type = B.model_type_of(str(args.model))
        self._model, self.tokenizer = B.load_hf(str(args.model))
        # A checkpoint may declare several terminators: GLM-4.7-Flash stops on
        # <|endoftext|>, <|user|> or <|observation|>, and `generate` honours all
        # three. Reading only `tokenizer.eos_token_id` would leave the other two
        # in the completion text and mislabel a natural stop as "length".
        eos = getattr(getattr(self._model, "generation_config", None), "eos_token_id", None)
        ids = set(eos) if isinstance(eos, (list, tuple)) else ({eos} if eos is not None else set())
        ids |= {self.tokenizer.eos_token_id, self.tokenizer.pad_token_id}
        self._eos_ids = {int(i) for i in ids if i is not None}

    def render(self, prompt: str) -> str:
        return self._B.build_prompt_text(self.tokenizer, prompt, self._model_type)

    def generate(self, chats: list[str], temperature: float, max_tokens: int,
                 seed: int, top_p: float | None = None,
                 pre_rendered: bool = False) -> list[_Completion]:
        # `pre_rendered` needs nothing here: this path already encodes with
        # add_special_tokens=False and pads left (`_hf_backend.load_hf`), which
        # is what continuing a rendered context requires.  It is accepted so both
        # backends answer to one call.
        torch = self._torch
        # vLLM takes the seed per request; transformers reads global RNG state,
        # so seed here to keep a (prompt, temperature, seed) draw reproducible.
        torch.manual_seed(seed)
        enc = self.tokenizer(chats, return_tensors="pt", padding=True,
                             add_special_tokens=False).to(self._model.device)
        do_sample = temperature is not None and temperature > 0
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens, "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs["temperature"] = temperature
            # A sharply peaked distribution can reproduce the argmax path at any
            # seed: GLM-4.7-Flash returns byte-identical completions at T=0.2
            # across three seeds. Nucleus sampling truncates the tail instead of
            # rescaling it, which is what actually breaks that tie.
            if top_p is not None:
                kwargs["top_p"] = top_p
        with torch.no_grad():
            out = self._model.generate(**enc, **kwargs)
        gen = out[:, enc["input_ids"].shape[1]:]
        eos_ids = self._eos_ids
        completions = []
        for row in gen:
            ids = row.tolist()
            # Trailing pad/eos is where this sequence actually ended; anything
            # after it is batch padding and must not reach the record.
            end = len(ids)
            while end > 0 and ids[end - 1] in eos_ids:
                end -= 1
            stopped = end < len(ids)
            completions.append(_Completion(
                self.tokenizer.decode(ids[:end], skip_special_tokens=False),
                "stop" if stopped else "length", ids[:end]))
        # Release this batch's KV cache and logits before the next call. Without
        # it the allocator holds the previous batch's blocks while the next one
        # is built, and a MoE layer's expert gather then fails on a fragmented
        # heap: GLM-4.7-Flash OOMed on its *second* batch of 8 with 4.7 GiB
        # reserved-but-unallocated, having completed the first one.
        del out, enc, gen
        torch.cuda.empty_cache()
        return completions


def make_backend(args: argparse.Namespace) -> Any:
    """Pick the generation backend, defaulting to whatever the model supports."""
    choice = args.backend
    if choice == "auto":
        choice = "vllm" if _vllm_supports(args.model) else "hf"
        print(f"[generate] backend=auto -> {choice}")
    return _VLLMBackend(args) if choice == "vllm" else _HFBackend(args)


def _vllm_supports(model: str) -> bool:
    """Whether vLLM has a kernel for this checkpoint's architecture."""
    try:
        import json as _json
        arch = _json.loads((Path(model) / "config.json").read_text()).get("architectures") or []
        from vllm.model_executor.models.registry import ModelRegistry  # type: ignore
        supported = set(ModelRegistry.get_supported_archs())
        return any(a in supported for a in arch)
    except Exception:
        return False


def generate(args: argparse.Namespace, split: str) -> None:
    # The backend is built lazily so this file remains inspectable under plain Python.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    model = Path(args.model)
    backend = make_backend(args)
    tokenizer = backend.tokenizer
    base_seeds = [int(x) for x in str(args.seeds).split(",")] if args.seeds else [args.seed]
    provenance: dict[str, dict] = {}
    for category, subset in planned_tasks(args.categories, args.subsets):
        task = f"{category}/{subset}"
        pre_rendered = subset in PRE_RENDERED
        target = records_path(args.out, split, category, subset)
        if target.exists() and args.resume and not args.append:
            continue
        source = dataset_rows(args.dataset_root, category, subset, split, args.benign_test_rows)
        source = prepare_source(source, subset, backend.render)
        if args.source_limit:
            # Head of the prompt pool, taken before any exclusion so the set a run
            # draws from is stated by the file order alone and is reproducible.
            source = source[:args.source_limit]
        excluded = (other_split_prompts(args.out, split, category, subset)
                    if args.exclude_split_prompts else set())
        excluded |= prompts_in_files(args.exclude_prompts_from, category)
        # Appending adds *new* prompt groups: the trajectories already held are
        # kept with their ids intact, so attention caches keyed on those ids stay
        # valid, and their prompts are excluded so the sweep does not redraw them.
        kept = read_jsonl(target) if (args.append and target.exists()) else []
        if kept:
            excluded |= {str(row.get("prompt", "")) for row in kept}
        if excluded:
            source = [row for row in source if prompt_of(row) not in excluded]
        if category in BENIGN:
            temps, max_tokens, quota = [0.5], 16384, len(source)
        elif category == "repetitive_reasoning":
            temps, max_tokens, quota = (([0, .2, .4, .6], 16384, 20) if split == "train"
                                        else ([0, .2, .4, .6, .8, 1], 16384, 25))
        else:
            temps, max_tokens, quota = (([0], 8192, 20) if split == "train"
                                        else ([0, .2], 16384, 25))
        if args.quota:
            quota = args.quota
        elif args.exhaustive:
            # 穷尽扫的「目标」就是整个交叉积，日志里的 已收 N/quota 才有意义
            quota = len(source) * len(temps) * len(base_seeds) * args.attempts_per_temperature
        if args.max_tokens:
            max_tokens = args.max_tokens
        if args.temperatures:
            temps = [float(x) for x in str(args.temperatures).split(",")]
        # `quota` counts the records the file should end up with, so an append
        # run states the target size rather than a delta.
        generations = artifact_dir(args.out, split, "generations", subset) / f"{category}.jsonl"
        prior_attempts = read_jsonl(generations) if (args.append and generations.exists()) else []
        # Stream both files as the sweep runs. They used to be written once, after
        # the whole category finished — so an interrupted sweep lost everything it
        # had already drawn, which is how an OOM kill took out a complete benign
        # class after 45 minutes. An attack class can scan for hours; it must be
        # restartable from whatever it has.
        generations.parent.mkdir(parents=True, exist_ok=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(generations, prior_attempts)
        write_jsonl(target, kept)
        selected: list[dict] = list(kept)
        # Count the quota over DISTINCT completions. A sharply peaked model
        # returns byte-identical text across seeds — GLM-4.7-Flash gave the same
        # 48972-character trajectory at T=0.0 and at T=0.2 under three seeds —
        # so counting raw hits let three copies fill three quota slots and
        # stopped the sweep at "5/5" when only 2 independent trajectories
        # existed. Copies are not independent samples: keeping them would weight
        # one trajectory once per copy, and worse, end the search early.
        selected_digests = {completion_digest(r) for r in kept}
        # Ids continue past the highest one ever held, not past ``len(selected)``:
        # a trajectory dropped from the records may still have an attention cache
        # under its old id, and reusing that id would silently mismatch the two.
        next_id = max((int(r["id"]) for r in kept), default=-1) + 1
        # 穷尽模式下配额不再是终止条件：每个温度都要完整扫一遍全部 prompt，
        # 「收满就停」会让高温那几轮根本跑不到。配额仍记进溯源，只是不再 break。
        reached = (lambda: False) if args.exhaustive else (lambda: len(selected) >= quota)
        attempts: list[dict] = []
        # Greedy decoding ignores the seed, so a second seed at temperature 0
        # would re-decode a completion already held.  Key such draws on the
        # prompt alone and the sweep spends its budget on new trajectories.
        drawn: set[tuple] = set()
        for base_seed in base_seeds:
            for temp in temps:
                for attempt in range(args.attempts_per_temperature):
                    seed = base_seed + attempt
                    # vLLM schedules a small cohort together.  This retains one
                    # independent completion per prompt, while avoiding the very
                    # poor GPU utilisation of serial 16k-token decoding.
                    for start in range(0, len(source), args.generation_batch_size):
                        if reached():
                            break
                        batch = [(i, row) for i, row in enumerate(source[start:start + args.generation_batch_size],
                                                                  start=start)
                                 if (i, temp, None if temp == 0 else seed) not in drawn]
                        if not batch:
                            continue
                        for source_index, _ in batch:
                            drawn.add((source_index, temp, None if temp == 0 else seed))
                        prompts = [prompt_of(source_row) for _, source_row in batch]
                        # A pre-rendered row is continued as it stands; every other
                        # one is wrapped in a user turn as before.
                        chats = [source_row.get("context") or backend.render(prompt)
                                 for (_, source_row), prompt in zip(batch, prompts)]
                        outputs = backend.generate(chats, temp, max_tokens, seed, args.top_p,
                                                   pre_rendered=pre_rendered)
                        for (source_index, source_row), prompt, out in zip(batch, prompts, outputs):
                            if reached():
                                break
                            raw = out.text
                            # The seeded span belongs to the trajectory, not to the
                            # instruction: it was written into the assistant turn, and
                            # the record has to hold it because `prompt` no longer does.
                            prefill = source_row.get("prefill", "")
                            completion = prefill + raw
                            thinking, answer = split_thinking_and_output(completion)
                            # Loop verdict on the context plus the new tokens, so a
                            # continuation is judged together with the repetition it
                            # was handed — carrying a seeded loop on is the failure
                            # this subset measures, and the tail the detector reads
                            # spans both.
                            loop_text = (source_row["context"] + raw) if pre_rendered else raw
                            loop_thinking = (split_thinking_and_output(loop_text)[0]
                                             if pre_rendered else thinking)
                            # A malformed/unfinished answer must not abort a long sweep:
                            # it is simply not a usable looping trajectory.
                            try:
                                loop = (loop_check_for_category(loop_text, loop_thinking, category,
                                                                args.loop_blocks)
                                        if category in ATTACK else False)
                                loop_error = None
                            except Exception as exc:  # detector is best-effort at this stage
                                loop, loop_error = False, f"{type(exc).__name__}: {exc}"
                            row = {"id": next_id,
                                   "source_index": source_index, "source_id": source_row.get("id", source_index),
                                   "category": category, "label": (BENIGN | ATTACK)[category]["label"],
                                   "subset": subset, "model_name": model.name, "prompt": prompt, "raw_text": raw,
                                   # The attention extractor reads `thinking`; use the COMPLETE completion by design
                                   # — for a pre-rendered row that includes the seeded span the model was handed,
                                   # so `render(prompt) + thinking` is the context it actually generated in.
                                   "thinking": completion, "parsed_thinking": thinking, "output": answer,
                                   "pre_rendered": pre_rendered, "prefill": prefill,
                                   # `seed` is the value handed to SamplingParams; with `temperature` and the
                                   # prompt it is everything needed to redraw this exact trajectory.
                                   "temperature": temp, "top_p": args.top_p,
                                   "attempt": attempt, "base_seed": base_seed, "seed": seed,
                                   "loop": loop, "loop_check_error": loop_error,
                                   "finish_reason": getattr(out, "finish_reason", None), "completion_tokens": len(out.token_ids)}
                            attempts.append(row)
                            append_jsonl(str(generations), row)
                            digest = completion_digest(row)
                            duplicate = digest in selected_digests
                            if (category in BENIGN or loop) and not duplicate:
                                selected.append(row)
                                selected_digests.add(digest)
                                append_jsonl(str(target), row)
                                next_id += 1
                            note = " 内容与已收的重复，不计入配额" if (loop and duplicate) else ""
                            # Stamp each line: a sweep mixes 2-minute and 24-minute
                            # samples (a looping trajectory runs to the token cap),
                            # so without a clock the log cannot tell a slow model
                            # from one sample that simply ran long.
                            now = time.strftime("%H:%M:%S")
                            print(f"[generate {now}] {split}/{task} 第 {len(attempts)} 次: "
                                  f"prompt#{source_index} T={temp} seed={seed} "
                                  f"tok={row['completion_tokens']} {row['finish_reason']} "
                                  f"loop={loop} -> 已收 {len(selected)}/{quota}{note}", flush=True)
                    if reached():
                        break
                if reached():
                    break
            if reached():
                break
        attempts = prior_attempts + attempts
        write_jsonl(generations, attempts)
        write_jsonl(target, selected)
        provenance[task] = {
            "category": category, "subset": subset, "dataset": dataset_rel(category, subset),
            "pre_rendered": pre_rendered,
            "selected": len(selected), "quota": quota, "attempts": len(attempts),
            "kept_from_previous_run": len(kept), "appended": len(selected) - len(kept),
            "temperatures": temps, "base_seeds": base_seeds,
            "attempts_per_temperature": args.attempts_per_temperature, "max_tokens": max_tokens,
            "source_prompts_available": len(source), "excluded_other_split_prompts": len(excluded),
            "distinct_prompts_selected": len({r["prompt"] for r in selected}),
            "quota_counts_distinct_completions": True,
            "exhaustive": bool(args.exhaustive),
            "max_model_len": args.max_model_len,
            "seeds_used": sorted({r["seed"] for r in selected}),
        }
        print(f"[generate] {split}/{task}: selected {len(selected)}/{quota} from {len(attempts)} attempts "
              f"({provenance[task]['distinct_prompts_selected']} distinct prompts, "
              f"seeds {provenance[task]['seeds_used']})")
    if provenance:
        manifest = args.out / split / "generation_seeds.json"
        existing = json.loads(manifest.read_text()) if manifest.exists() else {}
        manifest.write_text(json.dumps({**existing, **provenance}, indent=2, ensure_ascii=False))
        print(f"[generate] {split}: seed provenance -> {manifest}")


def run_attention_and_metrics(args: argparse.Namespace, split: str) -> None:
    """Call existing, validated full-sequence attention and metric tools per class."""
    for category, subset in planned_tasks(getattr(args, "categories", None),
                                          getattr(args, "subsets", None)):
        attention = artifact_dir(args.out, split, "attention", subset)
        metrics = artifact_dir(args.out, split, "step_metrics", subset)
        tokens = artifact_dir(args.out, split, "tokens", subset)
        records = records_path(args.out, split, category, subset)
        rows = read_jsonl(records) if records.exists() else []
        if not rows:
            print(f"[attention] {split}/{category}/{subset}: no selected trajectories; skip")
            continue
        ids = [str(r["id"]) for r in rows]
        # Use the architecture-native, query-chunked eager implementation for
        # every supported decoder family.  The original hook extractor
        # materialises all attention heads at once and OOMs on 16k trajectories.
        extractor = "recur_code.src.analysis.extract_layer_avg_attention_v2"
        cmd = [args.analysis_python, "-m", extractor,
               "--model", args.model, "--records", *([str(records)] * len(ids)), "--ids", *ids,
               "--output-dir", str(attention), "--max-seq-len", str(args.attention_max_seq_len),
               # Preserve the full selected trajectory; v2 otherwise caps a
               # repetitive-reasoning loop at five detected repetitions.
               "--max-iters", "100000", "--q-chunk", "256", "--gpu", str(args.gpu),
               "--include-capped"]
        if args.fused_metrics:
            # Compute the per-position metrics while the layer average is still
            # on the GPU. The split path wrote a (seq, seq) matrix per sample
            # and read it straight back: 648 MB and 191 s for one 18k
            # trajectory, against 25 s for the metric the classifier uses.
            cmd += ["--metrics-dir", str(metrics), "--attn-dtype", args.attn_dtype,
                    "--metric-chunk", str(args.metric_chunk)]
            if args.save_attn:
                cmd += ["--save-attn"]
        subprocess.run(cmd, cwd=REPO, check=True)
        if args.fused_metrics:
            continue
        option = {"concise_reasoning": "--records-concise", "productive_reasoning": "--records-productive-reasoning",
                  "repetitive_reasoning": "--records-repetitive", "repetitive_string": "--records-repetitive-string"}[category]
        cmd = [args.analysis_python, "-m", "recur_code.visualization.reasoning_chat.compute_step_trajectory",
               "--cache-dir", str(attention), "--output-dir", str(metrics), "--tokens-dir", str(tokens),
               "--model", args.model, option, str(records), "--ids", *ids, "--attn-filename", "attn_layer_avg.npy",
               "--current-tokenizer", "--include-capped"]
        subprocess.run(cmd, cwd=REPO, check=True)


def allocation(lengths: list[int], target: int) -> list[int]:
    quotas = [target * n / sum(lengths) for n in lengths]
    out = [max(1, math.floor(x)) for x in quotas]
    while sum(out) < target:
        i = max(range(len(out)), key=lambda j: (quotas[j] - out[j], -j)); out[i] += 1
    while sum(out) > target:
        i = max((j for j in range(len(out)) if out[j] > 1), key=lambda j: (out[j] - quotas[j], -j)); out[i] -= 1
    return out


def build_train_rows(args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    metric_root = args.out / "train/step_metrics"
    for category, spec in (BENIGN | ATTACK).items():
        files = metric_files(metric_root, category)
        blobs = [(subset, json.loads(p.read_text())) for subset, p in files]
        if not blobs:
            raise SystemExit(f"no training metrics for {category}")
        counts = allocation([len(x["prompt_mean_fraction"]) for _, x in blobs], args.points_per_class)
        # 每类的取点预算按轨迹长度分配，与该类用了几个数据集无关：加一个数据集
        # 会把同一批点摊到更多轨迹上，而不是让这一类在训练集里变重。
        for (subset, blob), n in zip(blobs, counts):
            pmf = np.asarray(blob["prompt_mean_fraction"], float)
            cons = np.asarray(blob["attn_consistency_independent"], float)
            for seg in range(1, n + 1):
                i = math.ceil(len(pmf) * seg / n) - 1
                if i < 2 or not np.isfinite(pmf[:i + 1]).all() or not np.isfinite(cons[:i + 1]).all():
                    continue
                L = int(blob["prompt_len"]) + i + 1
                # group_id 带 subset：两个数据集的轨迹 id 都从 0 开始，不带就会
                # 把不同轨迹算成同一组，n_train_trajectories 与留出集都会错。
                rows.append({"model": Path(args.model).name, "category": category, "subset": subset,
                             "id": blob["id"], "label": spec["label"],
                             "prompt_len": blob["prompt_len"], "gen_len": len(pmf), "seg_idx": seg, "gen_pos": i, "L": L,
                             "L_feat": float(np.log(L)), "cons_ind_mean": float(np.mean(cons[:i + 1])),
                             "pmf_slope": float(np.polyfit(np.arange(i + 1), pmf[:i + 1], 1)[0]),
                             "group_id": f"{Path(args.model).name}|{category}|{subset}|{blob['id']}"})
    write_jsonl(args.out / "train/classifier_points.jsonl", rows)
    return rows


def train(args: argparse.Namespace) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if args.train_points:
        # 外部取点：从 exp/step_trajectory_total 那份已算好的多模型逐位置指标取的点
        # （build_classifier_points.py 产出，同 schema）。用于合并训练，或用一个模型
        # 已有的指标重训它的专属检测器——两种情况下 `--out` 下都不需要有 step_metrics。
        rows = read_jsonl(args.train_points)
        if not rows:
            raise SystemExit(f"[train] {args.train_points} 是空的")
        print(f"[train] 用外部取点 {args.train_points}（{len(rows)} 行 / "
              f"{len({r['group_id'] for r in rows})} 条轨迹）")
    else:
        rows = build_train_rows(args)
    X = np.asarray([[r["cons_ind_mean"], r["pmf_slope"], r["L_feat"]] for r in rows])
    y = np.asarray([r["label"] for r in rows])
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0))
    pipe.fit(X, y)
    import joblib
    bundle = {"pipeline": pipe, "labels": CLASSES, "use_L": True, "log_L": True,
              "feature_names": ["cons_ind_mean", "pmf_slope", "log_L"], "test_group_ids": [],
              "n_train_rows": len(rows), "n_train_trajectories": len({r['group_id'] for r in rows}),
              "fit_scope": "full_training_split"}
    model_dir = args.out / "model"; model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_dir / "model.joblib")
    scaler, clf = pipe.named_steps["standardscaler"], pipe.named_steps["logisticregression"]
    (model_dir / "model.json").write_text(json.dumps({"classes": [str(x) for x in clf.classes_],
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(), "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(), "use_L": True, "log_L": True,
        "feature_names": bundle["feature_names"], "test_group_ids": [], "meta": {"n_train_rows": len(rows), "n_train_trajectories": bundle["n_train_trajectories"]}}, indent=2))
    print(f"[train] {len(rows)} points -> {model_dir}")


def score_test(args: argparse.Namespace) -> None:
    cmd = [sys.executable, str(RC / "script/experiments/score_offline_detection_metrics.py"),
           "--metrics-dir", str(args.out / "test/step_metrics"), "--model", str(args.out / "model/model.json"),
           "--out", str(args.out / "test/detection_positions.jsonl")]
    subprocess.run(cmd, cwd=REPO, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model", required=True, help="local model directory")
    ap.add_argument("--dataset-root", type=Path, required=True, help="recur_code/dataset/<model>")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stage", choices=["all", "generate", "attention", "train", "detect"], default="all")
    ap.add_argument("--vllm-python", default="/root/miniconda3/envs/vllm/bin/python",
                    help="生成段用的解释器。HF-only 模型（GLM-4.7-Flash / Qwen3.6-27B）要指向 "
                         "Recur2（transformers 5.12.1），vLLM 0.9.0 没有它们的 kernel")
    ap.add_argument("--top-p", type=float, default=None,
                    help="核采样阈值。默认不传（保持历史行为）。分布尖锐的模型仅靠温度无法脱离 "
                         "argmax —— GLM-4.7-Flash 在 T=0.2 下三个种子给出字节相同的 completion")
    ap.add_argument("--backend", choices=["auto", "vllm", "hf"], default="auto",
                    help="生成后端。auto 按 config.json 的 architectures 查 vLLM 注册表，查不到就走 transformers")
    ap.add_argument("--analysis-python", default=sys.executable)
    ap.add_argument("--gpu", default="0"); ap.add_argument("--gpu-memory-utilization", type=float, default=.9)
    ap.add_argument("--max-model-len", type=int, default=32768); ap.add_argument("--attention-max-seq-len", type=int, default=32768)
    ap.add_argument("--no-fused-metrics", dest="fused_metrics", action="store_false", default=True,
                    help="退回「先落盘 (seq,seq) 注意力，再读回算指标」的两段式；只在需要复现旧产物时用")
    ap.add_argument("--save-attn", action="store_true",
                    help="融合模式下仍然保留 attn_layer_avg.npy（可视化才需要；16k 轨迹每条 648MB）")
    ap.add_argument("--attn-dtype", choices=["fp16", "fp32"], default="fp16",
                    help="算指标前层平均量化到的精度；fp16 与既有 step_metrics 可比")
    ap.add_argument("--metric-chunk", type=int, default=256)
    ap.add_argument("--attempts-per-temperature", type=int, default=1); ap.add_argument("--generation-batch-size", type=int, default=8); ap.add_argument("--loop-blocks", type=int, default=5)
    ap.add_argument("--points-per-class", type=int, default=1000); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seeds", default=None,
                    help="逗号分隔的多个基础种子，依次扫（如 0,100,200）；每条记录都写下自己用的 seed。"
                         "不传则只用 --seed")
    ap.add_argument("--swap-space", type=float, default=4,
                    help="vLLM CPU swap space (GiB)。每个实例按此 pin 主机内存；"
                         "`ulimit -l` 偏低或多实例并行时设 0，preemption 改用 recompute")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="覆盖该类别写死的生成长度上限（良性/rep_reasoning 16384、rep_string train 8192）")
    ap.add_argument("--temperatures", default=None,
                    help="逗号分隔，覆盖该类别写死的温度序列，例如 0,0.2,0.4,0.6")
    ap.add_argument("--source-limit", type=int, default=None,
                    help="只用 prompt 池的前 N 条（攻击类默认扫全池）")
    ap.add_argument("--quota", type=int, default=None,
                    help="覆盖本次生成每类的目标轨迹数（默认按指南：train 20 / test 25）")
    ap.add_argument("--benign-test-rows", type=int, default=25,
                    help="良性两类的 test 取数据集末尾多少条 prompt；train 固定取前 20，故不超过 len-20 即与 train 不重叠")
    ap.add_argument("--exclude-split-prompts", action="store_true",
                    help="生成前剔除另一个 split 已用过的 prompt，使 train/test 在生成时即按 prompt 不重叠")
    ap.add_argument("--append", action="store_true",
                    help="保留 records 里已有的轨迹（id 不变，已算好的注意力缓存仍然有效），"
                         "只补新的 prompt 组直到 --quota；已有轨迹的 prompt 自动排除")
    ap.add_argument("--exclude-prompts-from", default=None,
                    help="逗号分隔的 JSONL 清单，其中的 prompt 不参与本次生成。"
                         "用它指向已训练检测器的训练 prompt 清单，测试集才对该检测器真正独立")
    ap.add_argument("--splits", default="train,test",
                    help="generate/attention 阶段处理哪些 split；train 侧记录已定时可单独跑 train，"
                         "以免与正在生成的 test 记录相互干扰")
    ap.add_argument("--train-points", type=Path, default=None,
                    help="用这份取点 JSONL 训练，跳过从 --out/train/step_metrics 取点；"
                         "由 build_classifier_points.py 产出（同 schema）")
    ap.add_argument("--categories", default=None,
                    help="comma-separated categories for --stage generate; useful for isolated parallel candidate sweeps")
    ap.add_argument("--exhaustive", action="store_true",
                    help="每个温度都完整扫一遍全部 prompt，不因达到 --quota 提前结束。"
                         "用于「先把所有样例的轨迹都生成出来，之后再划分 train/test」")
    ap.add_argument("--subsets", default=None,
                    help="逗号分隔的数据集名（GSM8k,MMLU,SimpleQA,GPQA,MMLU_Econometrics,"
                         "MMLU_World_History,Recur,LoopLLM,concat）。不传则跑该类别配置的全部数据集；"
                         "generate 与 attention 两段都按它过滤")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    if a.categories:
        invalid = set(a.categories.split(",")) - set(BENIGN) - set(ATTACK)
        if invalid:
            ap.error(f"unknown --categories values: {', '.join(sorted(invalid))}")
    known_subsets = {sub for cat in (BENIGN | ATTACK) for sub in subsets_of(cat)}
    if a.subsets:
        invalid = set(a.subsets.split(",")) - known_subsets
        if invalid:
            ap.error(f"unknown --subsets values: {', '.join(sorted(invalid))}")
    if not planned_tasks(a.categories, a.subsets):
        ap.error("--categories 与 --subsets 的组合没有匹配到任何数据集")
    # vLLM's environment is the complete runtime on this host (vLLM,
    # Transformers, CUDA, sklearn). Re-exec the full command there when a
    # caller starts from another interpreter.
    if Path(sys.executable).resolve() != Path(a.vllm_python).resolve():
        subprocess.run([a.vllm_python, __file__, *sys.argv[1:]], check=True)
        return
    splits = [s for s in a.splits.split(",") if s]
    invalid_splits = set(splits) - {"train", "test"}
    if invalid_splits:
        ap.error(f"unknown --splits values: {', '.join(sorted(invalid_splits))}")
    if a.stage in ("all", "generate"):
        for split in splits:
            generate(a, split)
        if a.stage == "generate": return
    if a.stage in ("all", "attention"):
        for split in splits:
            run_attention_and_metrics(a, split)
        if a.stage == "attention": return
    if a.stage in ("all", "train"):
        train(a)
        if a.stage == "train": return
    if a.stage in ("all", "detect"):
        score_test(a)


if __name__ == "__main__":
    main()
