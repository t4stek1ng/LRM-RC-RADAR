"""Extract per-step attention rows for cross-category attention visualization.

For each sample (a record with `question` + `thinking` from a Llama-8B
generation), this script:
  1. Concatenates the chat-template prompt + thinking into a single sequence.
  2. Tokenizes with the model's tokenizer.
  3. Identifies "step-end" token positions — every token whose decoded text
     contains "\\n\\n" (Llama-8B BPE produces a single token id 271 for plain
     "\\n\\n" or a merged token like ".\\n\\n" id 382 for punctuation+blank
     line). Per docs/drafts/revise.md: "如果是一个 token，则以它为最后一个
     token". Since "\\n\\n" is a single token in Llama-8B, we use it directly.
  4. Runs a forward pass and, for each target layer, computes the attention
     row at each step-end position, averaged over heads (revise.md: 不按
     head 分开).
  5. Saves per-layer step-row matrices (shape: n_steps × seq_len, fp16) plus
     a meta.json with step positions, layer ids, prompt_len, etc.

Reuses _compute_attention_for_layer from extract_dynamics_features so the
RoPE / GQA handling stays consistent with the existing pipeline.

Output layout:
    <cache-dir>/<category>_<id>/
        meta.json
        attn_step_L<layer>.npy       # (n_steps, seq_len) fp16
        step_token_text.json         # decoded text of each step (for inspection)

Usage:
    PYTHONPATH=. python -m src.analysis.extract_step_attention \
        --model $RC_MODELS_ROOT/DeepSeek-R1-Distill-Llama-8B \
        --records dataset/reflection_categories/concise_reasoning_llama8b.jsonl \
                  dataset/reflection_categories/productive_reflection_llama8b.jsonl \
        --ids 0 0 \
        --target-layers 3 7 11 15 19 23 27 31 \
        --output-dir exp/step_attention_cache
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Reuse the layer-level attention computation from the existing pipeline so
# RoPE / GQA / fp16->fp32 handling stays in one place.
from .extract_dynamics_features import _compute_attention_for_layer  # noqa: E402


def load_record(path: Path, sample_id: int) -> dict[str, Any]:
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("id") == sample_id:
                return r
    raise SystemExit(f"id={sample_id} not found in {path}")


def find_step_end_positions(
    input_ids: list[int], tokenizer: Any, prompt_len: int
) -> tuple[list[int], list[str]]:
    """Find token positions whose decoded text contains "\\n\\n".

    Only considers positions in the thinking portion (>= prompt_len).
    Returns (positions, decoded_text_per_position) so the meta json can show
    what each "step" actually ends with.
    """
    positions: list[int] = []
    texts: list[str] = []
    for i in range(prompt_len, len(input_ids)):
        # Decode this single token; with skip_special_tokens=False so we see
        # the raw "\n\n" or merged punctuation form like ".\n\n".
        piece = tokenizer.decode([input_ids[i]], skip_special_tokens=False)
        if "\n\n" in piece:
            positions.append(i)
            texts.append(piece)
    return positions, texts


def find_prompt_step_positions(
    input_ids: list[int], tokenizer: Any, prompt_len: int
) -> tuple[list[int], list[str]]:
    """Same logic as ``find_step_end_positions`` but restricted to the
    prompt portion (positions ``< prompt_len``).

    Used by visualization code to anchor the default view-crop at the
    second-to-last ``\\n\\n`` inside the prompt — keeps the matrix focused
    on the prompt's tail boundary instead of the entire input.
    """
    positions: list[int] = []
    texts: list[str] = []
    end = min(prompt_len, len(input_ids))
    for i in range(end):
        piece = tokenizer.decode([input_ids[i]], skip_special_tokens=False)
        if "\n\n" in piece:
            positions.append(i)
            texts.append(piece)
    return positions, texts


def get_decoder_layers(model: Any) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise SystemExit("Could not locate decoder layers on model")


@torch.no_grad()
def extract_one_sample(
    model: Any,
    tokenizer: Any,
    record: dict[str, Any],
    target_layers: list[int],
    output_dir: Path,
    max_seq_len: int,
) -> dict[str, Any]:
    category = record["category"]
    sample_id = record["id"]
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": record["question"]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    thinking = record["thinking"] or ""

    full_text = prompt_text + thinking
    enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids.to(model.device)
    seq_len = input_ids.shape[1]
    if seq_len > max_seq_len:
        input_ids = input_ids[:, :max_seq_len]
        seq_len = max_seq_len

    prompt_ids = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False
    ).input_ids
    prompt_len = prompt_ids.shape[1]

    flat_ids = input_ids.squeeze(0).tolist()
    step_positions, step_texts = find_step_end_positions(
        flat_ids, tokenizer, prompt_len
    )
    if not step_positions:
        raise SystemExit(
            f"No step-end tokens (containing \\n\\n) found for "
            f"{category} id={sample_id}; cannot proceed."
        )

    n_layers = model.config.num_hidden_layers
    norm_layers = [l % n_layers for l in target_layers]

    # Forward pass: hidden states only (output_attentions=False keeps memory
    # manageable; we'll re-derive attention layer-by-layer below).
    outputs = model(
        input_ids,
        output_attentions=False,
        output_hidden_states=True,
    )
    hidden_inputs: dict[int, torch.Tensor] = {}
    for layer_idx in norm_layers:
        hidden_inputs[layer_idx] = outputs.hidden_states[layer_idx].detach()
    del outputs
    torch.cuda.empty_cache()

    decoder_layers = get_decoder_layers(model)
    position_ids = torch.arange(seq_len, device=model.device).unsqueeze(0)
    rotary_emb = getattr(model.model, "rotary_emb", None)

    sample_dir = output_dir / f"{category}_{sample_id}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    step_pos_tensor = torch.tensor(step_positions, device="cpu")

    for layer_idx in norm_layers:
        # (heads, seq, seq); on CPU
        attn = _compute_attention_for_layer(
            decoder_layers[layer_idx],
            hidden_inputs[layer_idx],
            position_ids,
            model_config=model.config,
            rotary_emb=rotary_emb,
        )
        # Index step rows: (heads, n_steps, seq) -> mean over heads -> (n_steps, seq)
        rows = attn.index_select(dim=1, index=step_pos_tensor)
        rows_mean = rows.mean(dim=0).numpy().astype(np.float16)
        np.save(sample_dir / f"attn_step_L{layer_idx}.npy", rows_mean)
        del attn, rows, rows_mean
        torch.cuda.empty_cache()

    del hidden_inputs
    torch.cuda.empty_cache()

    meta = {
        "id": sample_id,
        "category": category,
        "model_name": record.get("model_name"),
        "prompt_len": int(prompt_len),
        "thinking_len": int(seq_len - prompt_len),
        "seq_len": int(seq_len),
        "step_positions": [int(p) for p in step_positions],
        "n_steps": len(step_positions),
        "norm_layers": [int(l) for l in norm_layers],
        "n_heads": int(model.config.num_attention_heads),
    }
    with open(sample_dir / "meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(sample_dir / "step_token_text.json", "w") as f:
        json.dump(step_texts, f, ensure_ascii=False, indent=2)

    print(f"[step-attn] {category} id={sample_id}: "
          f"prompt_len={prompt_len} seq_len={seq_len} "
          f"n_steps={len(step_positions)} layers={norm_layers}")
    return meta


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--records", nargs="+", required=True,
                   help="JSONL files; each provides one sample (paired with --ids)")
    p.add_argument("--ids", nargs="+", type=int, required=True,
                   help="Sample id within each --records file")
    p.add_argument("--target-layers", nargs="+", type=int,
                   default=[3, 7, 11, 15, 19, 23, 27, 31],
                   help="Layer indices (32-layer Llama-8B every-4th)")
    p.add_argument("--output-dir", type=Path,
                   default=Path("recur_code/exp/step_attention_cache"))
    p.add_argument("--gpu", default="0")
    p.add_argument("--max-seq-len", type=int, default=8192)
    args = p.parse_args()

    if len(args.records) != len(args.ids):
        raise SystemExit("--records and --ids must have equal length")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[step-attn] loading {args.model} on cuda (visible={args.gpu})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="cuda:0",
    )
    model.eval()
    print(f"[step-attn] model loaded; "
          f"layers={model.config.num_hidden_layers} "
          f"heads={model.config.num_attention_heads}")

    for rec_path, sample_id in zip(args.records, args.ids):
        record = load_record(Path(rec_path), sample_id)
        extract_one_sample(
            model, tokenizer, record, args.target_layers,
            args.output_dir, args.max_seq_len,
        )

    print(f"[step-attn] done. cache dir = {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
