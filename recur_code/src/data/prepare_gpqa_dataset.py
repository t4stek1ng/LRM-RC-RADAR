import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert GPQA CSV into processed JSONL for reasoning-trace collection."
    )
    parser.add_argument(
        "--input-csv",
        default="recur_code/dataset/gpqa/gpqa_main.csv",
        help="Path to the raw GPQA CSV file.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="recur_code/dataset/ours/gpqa_main.jsonl",
        help="Path to the processed JSONL output.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used to shuffle answer choices deterministically.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of samples to export.",
    )
    return parser


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def clean_text(text: str | None) -> str:
    if text is None:
        return ""
    return " ".join(text.replace("\r", "\n").split())


def build_question(question: str, options: dict[str, str]) -> str:
    option_lines = [f"{label}. {options[label]}" for label in ["A", "B", "C", "D"]]
    return (
        f"{question.strip()}\n\n"
        "Choose the correct answer from the following options.\n"
        + "\n".join(option_lines)
        + "\n\nRespond with the option letter and the answer text."
    )


def convert_row(row: dict[str, str], idx: int, rng: random.Random) -> dict[str, Any]:
    question = clean_text(row["Question"])
    correct_answer = clean_text(row["Correct Answer"])
    incorrect_answers = [
        clean_text(row["Incorrect Answer 1"]),
        clean_text(row["Incorrect Answer 2"]),
        clean_text(row["Incorrect Answer 3"]),
    ]

    choices = [correct_answer, *incorrect_answers]
    rng.shuffle(choices)
    option_labels = ["A", "B", "C", "D"]
    options = {label: choice for label, choice in zip(option_labels, choices)}
    correct_option = next(label for label, choice in options.items() if choice == correct_answer)

    return {
        "id": idx,
        "record_id": clean_text(row.get("Record ID")),
        "question": build_question(question, options),
        "question_stem": question,
        "options": options,
        "correct_option": correct_option,
        "answer": correct_answer,
        "incorrect answers": incorrect_answers,
        "explanation": clean_text(row.get("Explanation")),
        "high_level_domain": clean_text(row.get("High-level domain")),
        "subdomain": clean_text(row.get("Subdomain")),
        "writer_difficulty_estimate": clean_text(row.get("Writer's Difficulty Estimate")),
        "source_dataset": "gpqa_main",
    }


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output_jsonl)
    ensure_parent(output_path)

    rng = random.Random(args.seed)
    rows: list[dict[str, str]] = []
    with input_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if args.limit is not None:
        rows = rows[: args.limit]

    with output_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            record = convert_row(row, idx, rng)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} records to {output_path}")


if __name__ == "__main__":
    main()
