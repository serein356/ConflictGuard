#!/usr/bin/env python3
"""Convert the model-neutral ConflictGUI release into evaluator-ready JSON."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = ROOT / "prompts"
MODELS = ("qwen3vl", "uitars", "uivenus")
VARIANTS = ("vanilla", "feasibility")
TRAILING_NEWLINES = {
    ("qwen3vl", "vanilla"): 0,
    ("qwen3vl", "feasibility"): 0,
    ("uitars", "vanilla"): 1,
    ("uitars", "feasibility"): 1,
    ("uivenus", "vanilla"): 2,
    ("uivenus", "feasibility"): 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the model-ready vanilla and feasibility-prompting JSON files "
            "used by the ConflictGuard evaluator."
        )
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="ConflictGUI data/conflictgui.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-dir", type=Path, default=PROMPT_DIR)
    parser.add_argument("--default-split", default="test",
                        help="Split assigned when input rows have no split field")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS,
                        default=list(VARIANTS))
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("data", "rows", "records"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list of ConflictGUI rows: {path}")
    return payload


def load_template(prompt_dir: Path, model: str, variant: str) -> str:
    path = prompt_dir / f"{model}_{variant}.txt"
    template = path.read_text(encoding="utf-8")
    template = template.rstrip("\n") + "\n" * TRAILING_NEWLINES[(model, variant)]
    if template.count("{instruction}") != 1:
        raise ValueError(f"Expected one {{instruction}} placeholder in {path}")
    return template


def validate_row(row: dict[str, Any], index: int) -> None:
    if not isinstance(row.get("instruction"), str) or not row["instruction"].strip():
        raise ValueError(f"Row {index} has no instruction")
    images = row.get("images")
    if not images and row.get("screenshot"):
        images = [row["screenshot"]]
    if not isinstance(images, list) or not images or not all(
        isinstance(path, str) and path for path in images
    ):
        raise ValueError(f"Row {index} has no usable images field")
    for key in ("action", "task_id", "dataset", "data_type", "imgwidth", "imgheight"):
        if key not in row:
            raise ValueError(f"Row {index} is missing required field {key!r}")


def adapt_row(row: dict[str, Any], template: str) -> dict[str, Any]:
    result = copy.deepcopy(row)
    if not result.get("images"):
        result["images"] = [result["screenshot"]]
    text = template.replace("{instruction}", result["instruction"], 1)
    result["messages"] = [{"role": "user", "content": text}]
    return result


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    if not rows:
        raise ValueError("Input dataset is empty")
    for index, row in enumerate(rows):
        validate_row(row, index)

    split_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        split = str(row.get("split") or args.default_split)
        split_rows.setdefault(split, []).append(row)

    for model in args.models:
        for variant in args.variants:
            template = load_template(args.prompt_dir, model, variant)
            for split, source_rows in split_rows.items():
                output = args.output_dir / model / f"{split}_{variant}.json"
                write_json(output, [adapt_row(row, template) for row in source_rows])
                counts = Counter(str(row["data_type"]) for row in source_rows)
                print(
                    f"{output}: total={len(source_rows)} "
                    + " ".join(f"{key}={counts[key]}" for key in sorted(counts))
                )


if __name__ == "__main__":
    main()
