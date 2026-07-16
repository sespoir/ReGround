#!/usr/bin/env python3
"""Convert generic visual QA records into veRL GRPO parquet files."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import random
from typing import Any, Iterator

INSTRUCTION = """Solve the visual question with careful reasoning.
Use <think>...</think> for reasoning and <answer>...</answer> for the final answer.
If fresh visual evidence is genuinely needed, end the first turn with a concise,
first-person <reground>...</reground> diagnosis instead of an answer."""


def get_nested(record: dict[str, Any], field: str, default: Any = None) -> Any:
    value: Any = record
    for key in field.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def iter_json(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".jsonl":
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} is not an object")
                yield value
            return

        value = json.load(handle)
        if isinstance(value, dict):
            value = value.get("data", value.get("records", value))
        if isinstance(value, dict):
            yield value
        elif isinstance(value, list):
            for item in value:
                if not isinstance(item, dict):
                    raise ValueError(f"{path} contains a non-object record")
                yield item
        else:
            raise ValueError(f"Unsupported JSON root in {path}")


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        yield from iter_json(path)
        return
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError("pyarrow is required for parquet input") from error
        yield from pq.read_table(path).to_pylist()
        return
    raise ValueError(f"Unsupported input format: {path}")


def resolve_inputs(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        expanded = Path(value).expanduser()
        if expanded.is_dir():
            paths.extend(
                path
                for path in sorted(expanded.iterdir())
                if path.suffix.lower() in {".json", ".jsonl", ".parquet"}
            )
        else:
            matches = [Path(match) for match in sorted(glob.glob(str(expanded)))]
            paths.extend(matches or [expanded])
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Input files not found: {', '.join(missing)}")
    if not paths:
        raise ValueError("No input files found")
    return paths


def image_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.lstrip().startswith("["):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else [parsed]
    return [value]


def resolve_image(value: Any, record_dir: Path, image_root: Path | None) -> dict[str, str]:
    if isinstance(value, dict):
        value = value.get("image", value.get("path", value.get("file_name")))
    if not isinstance(value, str) or not value.strip():
        raise ValueError("image must be a local path or a path-bearing object")
    path = Path(value).expanduser()
    if not path.is_absolute():
        candidates = [record_dir / path]
        if image_root is not None:
            candidates.insert(0, image_root / path)
        path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    return {"image": str(path)}


def normalize_choices(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        if value.lstrip().startswith("["):
            value = json.loads(value)
        else:
            return [value]
    if isinstance(value, dict):
        return [str(value[key]) for key in sorted(value)]
    return [str(item) for item in value]


def canonical_answer(value: Any, choices: list[str], answer_index_base: str) -> tuple[str, list[str]]:
    accepted = value if isinstance(value, list) else [value]
    accepted = [str(item).strip() for item in accepted if item is not None]
    if not accepted:
        raise ValueError("answer is missing")
    answer = accepted[0]

    if choices:
        try:
            index = int(answer) - (1 if answer_index_base == "one" else 0)
        except ValueError:
            index = -1
        if 0 <= index < len(choices) and index < 26:
            label = chr(ord("A") + index)
            accepted.extend([label, choices[index]])
            answer = label
        elif len(answer) == 1 and answer.upper().isalpha():
            answer = answer.upper()
            index = ord(answer) - ord("A")
            if 0 <= index < len(choices):
                accepted.append(choices[index])

    return answer, list(dict.fromkeys(accepted))


def choices_block(choices: list[str]) -> str:
    if not choices:
        return ""
    lines = [f"{chr(ord('A') + index)}. {choice}" for index, choice in enumerate(choices)]
    return "\nOptions:\n" + "\n".join(lines)


def build_record(
    raw: dict[str, Any],
    record_dir: Path,
    index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    question = str(get_nested(raw, args.question_field, "")).strip()
    if not question:
        raise ValueError("question is missing")
    choices = normalize_choices(get_nested(raw, args.choices_field))
    answer, accepted = canonical_answer(
        get_nested(raw, args.answer_field), choices, args.answer_index_base
    )
    images = [
        resolve_image(value, record_dir, args.image_root)
        for value in image_values(get_nested(raw, args.image_field))
    ]
    if not images:
        raise ValueError("image is missing")

    source = str(get_nested(raw, args.source_field, "reground")).strip() or "reground"
    sample_id = str(get_nested(raw, args.id_field, index))
    prompt = "<image>\n" * len(images) + question + choices_block(choices)
    prompt += "\n\n" + INSTRUCTION

    return {
        "data_source": source,
        "prompt": [{"role": "user", "content": prompt}],
        "images": images,
        "ability": str(get_nested(raw, args.ability_field, "vision_reasoning")),
        "reward_model": {"style": "rule", "ground_truth": answer},
        "agent_name": "reground_agent",
        "extra_info": {
            "index": index,
            "sample_id": sample_id,
            "question": question,
            "accepted_answers": accepted,
            "choices": choices,
        },
    }


def load_split(paths: list[Path], args: argparse.Namespace, start_index: int = 0) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    index = start_index
    for path in paths:
        for row_number, raw in enumerate(iter_records(path)):
            try:
                records.append(build_record(raw, path.parent, index, args))
                index += 1
            except Exception as error:
                failures.append(f"{path}:{row_number}: {type(error).__name__}: {error}")
    if failures and args.strict:
        raise ValueError("\n".join(failures[:20]))
    if failures:
        print(f"Skipped {len(failures)} invalid records; first error: {failures[0]}")
    return records


def write_parquet(records: list[dict[str, Any]], path: Path) -> None:
    if not records:
        raise ValueError(f"Cannot write empty split: {path}")
    try:
        from datasets import Dataset
    except ImportError as error:
        raise RuntimeError("Install datasets and pyarrow before preparing GRPO data") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).to_parquet(str(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--val-input", action="append")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/reground/data/grpo"))
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--choices-field", default="choices")
    parser.add_argument("--image-field", default="image")
    parser.add_argument("--source-field", default="source")
    parser.add_argument("--ability-field", default="ability")
    parser.add_argument("--answer-index-base", choices=("zero", "one"), default="zero")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")
    args.image_root = args.image_root.expanduser().resolve() if args.image_root else None

    train = load_split(resolve_inputs(args.input), args)
    if args.val_input:
        val = load_split(resolve_inputs(args.val_input), args, len(train))
    else:
        rng = random.Random(args.seed)
        rng.shuffle(train)
        val_size = max(1, round(len(train) * args.val_ratio))
        val, train = train[:val_size], train[val_size:]

    write_parquet(train, args.output_dir / "train.parquet")
    write_parquet(val, args.output_dir / "val.parquet")
    print(json.dumps({"train": len(train), "val": len(val)}, indent=2))


if __name__ == "__main__":
    main()
