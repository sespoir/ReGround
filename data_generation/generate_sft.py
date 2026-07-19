#!/usr/bin/env python3
"""Generate dataset-agnostic ReGround supervised fine-tuning trajectories.

The generator builds trajectories with the following routing:

1. A policy model produces an initial visual observation, reasoning, and answer.
2. Incorrect answers receive a corrective cue.
3. Correct but weakly grounded answers receive a grounding cue.
4. A deterministic epsilon sample of grounded answers receives a verification cue.
5. Remaining grounded answers become No-ReGround examples.

Input files may be JSON, JSONL, or Parquet.  The default record contract is:

    {
      "id": "sample-1",
      "question": "...",
      "answer": "...",
      "choices": ["...", "..."],       # optional
      "image": "/path/to/image.jpg",   # one image or a list
      "knowledge": "...",              # optional, teacher only
      "source": "dataset-name"          # optional
    }

All field names are configurable.  The output ``sft.jsonl`` uses the messages
and images schema consumed by LlamaFactory and the ReGround training pipeline.
No API credential is accepted on the command line; optional keys are read only
from environment variables.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from collections import Counter
from dataclasses import dataclass, field
import glob
import hashlib
from io import BytesIO
import json
import logging
import numbers
import os
from pathlib import Path
import random
import re
import string
from typing import Any, Iterator, Sequence

import httpx
from PIL import Image, ImageOps
from tqdm import tqdm


LOGGER = logging.getLogger("reground_sft")
LETTERS = string.ascii_uppercase


ROUND1_PROMPT = """You are solving a visual question.

Question:
{question}
{options_block}

Your task has three phases.

Phase 1 — Visual Scan <observation>
Systematically inspect the entire image. Record the concrete visual facts that
may matter: objects, labels, text, values, attributes, positions, and spatial
relationships. Be specific and do not replace observation with assumptions.

Phase 2 — Step-by-Step Reasoning <reasoning>
Build a logical chain grounded in the visual evidence. Relevant domain
knowledge may be used, but clearly connect it to what is visible.

Phase 3 — Final Answer <answer>
{answer_instruction}

Output exactly:
<observation>visual evidence</observation>
<reasoning>step-by-step reasoning</reasoning>
{answer_format}
"""


CORRECTIVE_CUE_PROMPT = """You are the student yourself.
Your answer is WRONG. Identify WHAT visual element or reasoning link to
re-examine — NOT what the correct answer should be.

{teacher_knowledge}
[MY QUESTION]
{question}
{options_block}

[MY OBSERVATION]
{observation}

[MY REASONING]
{reasoning}

=== STRICT RULES ===
FORBIDDEN:
- The correct answer, answer letter, or correct option text
- A numeric correction or direct conclusion
- Saying which option is right or wrong

REQUIRED:
- Say only WHAT to re-check: a position, relationship, label, attribute, or
  reasoning step
- Use one sentence of at most 25 words
- Focus on the visual or logical source of the error, not its result

Output exactly:
<suggest_reground>yes</suggest_reground>
<doubt_reason>one targeted sentence</doubt_reason>
"""


GROUNDING_JUDGE_PROMPT = """You are the student yourself.
The final answer has already been checked for correctness. Perform a pragmatic
visual-grounding safety check before submitting. When in doubt, choose ReGround.

{teacher_knowledge}
[MY QUESTION]
{question}
{options_block}

[MY OBSERVATION]
{observation}

[MY REASONING]
{reasoning}

You may output NO ReGround only if ALL conditions are true:
(A) VISUAL ENTAILMENT: the cited image evidence directly supports the answer.
(B) CONCRETE ANCHORS: the reasoning names specific visible objects, labels,
    values, positions, or relationships.
(C) NO HALLUCINATION: no decisive visual fact is invented or merely assumed.
(D) CONSISTENCY: the reasoning is logically consistent with the cited evidence.

If any condition fails, choose ReGround and identify one targeted item to check.
Never reveal the answer or correct option content in the diagnostic cue.

If NO ReGround, output exactly:
<suggest_reground>no</suggest_reground>
<confidence_reason>one sentence citing concrete visual anchors</confidence_reason>

If ReGround, output exactly:
<suggest_reground>yes</suggest_reground>
<doubt_reason>one sentence of at most 25 words</doubt_reason>
"""


VERIFICATION_CUE_PROMPT = """You are the student yourself.
Do one quick sanity check before finalizing. Pick the SINGLE weakest link in
your reasoning.

{teacher_knowledge}
[MY QUESTION]
{question}
{options_block}

[MY OBSERVATION]
{observation}

[MY REASONING]
{reasoning}

Identify the one visual detail or reasoning relationship most likely to cause
an error if misread. Focus on WHAT to verify, not what the answer or value
should be. Do not reveal an answer letter or option content.

Output exactly:
<suggest_reground>yes</suggest_reground>
<doubt_reason>one sentence of at most 25 words</doubt_reason>
"""


REGROUND_CONTENT_PROMPT = """You are the student yourself. Write a brief
internal monologue to guide your visual re-examination.

[MY DOUBT]
{doubt_reason}

=== STRICT RULES ===
MUST:
- Use first person ("I" or "my")
- Start with "Wait...", "Hold on...", or "Let me double-check..."
- Say WHAT to inspect: a position, mark, label, object, attribute, or relation
- Use two or three short sentences

MUST NOT:
- Reveal the answer, an answer letter, correct option text, or numeric correction
- State a direct conclusion
- Speak about "the student" in third person

Output exactly:
<reground>targeted first-person re-examination plan</reground>
"""


ROUND2_PROMPT = """You are solving a visual question. You produced an initial
solution but had a moment of self-doubt.

Question:
{question}
{options_block}

Your previous observation:
{previous_observation}

Your self-reflection:
<reground>{reground_instruction}</reground>

Act on that doubt.

Phase 1 — Visual Reset <observation>
Re-examine the image from scratch, concentrating on the diagnostic target.
State what is corrected, confirmed, or newly clarified using concrete evidence.

Phase 2 — Step-by-Step Reasoning <reasoning>
Rebuild the reasoning from the updated visual observation.

Phase 3 — Final Answer <answer>
{answer_instruction}

Output exactly:
<observation>updated visual evidence</observation>
<reasoning>revised reasoning</reasoning>
{answer_format}
"""


ROUND2_JUDGE_PROMPT = """Audit a visual re-examination trajectory.

Question:
{question}
{options_block}

Initial observation:
{initial_observation}

Diagnostic cue:
{reground_instruction}

Updated observation:
{updated_observation}

Updated reasoning:
{updated_reasoning}

Evaluate whether the second pass follows the diagnostic cue, contains a
material visual update or explicit evidence-based confirmation, and grounds its
reasoning in the image. Do not judge answer correctness; it is checked outside.

Output JSON only:
{{
  "is_grounded": true,
  "has_material_update": true,
  "follows_diagnostic_cue": true,
  "pass": true,
  "issues": []
}}
"""


@dataclass
class Sample:
    sample_id: str
    source: str
    question: str
    canonical_answer: str
    accepted_answers: list[str]
    choices: list[str]
    image_refs: list[str]
    image_b64: list[str]
    knowledge: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Generation:
    observation: str = ""
    reasoning: str = ""
    answer: str = ""
    raw: str = ""

    @property
    def valid(self) -> bool:
        return bool(self.observation and self.reasoning and self.answer)


@dataclass
class TriggerDecision:
    suggest_reground: bool
    reason: str
    confidence_reason: str
    raw: str
    parsed: bool


@dataclass
class Outcome:
    sample_id: str
    status: str
    route: str = ""
    sft: dict[str, Any] | None = None
    meta: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None


@dataclass
class LoadingStats:
    records: int = 0
    loaded: int = 0
    invalid: int = 0
    duplicate_ids: int = 0
    missing_images: int = 0
    errors: Counter = field(default_factory=Counter)


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def get_nested(record: dict[str, Any], field_name: str, default: Any = None) -> Any:
    """Read a dotted field path such as ``payload.question``."""
    if not field_name:
        return default
    value: Any = record
    for part in field_name.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def resolve_input_files(inputs: Sequence[str], pattern: str) -> list[Path]:
    supported = {".json", ".jsonl", ".parquet", ".pq"}
    resolved: list[Path] = []
    for entry in inputs:
        matches = [Path(path) for path in glob.glob(entry)]
        if not matches:
            matches = [Path(entry)]
        for path in matches:
            if path.is_dir():
                resolved.extend(
                    sorted(
                        p
                        for p in path.glob(pattern)
                        if p.is_file() and p.suffix.lower() in supported
                    )
                )
            elif path.is_file():
                resolved.append(path)
            else:
                raise FileNotFoundError(f"Input does not exist: {path}")
    unique = list(dict.fromkeys(path.resolve() for path in resolved))
    if not unique:
        raise FileNotFoundError("No input files were found")
    return unique


def iter_records(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                yield line_number, value
        return

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if isinstance(value, dict) and isinstance(value.get("data"), list):
            value = value["data"]
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, list):
            raise ValueError(f"{path} must contain an object, a list, or a data list")
        for index, record in enumerate(value):
            if not isinstance(record, dict):
                raise ValueError(f"{path} record {index} is not an object")
            yield index, record
        return

    if suffix in {".parquet", ".pq"}:
        try:
            import pandas as pd
        except ImportError as error:
            raise RuntimeError("Parquet input requires pandas and pyarrow") from error
        frame = pd.read_parquet(path)
        for position, (_, row) in enumerate(frame.iterrows()):
            yield position, row.to_dict()
        return

    raise ValueError(f"Unsupported input format: {path}")


def normalize_choices(raw: Any) -> list[str]:
    if raw is None:
        return []
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.startswith("["):
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError:
                raw = [raw]
        else:
            raw = [raw]
    if not isinstance(raw, (list, tuple)):
        raise ValueError("choices must be a list")
    choices = [str(choice).strip() for choice in raw]
    if choices and (len(choices) < 2 or len(choices) > len(LETTERS)):
        raise ValueError(f"choices must contain 2-{len(LETTERS)} entries")
    if any(not choice for choice in choices):
        raise ValueError("choices contain an empty entry")
    return choices


def scalar_value(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value


def optional_text(value: Any) -> str:
    value = scalar_value(value)
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value).strip()


def prepare_answer(
    raw_answer: Any,
    choices: list[str],
    index_base: str,
) -> tuple[str, list[str]]:
    if raw_answer is None:
        raise ValueError("answer is missing")

    raw_answer = scalar_value(raw_answer)
    values = raw_answer if isinstance(raw_answer, (list, tuple)) else [raw_answer]
    values = [scalar_value(value) for value in values]
    if not values:
        raise ValueError("answer is empty")

    if choices:
        resolved_letters: list[str] = []
        for value in values:
            if isinstance(value, numbers.Integral):
                index = int(value) - (1 if index_base == "one" else 0)
            else:
                text = str(value).strip()
                if re.fullmatch(r"[A-Za-z]", text):
                    index = (
                        LETTERS.index(text.upper()) if text.upper() in LETTERS else -1
                    )
                elif re.fullmatch(r"\d+", text):
                    index = int(text) - (1 if index_base == "one" else 0)
                else:
                    normalized = normalize_text_answer(text)
                    index = next(
                        (
                            idx
                            for idx, choice in enumerate(choices)
                            if normalize_text_answer(choice) == normalized
                        ),
                        -1,
                    )
            if not 0 <= index < len(choices):
                raise ValueError(f"answer {value!r} does not identify a valid choice")
            resolved_letters.append(LETTERS[index])
        canonical = resolved_letters[0]
        canonical_index = LETTERS.index(canonical)
        accepted = list(dict.fromkeys(resolved_letters + [choices[canonical_index]]))
        return canonical, accepted

    accepted = [str(value).strip() for value in values if str(value).strip()]
    if not accepted:
        raise ValueError("answer is empty")
    return accepted[0], list(dict.fromkeys(accepted))


def normalize_text_answer(value: str) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\\boxed\s*\{(.*?)\}", r"\1", text)
    text = re.sub(r"^(?:answer|option)\s*[:：]?\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n.,;:!?()[]{}\"'")


def answers_match(prediction: str, sample: Sample, numeric_tolerance: float) -> bool:
    predicted = normalize_text_answer(prediction)
    if sample.choices:
        match = re.search(r"\b([A-Z])\b", prediction.upper())
        if match and match.group(1) == sample.canonical_answer:
            return True
        correct_index = LETTERS.index(sample.canonical_answer)
        return predicted == normalize_text_answer(sample.choices[correct_index])

    for answer in sample.accepted_answers:
        normalized = normalize_text_answer(answer)
        if predicted == normalized:
            return True
        try:
            if (
                abs(
                    float(predicted.replace(",", ""))
                    - float(normalized.replace(",", ""))
                )
                <= numeric_tolerance
            ):
                return True
        except ValueError:
            pass
    return False


def coerce_image_values(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if hasattr(raw, "tolist") and not isinstance(raw, (bytes, bytearray, memoryview)):
        raw = raw.tolist()
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return [raw]


def image_from_value(
    value: Any,
    record_dir: Path,
    image_root: Path | None,
) -> tuple[Image.Image, Path | None]:
    original_path: Path | None = None
    payload: bytes | None = None

    if isinstance(value, dict):
        if value.get("bytes") is not None:
            raw_bytes = value["bytes"]
            payload = (
                bytes(raw_bytes) if not isinstance(raw_bytes, bytes) else raw_bytes
            )
        elif value.get("path") is not None:
            value = value["path"]
        elif value.get("base64") is not None:
            payload = base64.b64decode(str(value["base64"]))

    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
    elif isinstance(value, str) and payload is None:
        if value.startswith("data:image") and "," in value:
            payload = base64.b64decode(value.split(",", 1)[1])
        else:
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                roots = [image_root, record_dir] if image_root else [record_dir]
                candidate = next(
                    (root / candidate for root in roots if (root / candidate).exists()),
                    roots[0] / candidate,
                )
            original_path = candidate.resolve()
            if not original_path.is_file():
                raise FileNotFoundError(f"image not found: {original_path}")
            payload = original_path.read_bytes()

    if payload is None:
        raise ValueError(f"unsupported image value: {type(value).__name__}")

    with Image.open(BytesIO(payload)) as image:
        normalized = ImageOps.exif_transpose(image).convert("RGB")
        normalized.load()
    return normalized, original_path


def image_to_jpeg_b64(image: Image.Image, quality: int) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return (cleaned or "sample")[:64]


def materialize_images(
    raw_images: Any,
    sample_id: str,
    record_dir: Path,
    output_dir: Path,
    image_root: Path | None,
    image_mode: str,
    jpeg_quality: int,
) -> tuple[list[str], list[str]]:
    values = coerce_image_values(raw_images)
    if not values:
        raise ValueError("image is missing")

    refs: list[str] = []
    encodings: list[str] = []
    images_dir = output_dir / "images"
    if image_mode == "copy":
        images_dir.mkdir(parents=True, exist_ok=True)

    for image_index, value in enumerate(values):
        image, original_path = image_from_value(value, record_dir, image_root)
        encodings.append(image_to_jpeg_b64(image, jpeg_quality))

        if image_mode == "reference" and original_path is not None:
            refs.append(str(original_path))
            continue

        digest = hashlib.sha1(f"{sample_id}:{image_index}".encode("utf-8")).hexdigest()[
            :10
        ]
        filename = f"{safe_id(sample_id)}_{digest}_{image_index}.jpg"
        destination = images_dir / filename
        if not destination.exists():
            image.save(destination, format="JPEG", quality=jpeg_quality, optimize=True)
        refs.append(str(destination.resolve()))

    return refs, encodings


def load_samples(args: argparse.Namespace) -> tuple[list[Sample], LoadingStats]:
    output_dir = Path(args.output_dir).resolve()
    image_root = (
        Path(args.image_root).expanduser().resolve() if args.image_root else None
    )
    files = resolve_input_files(args.input, args.input_pattern)
    LOGGER.info("Input files: %d", len(files))

    candidates: list[tuple[Path, int, dict[str, Any]]] = []
    for path in files:
        for row_index, record in iter_records(path):
            candidates.append((path, row_index, record))

    rng = random.Random(args.seed)
    if args.shuffle:
        rng.shuffle(candidates)
    if args.num_samples > 0:
        candidates = candidates[: args.num_samples]

    stats = LoadingStats(records=len(candidates))
    samples: list[Sample] = []
    seen_ids: set[str] = set()
    metadata_fields = [
        field.strip() for field in args.metadata_fields.split(",") if field.strip()
    ]

    for path, row_index, record in tqdm(candidates, desc="Loading inputs"):
        try:
            question = str(get_nested(record, args.question_field, "")).strip()
            if not question:
                raise ValueError("question is missing")

            choices = normalize_choices(get_nested(record, args.choices_field))
            canonical, accepted = prepare_answer(
                get_nested(record, args.answer_field), choices, args.answer_index_base
            )

            source = optional_text(get_nested(record, args.source_field)) or path.stem
            raw_id = get_nested(record, args.id_field)
            sample_id = (
                str(raw_id) if raw_id is not None else f"{path.stem}_{row_index}"
            )
            if sample_id in seen_ids:
                stats.duplicate_ids += 1
                sample_id = f"{sample_id}@{path.stem}_{row_index}"
            if sample_id in seen_ids:
                raise ValueError("duplicate sample id")

            image_refs, image_b64 = materialize_images(
                get_nested(record, args.image_field),
                sample_id,
                path.parent,
                output_dir,
                image_root,
                args.image_mode,
                args.jpeg_quality,
            )

            knowledge = optional_text(get_nested(record, args.knowledge_field))
            metadata = {
                field: json_safe(get_nested(record, field)) for field in metadata_fields
            }
            metadata["input_file"] = str(path)
            metadata["input_row"] = row_index

            samples.append(
                Sample(
                    sample_id=sample_id,
                    source=source,
                    question=question,
                    canonical_answer=canonical,
                    accepted_answers=accepted,
                    choices=choices,
                    image_refs=image_refs,
                    image_b64=image_b64,
                    knowledge=knowledge,
                    metadata=metadata,
                )
            )
            seen_ids.add(sample_id)
            stats.loaded += 1
        except FileNotFoundError as error:
            stats.invalid += 1
            stats.missing_images += 1
            stats.errors[str(error)] += 1
            LOGGER.debug("Skipping %s:%s: %s", path, row_index, error)
        except Exception as error:
            stats.invalid += 1
            stats.errors[str(error)] += 1
            LOGGER.debug("Skipping %s:%s: %s", path, row_index, error)

    return samples, stats


def format_options(choices: Sequence[str]) -> str:
    if not choices:
        return ""
    return "Options:\n" + "\n".join(
        f"{LETTERS[index]}. {choice}" for index, choice in enumerate(choices)
    )


def prompt_fields(sample: Sample) -> dict[str, str]:
    if sample.choices:
        allowed = "/".join(LETTERS[: len(sample.choices)])
        answer_instruction = f"Output only the correct option letter ({allowed})."
        answer_format = f"<answer>{allowed}</answer>"
    else:
        answer_instruction = (
            "Output a concise answer without explanation in this section."
        )
        answer_format = "<answer>concise final answer</answer>"
    knowledge = (
        "[REFERENCE FOR AUDITING ONLY]\n" + sample.knowledge + "\n"
        if sample.knowledge
        else ""
    )
    return {
        "question": sample.question,
        "options_block": format_options(sample.choices),
        "answer_instruction": answer_instruction,
        "answer_format": answer_format,
        "teacher_knowledge": knowledge,
    }


def fill_prompt(template: str, sample: Sample, **extra: str) -> str:
    values = prompt_fields(sample)
    values.update(extra)
    return template.format(**values).strip()


def parse_tag(text: str, tag: str) -> str:
    if not text:
        return ""
    match = re.search(
        rf"<\s*{re.escape(tag)}\s*>\s*(.*?)\s*<\s*/\s*{re.escape(tag)}\s*>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def parse_generation(raw: str) -> Generation:
    return Generation(
        observation=parse_tag(raw, "observation"),
        reasoning=parse_tag(raw, "reasoning"),
        answer=parse_tag(raw, "answer"),
        raw=raw,
    )


def parse_trigger_decision(raw: str) -> TriggerDecision:
    suggestion = parse_tag(raw, "suggest_reground").lower()
    parsed = suggestion in {"yes", "no"}
    return TriggerDecision(
        suggest_reground=suggestion != "no",
        reason=parse_tag(raw, "doubt_reason"),
        confidence_reason=parse_tag(raw, "confidence_reason"),
        raw=raw,
        parsed=parsed,
    )


def extract_json(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None


def pathological_output(text: str, max_chars: int) -> str | None:
    if not text.strip():
        return "empty output"
    if len(text) > max_chars:
        return f"output exceeds {max_chars} characters"
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 20]
    if lines and max(Counter(lines).values()) >= 5:
        return "repeated lines"
    if re.search(r"(.)\1{30,}", text):
        return "character repetition"
    return None


def cue_leaks_answer(cue: str, sample: Sample) -> bool:
    text = cue.strip().lower()
    if not text:
        return True
    conclusion_patterns = [
        r"\b(?:correct|right|final|actual)\s+(?:answer|option)\b",
        r"\b(?:answer|option)\s+(?:is|should be)\b",
        r"\b(?:choose|select)\s+(?:option\s+)?[a-z]\b",
        r"\b(?:option\s+)?[a-z]\s+is\s+(?:correct|right)\b",
        r"\b(?:should equal|equals instead|not\s+\d+(?:\.\d+)?)\b",
    ]
    if any(re.search(pattern, text) for pattern in conclusion_patterns):
        return True

    if sample.choices:
        correct_index = LETTERS.index(sample.canonical_answer)
        correct_text = normalize_text_answer(sample.choices[correct_index])
        if len(correct_text) >= 5 and correct_text in normalize_text_answer(text):
            return True
    else:
        for answer in sample.accepted_answers:
            normalized = normalize_text_answer(answer)
            if len(normalized) >= 3 and re.search(
                rf"(?<!\w){re.escape(normalized)}(?!\w)", normalize_text_answer(text)
            ):
                return True
    return False


def valid_reground_content(cue: str, sample: Sample) -> bool:
    if cue_leaks_answer(cue, sample):
        return False
    words = cue.split()
    if not 4 <= len(words) <= 80:
        return False
    if not re.search(r"\b(?:I|my|me)\b", cue, flags=re.IGNORECASE):
        return False
    return cue.lower().startswith(("wait", "hold on", "let me double-check"))


def deterministic_epsilon(sample_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Number):
        return value != 0
    return str(value).strip().lower() in {"true", "yes", "1"}


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        concurrency: int,
        timeout: float,
        request_retries: int,
        name: str,
    ) -> None:
        endpoint = base_url.rstrip("/")
        if endpoint.endswith("/chat/completions"):
            self.url = endpoint
        elif endpoint.endswith("/v1"):
            self.url = endpoint + "/chat/completions"
        else:
            self.url = endpoint + "/v1/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.request_retries = request_retries
        self.name = name
        self.semaphore = asyncio.Semaphore(concurrency)
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_connections=max(concurrency * 2, 8),
                max_keepalive_connections=max(concurrency, 4),
            ),
        )

    async def generate(
        self,
        prompt: str,
        images_b64: Sequence[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        content: list[dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded}",
                    "detail": "high",
                },
            }
            for encoded in images_b64
        ]
        content.append({"type": "text", "text": prompt})
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key.upper() != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with self.semaphore:
            for attempt in range(self.request_retries + 1):
                try:
                    response = await self.client.post(
                        self.url, headers=headers, json=payload
                    )
                    response.raise_for_status()
                    return str(
                        response.json()["choices"][0]["message"]["content"] or ""
                    )
                except Exception as error:
                    if attempt >= self.request_retries:
                        LOGGER.warning("%s request failed: %s", self.name, error)
                        return ""
                    await asyncio.sleep(min(2**attempt, 8))
        return ""

    async def close(self) -> None:
        await self.client.aclose()


async def generate_structured_answer(
    client: OpenAICompatibleClient,
    prompt: str,
    images: Sequence[str],
    temperature: float,
    max_tokens: int,
    attempts: int,
    max_chars: int,
) -> Generation:
    last = Generation()
    for attempt in range(attempts):
        raw = await client.generate(
            prompt,
            images,
            min(temperature + 0.1 * attempt, 1.0),
            max_tokens,
        )
        last = parse_generation(raw)
        if last.valid and not pathological_output(raw, max_chars):
            return last
    return last


async def get_trigger_decision(
    teacher: OpenAICompatibleClient,
    template: str,
    sample: Sample,
    generation: Generation,
    attempts: int,
    teacher_max_tokens: int,
) -> TriggerDecision:
    prompt = fill_prompt(
        template,
        sample,
        observation=generation.observation,
        reasoning=generation.reasoning,
    )
    last = TriggerDecision(True, "", "", "", False)
    for attempt in range(attempts):
        raw = await teacher.generate(
            prompt, sample.image_b64, 0.2 + 0.1 * attempt, teacher_max_tokens
        )
        last = parse_trigger_decision(raw)
        reason_ok = not last.suggest_reground or (
            bool(last.reason) and not cue_leaks_answer(last.reason, sample)
        )
        confidence_ok = last.suggest_reground or bool(last.confidence_reason)
        if last.parsed and reason_ok and confidence_ok:
            return last
    return last


async def generate_reground_content(
    teacher: OpenAICompatibleClient,
    sample: Sample,
    doubt_reason: str,
    attempts: int,
    teacher_max_tokens: int,
) -> tuple[str, str]:
    prompt = REGROUND_CONTENT_PROMPT.format(doubt_reason=doubt_reason).strip()
    last_raw = ""
    for attempt in range(attempts):
        last_raw = await teacher.generate(
            prompt, sample.image_b64, 0.3 + 0.1 * attempt, teacher_max_tokens
        )
        cue = parse_tag(last_raw, "reground")
        if valid_reground_content(cue, sample):
            return cue, last_raw
    return "", last_raw


def image_tokens(count: int) -> str:
    return "\n".join("<image>" for _ in range(count))


def user_question(sample: Sample) -> str:
    parts = [image_tokens(len(sample.image_refs)), sample.question]
    options = format_options(sample.choices)
    if options:
        parts.append(options)
    return "\n".join(part for part in parts if part)


def natural_think(generation: Generation) -> str:
    return f"{generation.observation.strip()}\n\n{generation.reasoning.strip()}".strip()


def assemble_no_reground(sample: Sample, initial: Generation) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": user_question(sample)},
            {
                "role": "assistant",
                "content": (
                    f"<think>\n{natural_think(initial)}\n</think>\n"
                    f"<answer>{sample.canonical_answer}</answer>"
                ),
            },
        ],
        "images": sample.image_refs,
    }


def assemble_reground(
    sample: Sample,
    initial: Generation,
    cue: str,
    revised: Generation,
) -> dict[str, Any]:
    second_user = (
        image_tokens(len(sample.image_refs))
        + "\nBased on your self-reflection, re-examine the image and provide "
        "the final answer. Do not output <reground> again."
    )
    return {
        "messages": [
            {"role": "user", "content": user_question(sample)},
            {
                "role": "assistant",
                "content": (
                    f"<think>\n{natural_think(initial)}\n</think>\n"
                    f"<reground>\n{cue.strip()}\n</reground>"
                ),
            },
            {"role": "user", "content": second_user},
            {
                "role": "assistant",
                "content": (
                    f"<think>\n{natural_think(revised)}\n</think>\n"
                    f"<answer>{sample.canonical_answer}</answer>"
                ),
            },
        ],
        "images": sample.image_refs + sample.image_refs,
    }


def failure_outcome(sample: Sample, status: str, **details: Any) -> Outcome:
    return Outcome(
        sample_id=sample.sample_id,
        status=status,
        failure={
            "id": sample.sample_id,
            "source": sample.source,
            "status": status,
            **json_safe(details),
        },
    )


async def process_sample(
    sample: Sample,
    student: OpenAICompatibleClient,
    teacher: OpenAICompatibleClient,
    judge: OpenAICompatibleClient,
    args: argparse.Namespace,
) -> Outcome:
    initial_prompt = fill_prompt(ROUND1_PROMPT, sample)
    initial = await generate_structured_answer(
        student,
        initial_prompt,
        sample.image_b64,
        args.round1_temperature,
        args.max_tokens_round1,
        args.round1_attempts,
        args.max_output_chars,
    )
    if not initial.valid:
        return failure_outcome(sample, "dropped:round1_parse")
    if pathological_output(initial.raw, args.max_output_chars):
        return failure_outcome(sample, "dropped:round1_pathological")

    initial_correct = answers_match(initial.answer, sample, args.numeric_tolerance)
    decision: TriggerDecision
    route: str

    if not initial_correct:
        route = "correction"
        decision = await get_trigger_decision(
            teacher,
            CORRECTIVE_CUE_PROMPT,
            sample,
            initial,
            args.teacher_attempts,
            args.teacher_max_tokens,
        )
        decision.suggest_reground = True
    else:
        decision = await get_trigger_decision(
            teacher,
            GROUNDING_JUDGE_PROMPT,
            sample,
            initial,
            args.teacher_attempts,
            args.teacher_max_tokens,
        )
        if not decision.parsed:
            return failure_outcome(sample, "dropped:grounding_judge_parse")
        if decision.suggest_reground:
            route = "grounding"
        elif deterministic_epsilon(sample.sample_id, args.seed) < args.epsilon:
            route = "verification"
            decision = await get_trigger_decision(
                teacher,
                VERIFICATION_CUE_PROMPT,
                sample,
                initial,
                args.teacher_attempts,
                args.teacher_max_tokens,
            )
            decision.suggest_reground = True
        else:
            route = "no_reground"

    base_meta = {
        "id": sample.sample_id,
        "source": sample.source,
        "route": route,
        "initial_answer": initial.answer,
        "canonical_answer": sample.canonical_answer,
        "initial_correct": initial_correct,
        "images": sample.image_refs,
        "metadata": sample.metadata,
    }

    if route == "no_reground":
        return Outcome(
            sample_id=sample.sample_id,
            status="kept:no_reground",
            route=route,
            sft=assemble_no_reground(sample, initial),
            meta={
                **base_meta,
                "grounding_judgment": decision.confidence_reason,
            },
        )

    if not decision.parsed or not decision.reason:
        return failure_outcome(
            sample,
            "dropped:diagnostic_cue_parse",
            route=route,
        )
    if cue_leaks_answer(decision.reason, sample):
        return failure_outcome(sample, "dropped:doubt_reason_leak", route=route)

    cue, cue_raw = await generate_reground_content(
        teacher,
        sample,
        decision.reason,
        args.teacher_attempts,
        args.teacher_max_tokens,
    )
    if not cue:
        return failure_outcome(sample, "dropped:reground_content", route=route)

    round2_prompt = fill_prompt(
        ROUND2_PROMPT,
        sample,
        previous_observation=initial.observation,
        reground_instruction=cue,
    )
    revised = Generation()
    for attempt in range(args.round2_attempts):
        revised = await generate_structured_answer(
            student,
            round2_prompt,
            sample.image_b64,
            min(args.round2_temperature + 0.1 * attempt, 1.0),
            args.max_tokens_round2,
            1,
            args.max_output_chars,
        )
        if revised.valid and answers_match(
            revised.answer, sample, args.numeric_tolerance
        ):
            break
    if not revised.valid:
        return failure_outcome(sample, "dropped:round2_parse", route=route)
    if not answers_match(revised.answer, sample, args.numeric_tolerance):
        return failure_outcome(
            sample,
            "dropped:round2_incorrect",
            route=route,
            predicted_answer=revised.answer,
        )

    judge_result: dict[str, Any] = {
        "pass": True,
        "is_grounded": True,
        "has_material_update": True,
        "follows_diagnostic_cue": True,
        "issues": [],
    }
    if args.require_round2_judge:
        judge_prompt = fill_prompt(
            ROUND2_JUDGE_PROMPT,
            sample,
            initial_observation=initial.observation,
            reground_instruction=cue,
            updated_observation=revised.observation,
            updated_reasoning=revised.reasoning,
        )
        raw_judge = await judge.generate(
            judge_prompt,
            sample.image_b64,
            0.0,
            args.teacher_max_tokens,
        )
        parsed_judge = extract_json(raw_judge)
        if parsed_judge is None:
            return failure_outcome(sample, "dropped:round2_judge_parse", route=route)
        judge_result = parsed_judge
        required = (
            truthy(judge_result.get("pass"))
            and truthy(judge_result.get("is_grounded"))
            and truthy(judge_result.get("follows_diagnostic_cue"))
            and truthy(judge_result.get("has_material_update"))
        )
        if not required:
            return failure_outcome(
                sample,
                "dropped:round2_quality",
                route=route,
                issues=judge_result.get("issues", []),
            )

    return Outcome(
        sample_id=sample.sample_id,
        status=f"kept:{route}",
        route=route,
        sft=assemble_reground(sample, initial, cue, revised),
        meta={
            **base_meta,
            "doubt_reason": decision.reason,
            "reground_instruction": cue,
            "revised_answer": revised.answer,
            "round2_judgment": json_safe(judge_result),
            "teacher_raw_available": bool(decision.raw and cue_raw),
        },
    )


def write_json_line(handle: Any, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def write_checkpoint(path: Path, statuses: dict[str, str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump({"statuses": statuses}, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def load_checkpoint(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    statuses = value.get("statuses", {}) if isinstance(value, dict) else {}
    return {str(key): str(status) for key, status in statuses.items()}


async def run_generation(
    samples: Sequence[Sample],
    student: OpenAICompatibleClient,
    teacher: OpenAICompatibleClient,
    judge: OpenAICompatibleClient,
    args: argparse.Namespace,
) -> Counter:
    output_dir = Path(args.output_dir).resolve()
    sft_path = output_dir / "sft.jsonl"
    meta_path = output_dir / "meta.jsonl"
    failures_path = output_dir / "failures.jsonl"
    checkpoint_path = output_dir / "checkpoint.json"

    existing_outputs = [
        path
        for path in (sft_path, meta_path, failures_path, checkpoint_path)
        if path.exists() and path.stat().st_size > 0
    ]
    if existing_outputs and not args.resume and not args.overwrite:
        names = ", ".join(path.name for path in existing_outputs)
        raise FileExistsError(
            f"Output files already exist ({names}); use --resume or --overwrite"
        )

    statuses = load_checkpoint(checkpoint_path) if args.resume else {}
    if args.resume:
        if args.retry_failures:
            completed = {
                key for key, status in statuses.items() if status.startswith("kept:")
            }
        else:
            completed = set(statuses)
        mode = "a"
    else:
        completed = set()
        statuses = {}
        mode = "w"

    todo = [sample for sample in samples if sample.sample_id not in completed]
    LOGGER.info(
        "Samples ready: %d; skipped by resume: %d", len(todo), len(samples) - len(todo)
    )
    pipeline_semaphore = asyncio.Semaphore(args.sample_concurrent)

    async def guarded(sample: Sample) -> Outcome:
        async with pipeline_semaphore:
            try:
                return await process_sample(sample, student, teacher, judge, args)
            except Exception as error:
                LOGGER.exception("Unhandled sample failure for %s", sample.sample_id)
                return failure_outcome(
                    sample,
                    "dropped:exception",
                    error=f"{type(error).__name__}: {error}",
                )

    counts: Counter = Counter()
    pending_since_checkpoint = 0

    with (
        sft_path.open(mode, encoding="utf-8") as sft_handle,
        meta_path.open(mode, encoding="utf-8") as meta_handle,
        failures_path.open(mode, encoding="utf-8") as failures_handle,
    ):
        progress = tqdm(total=len(todo), desc="Generating SFT")
        task_buffer = max(args.sample_concurrent * 4, 1)
        for start in range(0, len(todo), task_buffer):
            tasks = [
                asyncio.create_task(guarded(sample))
                for sample in todo[start : start + task_buffer]
            ]
            for future in asyncio.as_completed(tasks):
                outcome = await future
                statuses[outcome.sample_id] = outcome.status
                counts[outcome.status] += 1
                if outcome.sft is not None:
                    write_json_line(sft_handle, outcome.sft)
                if outcome.meta is not None:
                    write_json_line(meta_handle, outcome.meta)
                if outcome.failure is not None:
                    write_json_line(failures_handle, outcome.failure)

                pending_since_checkpoint += 1
                if pending_since_checkpoint >= args.checkpoint_every:
                    sft_handle.flush()
                    meta_handle.flush()
                    failures_handle.flush()
                    write_checkpoint(checkpoint_path, statuses)
                    pending_since_checkpoint = 0
                progress.update(1)
                progress.set_postfix(
                    kept=sum(
                        value
                        for status, value in counts.items()
                        if status.startswith("kept:")
                    )
                )
        progress.close()

    write_checkpoint(checkpoint_path, statuses)
    return counts


def environment_key(variable_name: str, fallback: str = "EMPTY") -> str:
    if not variable_name:
        return fallback
    return os.getenv(variable_name) or fallback


def loading_report(stats: LoadingStats) -> dict[str, Any]:
    return {
        "records": stats.records,
        "loaded": stats.loaded,
        "invalid": stats.invalid,
        "duplicate_ids": stats.duplicate_ids,
        "missing_images": stats.missing_images,
        "top_errors": stats.errors.most_common(20),
    }


async def async_main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples, stats = load_samples(args)
    report = loading_report(stats)
    with (output_dir / "input_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    LOGGER.info("Loaded %d/%d records", stats.loaded, stats.records)
    if not samples:
        raise RuntimeError("No valid samples were loaded")

    if args.validate_only:
        preview = {
            "id": samples[0].sample_id,
            "source": samples[0].source,
            "question": samples[0].question,
            "answer": samples[0].canonical_answer,
            "choices": samples[0].choices,
            "images": samples[0].image_refs,
        }
        print(
            json.dumps(
                {"input_report": report, "preview": preview},
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    student_key = environment_key(args.student_api_key_env)
    teacher_key = environment_key(args.teacher_api_key_env)
    judge_key = environment_key(args.judge_api_key_env, teacher_key)

    student = OpenAICompatibleClient(
        args.student_url,
        args.student_model,
        student_key,
        args.student_concurrent,
        args.timeout,
        args.request_retries,
        "student",
    )
    teacher = OpenAICompatibleClient(
        args.teacher_url,
        args.teacher_model,
        teacher_key,
        args.teacher_concurrent,
        args.timeout,
        args.request_retries,
        "teacher",
    )
    judge = OpenAICompatibleClient(
        args.judge_url or args.teacher_url,
        args.judge_model or args.teacher_model,
        judge_key,
        args.judge_concurrent,
        args.timeout,
        args.request_retries,
        "judge",
    )

    try:
        counts = await run_generation(samples, student, teacher, judge, args)
    finally:
        await student.close()
        await teacher.close()
        await judge.close()

    kept = sum(value for key, value in counts.items() if key.startswith("kept:"))
    summary = {
        "loaded": len(samples),
        "kept": kept,
        "dropped": sum(counts.values()) - kept,
        "status_counts": dict(sorted(counts.items())),
        "epsilon": args.epsilon,
        "seed": args.seed,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate dataset-agnostic ReGround SFT trajectories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input file, directory, or glob; repeatable",
    )
    parser.add_argument(
        "--input-pattern", default="*", help="Pattern used when an input is a directory"
    )
    parser.add_argument("--output-dir", default="./output_reground_sft")
    parser.add_argument(
        "--num-samples", type=int, default=0, help="0 keeps all valid records"
    )
    parser.add_argument(
        "--shuffle", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--seed", type=int, default=42)

    fields = parser.add_argument_group("input field mapping")
    fields.add_argument("--id-field", default="id")
    fields.add_argument("--question-field", default="question")
    fields.add_argument("--answer-field", default="answer")
    fields.add_argument("--choices-field", default="choices")
    fields.add_argument("--image-field", default="image")
    fields.add_argument("--knowledge-field", default="knowledge")
    fields.add_argument("--source-field", default="source")
    fields.add_argument("--metadata-fields", default="")
    fields.add_argument("--answer-index-base", choices=("zero", "one"), default="zero")
    fields.add_argument(
        "--image-root", default="", help="Root for relative image paths"
    )
    fields.add_argument("--image-mode", choices=("copy", "reference"), default="copy")
    fields.add_argument("--jpeg-quality", type=int, default=90)

    models = parser.add_argument_group("OpenAI-compatible model services")
    models.add_argument("--student-url", default="http://localhost:8000/v1")
    models.add_argument("--student-model", default="policy-model")
    models.add_argument("--teacher-url", default="http://localhost:8001/v1")
    models.add_argument("--teacher-model", default="teacher-model")
    models.add_argument("--judge-url", default="", help="Defaults to teacher URL")
    models.add_argument("--judge-model", default="", help="Defaults to teacher model")
    models.add_argument("--student-api-key-env", default="REGROUND_STUDENT_API_KEY")
    models.add_argument("--teacher-api-key-env", default="REGROUND_TEACHER_API_KEY")
    models.add_argument("--judge-api-key-env", default="REGROUND_JUDGE_API_KEY")

    generation = parser.add_argument_group("generation and filtering")
    generation.add_argument("--epsilon", type=float, default=0.15)
    generation.add_argument("--round1-temperature", type=float, default=0.7)
    generation.add_argument("--round2-temperature", type=float, default=0.5)
    generation.add_argument("--max-tokens-round1", type=int, default=8192)
    generation.add_argument("--max-tokens-round2", type=int, default=8192)
    generation.add_argument("--teacher-max-tokens", type=int, default=512)
    generation.add_argument("--round1-attempts", type=int, default=2)
    generation.add_argument("--round2-attempts", type=int, default=2)
    generation.add_argument("--teacher-attempts", type=int, default=3)
    generation.add_argument("--numeric-tolerance", type=float, default=1e-6)
    generation.add_argument("--max-output-chars", type=int, default=30000)
    generation.add_argument(
        "--require-round2-judge", action=argparse.BooleanOptionalAction, default=True
    )

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--sample-concurrent", type=int, default=16)
    runtime.add_argument("--student-concurrent", type=int, default=16)
    runtime.add_argument("--teacher-concurrent", type=int, default=8)
    runtime.add_argument("--judge-concurrent", type=int, default=8)
    runtime.add_argument("--timeout", type=float, default=300)
    runtime.add_argument("--request-retries", type=int, default=3)
    runtime.add_argument("--checkpoint-every", type=int, default=20)
    runtime.add_argument("--resume", action="store_true")
    runtime.add_argument("--retry-failures", action="store_true")
    runtime.add_argument("--overwrite", action="store_true")
    runtime.add_argument("--validate-only", action="store_true")
    runtime.add_argument("--verbose", action="store_true")
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if not 0.0 <= args.epsilon <= 1.0:
        raise ValueError("--epsilon must be between 0 and 1")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")
    positive = {
        "sample_concurrent": args.sample_concurrent,
        "student_concurrent": args.student_concurrent,
        "teacher_concurrent": args.teacher_concurrent,
        "judge_concurrent": args.judge_concurrent,
        "checkpoint_every": args.checkpoint_every,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        raise ValueError("These options must be positive: " + ", ".join(invalid))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)
    validate_arguments(args)
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
