"""Rule-based ReGround reward for veRL GRPO.

The implementation follows the paper's Stage-2 reward:

    R = lambda_reg * I_reg
        + lambda_acc * (I_acc - gamma * I_reg * I_acc - beta)
        + lambda_form * R_form

where ``R_form`` is -1 for malformed output and 0 otherwise. Both online
indicators are deterministic: ``I_reg`` checks the trajectory structure, and
``I_acc`` compares answer values after the same VLMEvalKit parsing stage used
for final scoring. With the supplementary hyperparameters, the four
valid-format quadrants round to 1.0, 0.6, 0.4, and -0.1.
"""

from __future__ import annotations

from collections.abc import Iterable
import math
import re
from typing import Any

from grpo.protocol import has_reground_marker, strip_environment_text

TAG_PATTERN = re.compile(r"<\s*(/?)\s*(think|reground|answer)\s*>", re.IGNORECASE)
ANSWER_PATTERN = re.compile(
    r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", re.IGNORECASE | re.DOTALL
)
REGROUND_PATTERN = re.compile(
    r"<\s*reground\s*>(.*?)<\s*/\s*reground\s*>", re.IGNORECASE | re.DOTALL
)
BOXED_PATTERN = re.compile(r"^\\boxed\s*\{(.*)\}$", re.DOTALL)
OPTION_PATTERN = re.compile(r"^(?:option\s*)?\(?([A-Z])\)?[\s.:)]*$", re.IGNORECASE)
NUMBER_PATTERN = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?%?$"
)
DIRECT_TAG_SEQUENCE = ["think", "/think", "answer", "/answer"]
TRIGGERED_TAG_SEQUENCE = [
    "think",
    "/think",
    "reground",
    "/reground",
    "think",
    "/think",
    "answer",
    "/answer",
]


def _tag_sequence(text: str) -> list[str]:
    return [
        f"{'/' if match.group(1) else ''}{match.group(2).lower()}"
        for match in TAG_PATTERN.finditer(text)
    ]


def _format_valid(text: str, answer: str, max_answer_chars: int) -> bool:
    return (
        bool(answer)
        and len(answer) <= max_answer_chars
        and _tag_sequence(text) in (DIRECT_TAG_SEQUENCE, TRIGGERED_TAG_SEQUENCE)
    )


def _structurally_valid_reground(text: str, min_reground_chars: int) -> bool:
    """Check one non-empty reground span at the template-defined position.

    This intentionally does not score diagnostic semantics. The SFT prior and
    KL anchor supply that behavior; the online indicator is binary, so longer
    or padded content receives no additional reward.
    """

    if _tag_sequence(text) != TRIGGERED_TAG_SEQUENCE:
        return False
    diagnoses = REGROUND_PATTERN.findall(text)
    return len(diagnoses) == 1 and len(diagnoses[0].strip()) >= min_reground_chars


def _unwrap_answer(value: str) -> str:
    value = value.strip()
    boxed = BOXED_PATTERN.fullmatch(value)
    return boxed.group(1).strip() if boxed else value


def _normalize(value: Any) -> str:
    text = _unwrap_answer(str(value))
    text = text.replace("−", "-").replace("–", "-")
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return text.rstrip(". ,;:")


def _as_number(value: str) -> float | None:
    compact = value.replace(",", "").replace(" ", "")
    if not NUMBER_PATTERN.fullmatch(compact):
        return None
    percent = compact.endswith("%")
    if percent:
        compact = compact[:-1]
    try:
        number = float(compact)
    except ValueError:
        return None
    return number / 100.0 if percent else number


def _iter_answers(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, dict):
        for key in ("accepted_answers", "answers", "answer", "ground_truth"):
            if key in value:
                return _iter_answers(value[key])
        return [value]
    if isinstance(value, str | bytes):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _choice_label(value: Any, choices: list[Any]) -> str | None:
    normalized = _normalize(value)
    option = OPTION_PATTERN.fullmatch(normalized.upper())
    if option:
        return option.group(1).upper()
    for index, choice in enumerate(choices):
        if _normalize(choice) == normalized and index < 26:
            return chr(ord("A") + index)
    return None


def answers_match(prediction: str, ground_truth: Any, extra_info: dict[str, Any] | None = None) -> bool:
    """Compare prediction and target after VLMEvalKit answer parsing.

    Dataset-specific extraction and normalization are performed by the same
    VLMEvalKit parser used for final scoring before values enter the reward
    pipeline. This helper is the deterministic post-parser comparison layer,
    not a second benchmark parser and not a learned judge. It preserves
    canonical choice aliases and numeric equivalence in the prepared records.
    """

    extra_info = extra_info or {}
    accepted = _iter_answers(extra_info.get("accepted_answers")) or _iter_answers(ground_truth)
    choices = _iter_answers(extra_info.get("choices"))
    predicted_choice = _choice_label(prediction, choices) if choices else None

    for candidate in accepted:
        if choices:
            candidate_choice = _choice_label(candidate, choices)
            if predicted_choice and candidate_choice and predicted_choice == candidate_choice:
                return True

        predicted_text = _normalize(prediction)
        candidate_text = _normalize(candidate)
        if predicted_text == candidate_text:
            return True

        predicted_number = _as_number(predicted_text)
        candidate_number = _as_number(candidate_text)
        if predicted_number is not None and candidate_number is not None:
            if math.isclose(predicted_number, candidate_number, rel_tol=1e-6, abs_tol=1e-8):
                return True
    return False


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    lambda_reg: float = 0.5,
    lambda_acc: float = 0.7,
    lambda_form: float = 0.01,
    gamma: float = 0.14,
    beta: float = 0.14,
    max_answer_chars: int = 256,
    min_reground_chars: int = 1,
) -> dict[str, float]:
    """Compute the paper-aligned deterministic reward and log components."""

    del data_source
    trajectory = strip_environment_text(solution_str)
    answers = ANSWER_PATTERN.findall(trajectory)
    prediction = answers[-1].strip() if answers else ""

    reground = float(_structurally_valid_reground(trajectory, min_reground_chars))
    correct = float(answers_match(prediction, ground_truth, extra_info))
    format_ok = float(_format_valid(trajectory, prediction, max_answer_chars))
    format_reward = 0.0 if format_ok else -1.0
    accuracy_reward = correct - gamma * reground * correct - beta

    reground_component = lambda_reg * reground
    accuracy_component = lambda_acc * accuracy_reward
    format_component = lambda_form * format_reward
    score = reground_component + accuracy_component + format_component

    return {
        "score": float(score),
        "reground_component": float(reground_component),
        "accuracy_component": float(accuracy_component),
        "format_component": float(format_component),
        "trigger_rate": float(has_reground_marker(trajectory)),
        "structural_reground": reground,
        "acc": correct,
        "format_ok": format_ok,
    }
