"""Shared constants and helpers for the ReGround two-round protocol."""

from __future__ import annotations

import re

REGROUND_OPEN_PATTERN = re.compile(r"<\s*reground\s*>", re.IGNORECASE)

# Keep this synchronized with the public inference adapter and SFT generator.
REGROUND_USER_PROMPT = (
    "Based on your self-reflection, re-examine the image and provide "
    "the final answer. Do not output <reground> again."
)


def has_reground_marker(text: str) -> bool:
    """Return whether the policy explicitly requested visual re-examination."""

    return bool(REGROUND_OPEN_PATTERN.search(text or ""))


def strip_environment_text(text: str) -> str:
    """Remove the masked Round-2 environment message before reward parsing."""

    return (text or "").replace(REGROUND_USER_PROMPT, "")
