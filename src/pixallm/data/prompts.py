"""Prompt normalization for PixelLM datasets."""

from __future__ import annotations

import re

PROMPT_PREFIX = "Draw 16x16 pixel art as Palette Index Grid DSL:"
BACKGROUND_SUFFIX_RE = re.compile(r"\s+on an?\s+[-a-zA-Z ]+\s+background\.?$", re.IGNORECASE)


def normalize_caption(caption: str) -> str:
    """Normalize source captions while preserving their semantic content."""

    normalized = " ".join(caption.strip().split())
    return BACKGROUND_SUFFIX_RE.sub("", normalized).strip()


def build_prompt(caption: str) -> str:
    """Build the user-facing model prompt from a source caption."""

    normalized = normalize_caption(caption)
    return f"{PROMPT_PREFIX} {normalized}"
