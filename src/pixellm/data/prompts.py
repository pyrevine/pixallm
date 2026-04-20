"""Prompt normalization for PixelLM datasets."""

PROMPT_PREFIX = "Draw 16x16 pixel art as Palette Index Grid DSL:"


def normalize_caption(caption: str) -> str:
    """Normalize source captions while preserving their semantic content."""

    return " ".join(caption.strip().split())


def build_prompt(caption: str) -> str:
    """Build the user-facing model prompt from a source caption."""

    normalized = normalize_caption(caption)
    return f"{PROMPT_PREFIX} {normalized}"
