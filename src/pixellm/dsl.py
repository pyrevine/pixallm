"""Palette Index Grid DSL parsing and validation."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

GRID_SIZE = 16
TRANSPARENT_INDEX = "0"
MAX_PALETTE_SIZE = 8
VALID_INDICES = frozenset(str(i) for i in range(MAX_PALETTE_SIZE))
HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
PALETTE_TAG_RE = re.compile(r"<PALETTE>\s*(.*?)\s*</PALETTE>", re.DOTALL)
GRID_TAG_RE = re.compile(r"<GRID>\s*(.*?)\s*</GRID>", re.DOTALL)


class PixelArt(BaseModel):
    """Internal representation for a 16x16 Palette Index Grid sprite."""

    model_config = ConfigDict(extra="forbid")

    size: Literal[16] = GRID_SIZE
    palette: dict[str, str] = Field(
        description="Palette indices. 0 must be transparent; 1-7 are #RRGGBB colors."
    )
    pixels: list[str] = Field(description="Exactly 16 rows, each with 16 index chars.")

    @model_validator(mode="after")
    def _validate_constraints(self) -> "PixelArt":
        validate_pixel_art(self)
        return self


class DSLParseError(ValueError):
    """Raised when Palette Index Grid text cannot be parsed."""


def validate_pixel_art(pixel_art: PixelArt) -> None:
    """Validate Palette Index Grid constraints.

    The external DSL uses indices 0-7. Index 0 is always transparent, and any
    non-zero index used in the grid must be declared in the palette.
    """

    if pixel_art.size != GRID_SIZE:
        raise ValueError("PixelArt size must be 16.")

    if len(pixel_art.palette) > MAX_PALETTE_SIZE:
        raise ValueError("Palette can contain at most 8 entries, including 0.")

    if TRANSPARENT_INDEX not in pixel_art.palette:
        raise ValueError("Palette must include 0:transparent.")

    normalized_palette: dict[str, str] = {}
    for key, value in pixel_art.palette.items():
        if key not in VALID_INDICES:
            raise ValueError(f"Invalid palette index {key!r}; expected 0-7.")

        if key == TRANSPARENT_INDEX:
            if value.lower() != "transparent":
                raise ValueError("Palette index 0 must be transparent.")
            normalized_palette[key] = "transparent"
            continue

        if not HEX_COLOR_RE.match(value):
            raise ValueError(f"Palette index {key} must be a #RRGGBB color.")
        normalized_palette[key] = value.lower()

    if len(pixel_art.pixels) != GRID_SIZE:
        raise ValueError("Grid must contain exactly 16 rows.")

    declared_indices = set(normalized_palette)
    for row_idx, row in enumerate(pixel_art.pixels):
        if len(row) != GRID_SIZE:
            raise ValueError(f"Grid row {row_idx} must contain exactly 16 indices.")

        invalid_chars = set(row) - VALID_INDICES
        if invalid_chars:
            chars = ", ".join(sorted(invalid_chars))
            raise ValueError(f"Grid row {row_idx} contains invalid index chars: {chars}.")

        undeclared = {ch for ch in row if ch != TRANSPARENT_INDEX and ch not in declared_indices}
        if undeclared:
            chars = ", ".join(sorted(undeclared))
            raise ValueError(f"Grid row {row_idx} uses undeclared palette indices: {chars}.")

    pixel_art.palette = normalized_palette


def parse_dsl(text: str) -> PixelArt:
    """Parse a Palette Index Grid DSL string into a validated PixelArt."""

    palette_match = PALETTE_TAG_RE.search(text)
    grid_match = GRID_TAG_RE.search(text)
    if not palette_match or not grid_match:
        raise DSLParseError("DSL must contain <PALETTE> and <GRID> sections.")

    palette = _parse_palette(palette_match.group(1))
    pixels = _parse_grid(grid_match.group(1))
    try:
        return PixelArt(palette=palette, pixels=pixels)
    except ValueError as exc:
        raise DSLParseError(str(exc)) from exc


def serialize_dsl(pixel_art: PixelArt) -> str:
    """Serialize a PixelArt object to the external tag DSL format."""

    validate_pixel_art(pixel_art)
    palette_items = sorted(pixel_art.palette.items(), key=lambda item: int(item[0]))
    palette_text = ", ".join(f"{idx}:{value}" for idx, value in palette_items)
    grid_text = "\n".join(pixel_art.pixels)
    return f"<PALETTE>\n{palette_text}\n</PALETTE>\n<GRID>\n{grid_text}\n</GRID>"


def _parse_palette(body: str) -> dict[str, str]:
    entries = [part.strip() for part in re.split(r"[,\n]+", body) if part.strip()]
    if not entries:
        raise DSLParseError("Palette section is empty.")

    palette: dict[str, str] = {}
    for entry in entries:
        if ":" not in entry:
            raise DSLParseError(f"Invalid palette entry {entry!r}.")
        key, value = [part.strip() for part in entry.split(":", 1)]
        if key in palette:
            raise DSLParseError(f"Duplicate palette index {key!r}.")
        palette[key] = value.lower() if value.lower() == "transparent" else value
    return palette


def _parse_grid(body: str) -> list[str]:
    rows = [line.strip() for line in body.splitlines() if line.strip()]
    if not rows:
        raise DSLParseError("Grid section is empty.")
    return rows
