"""Dataset and image conversion utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from pixallm.data.prompts import build_prompt, normalize_caption
from pixallm.dsl import GRID_SIZE, PixelArt, serialize_dsl
from pixallm.render import render_pixel_art

DEFAULT_ALPHA_THRESHOLD = 127
DEFAULT_MAX_NONTRANSPARENT_COLORS = 7
DEFAULT_BACKGROUND_TOLERANCE = 8


@dataclass(frozen=True)
class PreparedImage:
    pixel_art: PixelArt
    canonical_image: Image.Image


def prepare_image(
    image: Image.Image,
    *,
    target_size: int = GRID_SIZE,
    alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
    max_nontransparent_colors: int = DEFAULT_MAX_NONTRANSPARENT_COLORS,
    remove_border_background: bool = False,
    background_tolerance: int = DEFAULT_BACKGROUND_TOLERANCE,
) -> PreparedImage:
    """Convert an image into PixelArt plus its canonical 16x16 rendering."""

    if target_size != GRID_SIZE:
        raise ValueError("Only 16x16 output is supported.")
    if not 0 <= alpha_threshold <= 255:
        raise ValueError("alpha_threshold must be between 0 and 255.")
    if not 1 <= max_nontransparent_colors <= DEFAULT_MAX_NONTRANSPARENT_COLORS:
        raise ValueError("max_nontransparent_colors must be between 1 and 7.")
    if background_tolerance < 0:
        raise ValueError("background_tolerance must be >= 0.")

    resized = image.convert("RGBA").resize((target_size, target_size), Image.Resampling.NEAREST)
    if remove_border_background:
        resized = remove_background_from_edges(resized, tolerance=background_tolerance)

    rgba_pixels = _image_data(resized)
    opaque_positions = [idx for idx, pixel in enumerate(rgba_pixels) if pixel[3] > alpha_threshold]

    if not opaque_positions:
        pixel_art = PixelArt(
            palette={"0": "transparent"},
            pixels=["0" * target_size for _ in range(target_size)],
        )
        return PreparedImage(pixel_art=pixel_art, canonical_image=render_pixel_art(pixel_art, scale=1))

    rgb_strip = Image.new("RGB", (len(opaque_positions), 1))
    rgb_strip.putdata([rgba_pixels[idx][:3] for idx in opaque_positions])
    color_count = min(max_nontransparent_colors, len(set(_image_data(rgb_strip))))
    quantized = rgb_strip.quantize(colors=color_count, method=Image.Quantize.MEDIANCUT)
    raw_palette = quantized.getpalette() or []

    dsl_palette: dict[str, str] = {"0": "transparent"}
    quantized_to_dsl_index: dict[int, str] = {}
    for q_index in sorted(set(_image_data(quantized))):
        dsl_index = str(len(quantized_to_dsl_index) + 1)
        offset = q_index * 3
        rgb = raw_palette[offset : offset + 3]
        dsl_palette[dsl_index] = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        quantized_to_dsl_index[q_index] = dsl_index

    flat_indices = ["0"] * (target_size * target_size)
    for strip_x, pixel_index in enumerate(opaque_positions):
        q_index = quantized.getpixel((strip_x, 0))
        flat_indices[pixel_index] = quantized_to_dsl_index[q_index]

    rows = [
        "".join(flat_indices[row_start : row_start + target_size])
        for row_start in range(0, target_size * target_size, target_size)
    ]
    pixel_art = PixelArt(palette=dsl_palette, pixels=rows)
    return PreparedImage(pixel_art=pixel_art, canonical_image=render_pixel_art(pixel_art, scale=1))


def image_to_pixel_art(image: Image.Image, *, remove_border_background: bool = False) -> PixelArt:
    """Convert an image to internal PixelArt."""

    return prepare_image(image, remove_border_background=remove_border_background).pixel_art


def image_to_dsl(image: Image.Image, *, remove_border_background: bool = False) -> str:
    """Convert an image to external tag DSL."""

    return serialize_dsl(image_to_pixel_art(image, remove_border_background=remove_border_background))


def build_training_record(
    *,
    caption: str,
    dsl: str,
    source: str,
    source_id: str,
    category: str,
    view: str,
    license: str,
) -> dict[str, str]:
    """Build one normalized training JSONL record."""

    normalized_caption = normalize_caption(caption)
    return {
        "caption": normalized_caption,
        "prompt": build_prompt(normalized_caption),
        "dsl": dsl,
        "source": source,
        "source_id": source_id,
        "category": category,
        "view": view,
        "license": license,
    }


def write_jsonl(records: Iterable[dict[str, Any]], output_path: str | Path) -> None:
    """Write records as UTF-8 JSONL."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def iter_nouns_records(limit: int | None = None) -> Iterable[dict[str, str]]:
    """Yield normalized records from m1guelpf/nouns.

    This function imports datasets lazily so local DSL tests do not need to load
    dataset machinery.
    """

    if limit is not None and limit <= 0:
        return

    from datasets import load_dataset

    dataset = load_dataset("m1guelpf/nouns", split="train")
    for row_idx, item in enumerate(dataset):
        if limit is not None and row_idx >= limit:
            break
        dsl = image_to_dsl(item["image"], remove_border_background=True)
        yield build_training_record(
            caption=item["text"],
            dsl=dsl,
            source="m1guelpf/nouns",
            source_id=f"train:{row_idx:06d}",
            category="character",
            view="front",
            license="cc0-1.0",
        )


def remove_background_from_edges(image: Image.Image, *, tolerance: int) -> Image.Image:
    """Make border-connected background pixels transparent.

    Nouns-style samples have a flat background. Flood filling from edges avoids
    removing interior pixels that happen to share the background color.
    """

    rgba = image.convert("RGBA")
    pixels = rgba.load()
    width, height = rgba.size
    background = _dominant_corner_color(rgba)
    stack = [
        *[(x, 0) for x in range(width)],
        *[(x, height - 1) for x in range(width)],
        *[(0, y) for y in range(height)],
        *[(width - 1, y) for y in range(height)],
    ]
    seen: set[tuple[int, int]] = set()

    while stack:
        x, y = stack.pop()
        if (x, y) in seen:
            continue
        seen.add((x, y))

        color = pixels[x, y]
        if color[3] == 0 or not _color_close(color[:3], background, tolerance):
            continue

        pixels[x, y] = (color[0], color[1], color[2], 0)
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < width and 0 <= ny < height:
                stack.append((nx, ny))

    return rgba


def _dominant_corner_color(image: Image.Image) -> tuple[int, int, int]:
    width, height = image.size
    corners = [
        image.getpixel((0, 0))[:3],
        image.getpixel((width - 1, 0))[:3],
        image.getpixel((0, height - 1))[:3],
        image.getpixel((width - 1, height - 1))[:3],
    ]
    return max(set(corners), key=corners.count)


def _color_close(left: tuple[int, int, int], right: tuple[int, int, int], tolerance: int) -> bool:
    return all(abs(a - b) <= tolerance for a, b in zip(left, right, strict=True))


def _image_data(image: Image.Image) -> list[Any]:
    get_flattened_data = getattr(image, "get_flattened_data", None)
    if get_flattened_data is not None:
        return list(get_flattened_data())
    return list(image.getdata())
