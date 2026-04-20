"""Render prepared PixelLM JSONL samples into a contact sheet."""

from __future__ import annotations

import argparse
import json
import random
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from pixellm.dsl import parse_dsl
from pixellm.render import render_pixel_art


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview prepared PixelLM JSONL samples.")
    parser.add_argument("--input", default="data/processed/train_v1.jsonl")
    parser.add_argument("--output", default="docs/results/gallery/train_preview.png")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--sample", action="store_true", help="Randomly sample rows instead of taking the first N.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=int, default=12)
    parser.add_argument("--columns", type=int, default=5)
    args = parser.parse_args()

    records = load_records(args.input)
    selected = select_records(records, limit=args.limit, sample=args.sample, seed=args.seed)
    sheet = build_contact_sheet(selected, scale=args.scale, columns=args.columns)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    print(f"wrote {output_path} ({len(selected)} samples)")


def load_records(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def select_records(
    records: list[dict[str, Any]], *, limit: int, sample: bool, seed: int
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    if not sample:
        return records[:limit]
    rng = random.Random(seed)
    count = min(limit, len(records))
    return rng.sample(records, count)


def build_contact_sheet(
    records: list[dict[str, Any]], *, scale: int = 12, columns: int = 5
) -> Image.Image:
    if scale < 1:
        raise ValueError("scale must be >= 1.")
    if columns < 1:
        raise ValueError("columns must be >= 1.")

    font = ImageFont.load_default()
    sprite_size = 16 * scale
    label_height = 64
    padding = 12
    cell_width = sprite_size + padding * 2
    cell_height = sprite_size + label_height + padding * 2
    rows = max(1, (len(records) + columns - 1) // columns)

    sheet = Image.new("RGBA", (columns * cell_width, rows * cell_height), (250, 250, 250, 255))
    draw = ImageDraw.Draw(sheet)

    for idx, record in enumerate(records):
        col = idx % columns
        row = idx // columns
        x0 = col * cell_width
        y0 = row * cell_height

        sprite = render_pixel_art(parse_dsl(record["dsl"]), scale=scale)
        checker = make_checkerboard(sprite.size, square=max(4, scale // 2))
        checker.alpha_composite(sprite)
        sheet.alpha_composite(checker, (x0 + padding, y0 + padding))

        label = build_label(idx, record)
        draw.multiline_text(
            (x0 + padding, y0 + padding + sprite_size + 6),
            label,
            fill=(20, 20, 20, 255),
            font=font,
            spacing=2,
        )

    return sheet


def make_checkerboard(size: tuple[int, int], *, square: int) -> Image.Image:
    image = Image.new("RGBA", size, (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    for y in range(0, size[1], square):
        for x in range(0, size[0], square):
            if (x // square + y // square) % 2 == 0:
                draw.rectangle((x, y, x + square - 1, y + square - 1), fill=(220, 220, 220, 255))
    return image


def build_label(idx: int, record: dict[str, Any]) -> str:
    source_id = str(record.get("source_id", f"{idx:03d}")).replace("train:", "")
    caption = str(record.get("caption", "")).strip()
    wrapped = textwrap.wrap(caption, width=30, max_lines=2, placeholder="...")
    return f"{idx:02d} / {source_id}\n" + "\n".join(wrapped)


if __name__ == "__main__":
    main()
