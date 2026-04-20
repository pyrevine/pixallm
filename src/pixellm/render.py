"""Render Palette Index Grid sprites to PIL images."""

from __future__ import annotations

from PIL import Image

from pixellm.dsl import GRID_SIZE, PixelArt, parse_dsl


def render_pixel_art(pixel_art: PixelArt | str, scale: int = 20) -> Image.Image:
    """Render PixelArt or tag DSL to an RGBA PIL image."""

    if scale < 1:
        raise ValueError("scale must be >= 1.")

    art = parse_dsl(pixel_art) if isinstance(pixel_art, str) else pixel_art
    pixels: list[tuple[int, int, int, int]] = []
    for row in art.pixels:
        for index in row:
            if index == "0":
                pixels.append((0, 0, 0, 0))
            else:
                pixels.append((*hex_to_rgb(art.palette[index]), 255))

    image = Image.new("RGBA", (GRID_SIZE, GRID_SIZE))
    image.putdata(pixels)
    if scale == 1:
        return image
    return image.resize((GRID_SIZE * scale, GRID_SIZE * scale), Image.Resampling.NEAREST)


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert #RRGGBB to an RGB tuple."""

    value = hex_color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Invalid hex color: {hex_color!r}")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
