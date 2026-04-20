"""Deterministic evaluation metrics for Palette Index Grid outputs."""

from __future__ import annotations

from collections import deque

from pixellm.dsl import GRID_SIZE, PixelArt, parse_dsl


def parse_rate(outputs: list[str]) -> float:
    """Return the fraction of outputs that parse as valid tag DSL."""

    if not outputs:
        return 0.0
    success = 0
    for output in outputs:
        try:
            parse_dsl(output)
        except ValueError:
            continue
        success += 1
    return success / len(outputs)


def palette_constraint_score(pixel_art: PixelArt | str) -> float:
    """Score whether used grid indices are declared in the palette."""

    art = _ensure_pixel_art(pixel_art)
    used = {index for row in art.pixels for index in row}
    declared = set(art.palette)
    return len(used & declared) / len(used) if used else 0.0


def non_empty_score(pixel_art: PixelArt | str, min_pixels: int = 40) -> float:
    """Reward sprites with enough non-transparent pixels."""

    art = _ensure_pixel_art(pixel_art)
    total = sum(1 for row in art.pixels for index in row if index != "0")
    return min(total / min_pixels, 1.0)


def symmetry_score(pixel_art: PixelArt | str) -> float:
    """Return left-right exact index symmetry over all mirrored cell pairs."""

    art = _ensure_pixel_art(pixel_art)
    matches = 0
    comparisons = GRID_SIZE * (GRID_SIZE // 2)
    for row in art.pixels:
        for col in range(GRID_SIZE // 2):
            if row[col] == row[GRID_SIZE - col - 1]:
                matches += 1
    return matches / comparisons


def connected_component_score(pixel_art: PixelArt | str) -> float:
    """Return largest non-transparent connected component ratio."""

    art = _ensure_pixel_art(pixel_art)
    filled = {
        (x, y)
        for y, row in enumerate(art.pixels)
        for x, index in enumerate(row)
        if index != "0"
    }
    if not filled:
        return 0.0

    seen: set[tuple[int, int]] = set()
    largest = 0
    for start in filled:
        if start in seen:
            continue
        size = _component_size(start, filled, seen)
        largest = max(largest, size)
    return largest / len(filled)


def _component_size(
    start: tuple[int, int], filled: set[tuple[int, int]], seen: set[tuple[int, int]]
) -> int:
    queue: deque[tuple[int, int]] = deque([start])
    seen.add(start)
    size = 0
    while queue:
        x, y = queue.popleft()
        size += 1
        for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if neighbor in filled and neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return size


def _ensure_pixel_art(pixel_art: PixelArt | str) -> PixelArt:
    return parse_dsl(pixel_art) if isinstance(pixel_art, str) else pixel_art
