# Palette Index Grid DSL

`pixallm` uses a compact text DSL for 16x16 pixel art. The model is trained to emit this DSL directly, and the rest of the pipeline parses, renders, and scores it.

## Format

```text
<PALETTE>
0:transparent, 1:#f6d9c5, 2:#d59c85, 3:#1b0809
</PALETTE>
<GRID>
0000000000000000
0000000330000000
0000013332200000
...
</GRID>
```

## Constraints

- Size is fixed at 16x16.
- Palette indices are single characters from `0` to `7`.
- `0` is always `transparent`.
- Non-transparent colors use `1-7`.
- Grid rows must be exactly 16 characters.
- There must be exactly 16 grid rows.
- Any non-zero index used in the grid must be declared in the palette.
- Invalid indices such as `8`, `9`, or letters make the sample invalid.

## Why This DSL

Palette Index Grid is deliberately smaller than JSON and easier to validate than free-form art instructions.

It preserves spatial structure in the token stream. Repeated colors, symmetry, and sprite silhouettes appear as repeated index patterns, which is useful for a small language model.

It also gives deterministic rewards:

- Syntax validity
- Palette/index consistency
- Non-empty output
- Left-right symmetry for front-view sprites
- Connected component quality

## Alternatives Considered

JSON was rejected as the external training target because it adds tokens that do not help the visual structure: braces, quotes, field names, commas, and nested lists.

Raw RGB grids were rejected because six-character colors per pixel are too expensive for a 16x16 target.

ASCII art without a palette was rejected because it cannot preserve consistent colors across examples.

Run-length encoding was deferred. It is token-efficient, but it hides the row-wise spatial pattern that the model should learn first.

## Internal Representation

The external target is always the tag DSL. Internally, the parser converts DSL into a Pydantic `PixelArt` object:

```python
PixelArt(
    size=16,
    palette={"0": "transparent", "1": "#f6d9c5"},
    pixels=["0000000000000000", ...],
)
```

The public helpers are:

- `parse_dsl(text) -> PixelArt`
- `serialize_dsl(pixel_art) -> str`
- `validate_pixel_art(pixel_art) -> None`
- `render_pixel_art(pixel_art, scale=20) -> Image`

## Data Conversion

For `m1guelpf/nouns`, the preparation pipeline removes flat border-connected backgrounds before quantization. This keeps the first SFT dataset aligned with the project goal: transparent front-view sprites rather than full tiles with background color.

The conversion flow is:

1. Convert image to RGBA.
2. Resize to 16x16 with nearest-neighbor sampling.
3. Remove border-connected background pixels for nouns samples.
4. Map transparent pixels to `0`.
5. Quantize non-transparent pixels to at most seven colors.
6. Serialize to Palette Index Grid DSL.
