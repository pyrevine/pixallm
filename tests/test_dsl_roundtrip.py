from PIL import Image, ImageDraw

import pytest

from pixellm.data.prepare import image_to_dsl, prepare_image
from pixellm.dsl import DSLParseError, PixelArt, parse_dsl, serialize_dsl
from pixellm.render import render_pixel_art


def test_dsl_parse_serialize_roundtrip() -> None:
    art = PixelArt(
        palette={"0": "transparent", "1": "#112233", "2": "#abcdef"},
        pixels=[
            "0000000000000000",
            "0000000110000000",
            "0000001221000000",
            "0000012222100000",
            "0000012222100000",
            "0000001221000000",
            "0000001111000000",
            "0000000110000000",
            "0000000110000000",
            "0000001111000000",
            "0000012222100000",
            "0000012222100000",
            "0000001221000000",
            "0000000110000000",
            "0000000000000000",
            "0000000000000000",
        ],
    )

    serialized = serialize_dsl(art)
    parsed = parse_dsl(serialized)

    assert parsed == art
    assert serialized.startswith("<PALETTE>")
    assert "<GRID>" in serialized


def test_parse_rejects_indices_outside_zero_to_seven() -> None:
    dsl = """<PALETTE>
0:transparent, 1:#112233
</PALETTE>
<GRID>
8888888888888888
0000000000000000
0000000000000000
0000000000000000
0000000000000000
0000000000000000
0000000000000000
0000000000000000
0000000000000000
0000000000000000
0000000000000000
0000000000000000
0000000000000000
0000000000000000
0000000000000000
0000000000000000
</GRID>"""

    with pytest.raises(DSLParseError):
        parse_dsl(dsl)


def test_image_to_dsl_roundtrip_matches_canonical_image() -> None:
    source = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    draw = ImageDraw.Draw(source)
    draw.rectangle((5, 2, 10, 13), fill=(40, 24, 16, 255))
    draw.rectangle((6, 4, 9, 9), fill=(190, 120, 40, 255))
    draw.point((6, 5), fill=(255, 255, 255, 255))
    draw.point((9, 5), fill=(255, 255, 255, 255))

    prepared = prepare_image(source)
    parsed = parse_dsl(image_to_dsl(source))
    rendered = render_pixel_art(parsed, scale=1)

    assert rendered.size == (16, 16)
    assert _image_data(rendered) == _image_data(prepared.canonical_image)
    assert set(parsed.palette).issubset(set("01234567"))


def _image_data(image: Image.Image) -> list[object]:
    get_flattened_data = getattr(image, "get_flattened_data", None)
    if get_flattened_data is not None:
        return list(get_flattened_data())
    return list(image.getdata())
