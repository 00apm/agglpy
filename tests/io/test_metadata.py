"""agglpy.io.read_image_metadata: size and pixel size from the tags.

Real Zeiss files from tests/data, and tiny synthetic TIFFs carrying
only the tag entries a reader uses, with values copied from the real
files of item 2.2 Findings (support.tiffs).
"""

import dataclasses
import os
import struct
import warnings
from pathlib import Path

import cv2
import numpy as np
import pytest
import tifffile

from agglpy.errors import FileFormatError, PixelSizeWarning
from agglpy.io import ImageMetadata, read_image_metadata

from support.paths import DATA_DIR
from support.tiffs import write_fei, write_imagej, write_zeiss

INPUT = DATA_DIR / "input"
D7_017 = INPUT / "multiple_image" / "images" / "D7-017" / "D7-017.tif"
D6_027 = INPUT / "multiple_image_old" / "D6-027" / "D6-027.tif"
ZEISS = "Zeiss CZ_SEM (1024-px reference scaled to {} px)"


@pytest.fixture(autouse=True)
def _no_unexpected_warnings():
    # every expected warning is caught with pytest.warns
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


def test_metadata_is_a_frozen_dataclass() -> None:
    meta = read_image_metadata(D7_017)
    assert isinstance(meta, ImageMetadata)
    with pytest.raises(dataclasses.FrozenInstanceError):
        meta.pixel_size = 1.0  # type: ignore[misc]


def test_real_zeiss_3072_px() -> None:
    # raw 1.171875e-07 m at 1024 px; text field 39.09 nm is 0.07 % off:
    # no warning
    assert read_image_metadata(D7_017) == ImageMetadata(
        pixel_size=pytest.approx(39.0625e-9, rel=1e-12),
        pixel_size_source=ZEISS.format(3072),
        width=3072,
        height=2304,
    )


def test_real_zeiss_2048_px() -> None:
    meta = read_image_metadata(D6_027)
    assert meta.pixel_size == pytest.approx(58.6404e-9, rel=1e-12)
    assert (meta.width, meta.height) == (2048, 1536)


@pytest.mark.parametrize(
    ("width", "raw", "text", "expected"),
    [
        # D7-019 (3072 px), D6-027 (2048 px), the 512-px sample (no
        # text field; legacy took ap_pixel_size, 2x too small)
        (3072, 1.172808e-07, (39.09, "nm"), 39.0936e-9),
        (2048, 1.172808e-07, (58.64, "nm"), 58.6404e-9),
        (512, 5.824988e-09, None, 11.649976e-9),
        (1024, 2.938034e-06, (2.938, "\N{MICRO SIGN}m"), 2.938034e-6),
    ],
)
def test_zeiss_raw_value_is_scaled_to_the_width(
    tmp_path: Path, width: int, raw: float, text, expected: float
) -> None:
    path = write_zeiss(tmp_path / "z.tif", width, 8, raw, text)
    meta = read_image_metadata(path)
    assert meta.pixel_size == pytest.approx(expected, rel=1e-12)
    assert meta.pixel_size_source == ZEISS.format(width)


@pytest.mark.parametrize(
    "unit_text",
    [
        (0.0395, "\N{MICRO SIGN}m"),  # micro sign, as Zeiss writes it
        (0.0395, "\N{GREEK SMALL LETTER MU}m"),  # Greek mu (UTF-8 bytes)
        (3.95e-5, "mm"),
        (3.95e-8, "m"),
    ],
)
def test_zeiss_text_field_more_than_1_percent_off_warns(
    tmp_path: Path, unit_text: tuple[float, str]
) -> None:
    # raw 39.0625 nm at 3072 px vs text 39.5 nm: 1.1 % apart
    encoding = (
        "utf-8" if unit_text[1] == "\N{GREEK SMALL LETTER MU}m" else "cp1252"
    )
    path = write_zeiss(
        tmp_path / "z.tif", 3072, 8, 1.171875e-07, unit_text, encoding=encoding
    )
    with pytest.warns(PixelSizeWarning, match=r"z\.tif.*1 %"):
        meta = read_image_metadata(path)
    assert meta.pixel_size == pytest.approx(39.0625e-9, rel=1e-12)


def test_zeiss_text_field_within_1_percent_is_silent(tmp_path: Path) -> None:
    # 39.4 nm is 0.86 % above 39.0625 nm
    path = write_zeiss(tmp_path / "z.tif", 3072, 8, 1.171875e-07, (39.4, "nm"))
    read_image_metadata(path)  # the fixture turns a warning into an error


@pytest.mark.parametrize("unit", ["pm", "furlong", ""])
def test_zeiss_unknown_unit_skips_the_check(tmp_path: Path, unit: str) -> None:
    path = write_zeiss(tmp_path / "z.tif", 1024, 8, 1e-6, (5.0, unit))
    assert read_image_metadata(path).pixel_size == pytest.approx(1e-6)


def test_zeiss_multipage_reads_the_first_page(tmp_path: Path) -> None:
    # RosettaSciIO multipage sample: 2048 px, raw 2.465245e-08 m
    path = write_zeiss(
        tmp_path / "z.tif", 2048, 8, 2.465245e-08, (12.33, "nm"), pages=2
    )
    meta = read_image_metadata(path)
    assert meta.pixel_size == pytest.approx(12.326225e-9, rel=1e-12)


def test_png_has_a_size_and_no_pixel_size(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    assert cv2.imwrite(str(path), np.zeros((8, 16), np.uint8))
    assert read_image_metadata(path) == ImageMetadata(None, None, 16, 8)


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_image_metadata(tmp_path / "missing.tif")


@pytest.mark.parametrize(
    "content",
    [
        b"II*\x00",
        b"II*\x00\xff\xff\xff\x7f",
        b"MM\x00*\x00\x00\x00\x08garbage",
        b"II*\x00\x08\x00\x00\x00\x00\x00",
    ],
    ids=["truncated", "offset past the end", "garbage tags", "no tags"],
)
def test_corrupt_tiff_raises_file_format_error(
    tmp_path: Path, content: bytes
) -> None:
    path = tmp_path / "corrupt.tif"
    path.write_bytes(content)
    with pytest.raises(FileFormatError, match=r"corrupt\.tif"):
        read_image_metadata(path)


def test_undecodable_other_file_raises_file_format_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "notes.png"
    path.write_text("not an image")
    with pytest.raises(FileFormatError, match=r"notes\.png"):
        read_image_metadata(path)


def test_thermo_fei_pixel_width(tmp_path: Path) -> None:
    # FEI Helios sample: 512 x 471 px, PixelWidth 3.3724e-06 m
    path = write_fei(tmp_path / "f.tif", 512, 471, 3.3724e-06, 3.3724e-06)
    assert read_image_metadata(path) == ImageMetadata(
        3.3724e-06, "Thermo/FEI [Scan] PixelWidth", 512, 471
    )


def test_thermo_fei_non_square_pixels_warn(tmp_path: Path) -> None:
    path = write_fei(tmp_path / "f.tif", 16, 8, 3.3724e-06, 3.5e-06)
    with pytest.warns(PixelSizeWarning, match=r"f\.tif.*not square"):
        meta = read_image_metadata(path)
    assert meta.pixel_size == 3.3724e-06  # the x value


@pytest.mark.parametrize(
    ("unit", "factor"),
    [
        ("micron", 1e-6),
        ("um", 1e-6),
        ("\N{MICRO SIGN}m", 1e-6),
        ("\N{GREEK SMALL LETTER MU}m", 1e-6),
        ("nm", 1e-9),
        ("mm", 1e-3),
        ("cm", 1e-2),
        ("m", 1.0),
    ],
)
def test_imagej_units(tmp_path: Path, unit: str, factor: float) -> None:
    # ImageJ 1.51a sample: 5.928571 px per micron -> 1.6867e-07 m
    path = write_imagej(tmp_path / "ij.tif", 16, 8, unit, (5.928571, 5.928571))
    meta = read_image_metadata(path)
    assert meta.pixel_size == pytest.approx(factor / 5.928571, rel=1e-6)
    assert meta.pixel_size_source == f"ImageJ calibration (unit={unit})"


@pytest.mark.parametrize("unit", [None, "pixel", "inch"])
def test_imagej_without_a_known_unit_has_no_pixel_size(
    tmp_path: Path, unit: str | None
) -> None:
    path = write_imagej(tmp_path / "ij.tif", 16, 8, unit, (5.0, 5.0))
    assert read_image_metadata(path).pixel_size is None


def test_imagej_non_square_pixels_warn(tmp_path: Path) -> None:
    path = write_imagej(tmp_path / "ij.tif", 16, 8, "micron", (5.0, 6.0))
    with pytest.warns(PixelSizeWarning, match="not square"):
        meta = read_image_metadata(path)
    assert meta.pixel_size == pytest.approx(0.2e-6)


def test_first_reader_wins(tmp_path: Path) -> None:
    # Zeiss before Thermo/FEI before ImageJ (D-050)
    path = tmp_path / "both.tif"
    zeiss = b"0\r\n0\r\n0\r\n1e-06\r\n"
    fei = b"[Scan]\r\nPixelWidth=2e-06\r\n"
    tifffile.imwrite(
        path,
        np.zeros((8, 1024), np.uint8),
        extratags=[
            (34118, 2, len(zeiss) + 1, zeiss, True),
            (34682, 2, len(fei) + 1, fei, True),
        ],
    )
    assert read_image_metadata(path).pixel_size == pytest.approx(1e-6)


def test_tiff_resolution_alone_is_never_used(tmp_path: Path) -> None:
    # 72 dpi in many files, 1/1 in Zeiss files: not a pixel size
    path = tmp_path / "plain.tif"
    tifffile.imwrite(
        path,
        np.zeros((8, 16), np.uint8),
        resolution=(72, 72),
        resolutionunit="INCH",
    )
    assert read_image_metadata(path) == ImageMetadata(None, None, 16, 8)


@pytest.mark.parametrize("kind", ["zeiss", "non-square"])
def test_pixel_size_warning_points_at_the_caller(
    tmp_path: Path, kind: str
) -> None:
    if kind == "zeiss":
        path = write_zeiss(
            tmp_path / "z.tif", 3072, 8, 1.171875e-07, (50, "nm")
        )
    else:
        path = write_fei(tmp_path / "f.tif", 16, 8, 3.3724e-06, 3.5e-06)
    with pytest.warns(PixelSizeWarning) as record:
        read_image_metadata(path)
    assert os.path.normcase(record[0].filename) == os.path.normcase(__file__)


def test_tiff_tag_of_the_wrong_type_raises_file_format_error(
    tmp_path: Path,
) -> None:
    # SamplesPerPixel stored as text instead of a number: tifffile
    # raises a TypeError, which must not escape as itself
    path = tmp_path / "corrupt.tif"
    tifffile.imwrite(path, np.zeros((4, 8), np.uint8))
    data = bytearray(path.read_bytes())
    ifd = struct.unpack_from("<I", data, 4)[0]
    for i in range(struct.unpack_from("<H", data, ifd)[0]):
        entry = ifd + 2 + 12 * i
        if struct.unpack_from("<H", data, entry)[0] == 277:
            struct.pack_into("<H", data, entry + 2, 2)  # type ASCII
    path.write_bytes(bytes(data))
    with pytest.raises(FileFormatError, match=r"corrupt\.tif"):
        read_image_metadata(path)
