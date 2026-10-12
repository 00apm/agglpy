"""agglpy.io.read_image: the grey array legacy used, from any path."""

import os
import shutil
import stat
from pathlib import Path

import cv2
import numpy as np
import pytest
import tifffile

from agglpy.errors import FileFormatError
from agglpy.io import read_image

from support.paths import DATA_DIR

INPUT = DATA_DIR / "input"
REAL_IMAGES = {
    "D7-017": INPUT / "multiple_image" / "images" / "D7-017" / "D7-017.tif",
    "D7-019": INPUT / "multiple_image" / "images" / "D7-019" / "D7-019.tif",
    "D6-027": INPUT / "multiple_image_old" / "D6-027" / "D6-027.tif",
    "D5-015": INPUT / "single_image" / "D5-015.tif",
}


def _legacy_grey(path: Path) -> np.ndarray:
    """What v0.4 used: cv2.imread, then BGR -> grey."""
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)


@pytest.mark.parametrize("name", list(REAL_IMAGES))
def test_real_zeiss_images_equal_legacy(name: str) -> None:
    # Zeiss palette TIFFs: palette -> RGB -> BT.601 grey, as legacy
    path = REAL_IMAGES[name]
    image = read_image(path)
    assert image.dtype == np.uint8 and image.ndim == 2
    np.testing.assert_array_equal(image, _legacy_grey(path))


def test_shape_is_height_by_width() -> None:
    assert read_image(REAL_IMAGES["D7-017"]).shape == (2304, 3072)


def test_path_with_characters_outside_the_code_page(tmp_path: Path) -> None:
    # cv2.imread fails on such paths on Windows; the bytes route doesn't
    folder = tmp_path / "próbka żółw µm 試料"
    folder.mkdir()
    path = shutil.copy(REAL_IMAGES["D5-015"], folder / "D5-015 ä.tif")
    expected = _legacy_grey(REAL_IMAGES["D5-015"])
    np.testing.assert_array_equal(read_image(path), expected)


def test_source_file_is_opened_read_only_and_unchanged(
    tmp_path: Path,
) -> None:
    path = shutil.copy(REAL_IMAGES["D5-015"], tmp_path / "D5-015.tif")
    before = path.read_bytes()
    os.chmod(path, stat.S_IREAD)  # a write attempt would raise
    try:
        read_image(path)
    finally:
        os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
    assert path.read_bytes() == before


def test_palette_becomes_bt601_grey(tmp_path: Path) -> None:
    colormap = np.zeros((3, 256), np.uint16)
    colormap[:, 1] = (65535, 0, 0)  # entry 1: pure red
    colormap[:, 2] = (65535, 65535, 65535)  # entry 2: white
    data = np.array([[0, 1, 2]], np.uint8)
    path = tmp_path / "palette.tif"
    tifffile.imwrite(path, data, photometric="palette", colormap=colormap)
    # BT.601: 0.299 * 255 = 76.2 for red
    assert read_image(path).tolist() == [[0, 76, 255]]


def test_colour_becomes_grey(tmp_path: Path) -> None:
    rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], np.uint8)
    path = tmp_path / "rgb.tif"
    tifffile.imwrite(path, rgb, photometric="rgb")
    # 0.299, 0.587, 0.114 of 255
    assert read_image(path).tolist() == [[76, 150, 29]]


def test_16_bit_keeps_the_top_byte(tmp_path: Path) -> None:
    path = tmp_path / "deep.tif"
    tifffile.imwrite(path, np.array([[0x1234, 0xFF00, 0x00FF]], np.uint16))
    assert read_image(path).tolist() == [[0x12, 0xFF, 0x00]]


def test_alpha_is_dropped(tmp_path: Path) -> None:
    rgba = np.zeros((1, 2, 4), np.uint8)
    rgba[0, 0] = (200, 200, 200, 0)  # BGRA, fully transparent
    rgba[0, 1] = (10, 10, 10, 255)
    path = tmp_path / "alpha.png"
    assert cv2.imwrite(str(path), rgba)
    assert read_image(path).tolist() == [[200, 10]]


def test_multipage_tiff_gives_the_first_page(tmp_path: Path) -> None:
    path = tmp_path / "pages.tif"
    with tifffile.TiffWriter(path) as tif:
        tif.write(np.full((2, 3), 10, np.uint8))
        tif.write(np.full((2, 3), 200, np.uint8))
    assert read_image(path).tolist() == [[10] * 3] * 2


@pytest.mark.parametrize("suffix", [".png", ".jpg", ".bmp"])
def test_other_formats(tmp_path: Path, suffix: str) -> None:
    grey = np.full((4, 5), 128, np.uint8)
    path = tmp_path / f"image{suffix}"
    assert cv2.imwrite(str(path), grey)
    np.testing.assert_array_equal(read_image(path), grey)


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_image(tmp_path / "missing.tif")


@pytest.mark.parametrize(
    "content",
    [b"", b"not an image\n", b"II*\x00broken"],
    ids=["empty", "text", "bad tiff"],
)
def test_undecodable_file_raises_file_format_error(
    tmp_path: Path, content: bytes
) -> None:
    path = tmp_path / "bad.tif"
    path.write_bytes(content)
    with pytest.raises(FileFormatError, match=r"bad\.tif"):
        read_image(path)
