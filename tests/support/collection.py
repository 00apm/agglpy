"""Collections with synthetic disc images, for the application tests."""

from pathlib import Path

import numpy as np
import tifffile

from agglpy.app import Collection

from support.discs import draw_discs

# Two discs per image, at different places, so the files differ
DISCS = {
    "D7-017": [(60.3, 50.6, 14.2), (150.4, 110.7, 14.2)],
    "D7-018": [(50.2, 60.4, 14.2), (140.6, 100.3, 14.2)],
    "D7-019": [(70.5, 40.2, 14.2), (130.1, 120.8, 14.2)],
}
HCT = {"ranges": [{"d_min": 12, "d_max": 44}]}


def write_disc_image(path: Path, discs) -> Path:
    """A 200 x 160 px TIFF with bright discs (x, y, r in px)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, draw_discs(160, 200, discs))
    return path


def write_blank_image(path: Path) -> Path:
    """A 200 x 160 px TIFF with nothing on it."""
    tifffile.imwrite(path, np.full((160, 200), 20, np.uint8))
    return path


def work(col: Collection, name: str) -> Path:
    """The work folder of one image."""
    return col.folder / "agglpy" / name
