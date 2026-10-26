"""Synthetic grey images with bright discs, for detection tests."""

import cv2
import numpy as np

# cv2.circle's sub-pixel mode: coordinates and radius in 1/16 px
SHIFT = 4


def draw_discs(height: int, width: int, discs) -> np.ndarray:
    """Bright anti-aliased discs (x, y, r in px) on a dark image."""
    image = np.full((height, width), 20, dtype=np.uint8)
    for x, y, r in discs:
        cv2.circle(
            image,
            (round(x * 2**SHIFT), round(y * 2**SHIFT)),
            round(r * 2**SHIFT),
            200,
            thickness=-1,
            lineType=cv2.LINE_AA,
            shift=SHIFT,
        )
    return image
