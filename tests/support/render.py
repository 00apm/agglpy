"""Render a matplotlib figure the way a user saves it, for pixel checks."""

import io

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from PIL import Image


def render(fig: Figure) -> NDArray[np.uint8]:
    """The figure saved as PNG and read back: (height, width, 4) RGBA."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("RGBA"))


def rgba(colour: tuple[float, float, float, float]) -> NDArray[np.uint8]:
    """A matplotlib RGBA colour (0-1) as the PNG stores it (0-255)."""
    return np.round(np.asarray(colour) * 255).astype(np.uint8)
