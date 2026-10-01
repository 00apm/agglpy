"""Transformations that must not change the result of a case.

Only exact operations: reordering, shifting by whole pixels, mirroring
and swapping axes keep every distance bit-for-bit, so exact tangencies
stay exact. Rotation would not, so it is left out.
"""

import random
from collections.abc import Callable
from dataclasses import replace

from .cases import Case, Circle

Transform = Callable[[Case], Case]


def shuffled(case: Case, seed: int = 1) -> Case:
    """Same circles in a different input order."""
    circles = list(case.circles)
    random.Random(seed).shuffle(circles)
    return replace(case, circles=tuple(circles))


def shifted(case: Case, dx: float = -1000, dy: float = -2000) -> Case:
    """Moved by whole pixels; the default puts every centre at negative
    coordinates, i.e. outside the image."""
    return replace(
        case,
        circles=tuple(
            Circle(c.key, c.x + dx, c.y + dy, c.r) for c in case.circles
        ),
    )


def mirrored(case: Case) -> Case:
    """Mirrored across the y axis (x -> -x)."""
    return replace(
        case,
        circles=tuple(Circle(c.key, -c.x, c.y, c.r) for c in case.circles),
    )


def swapped_axes(case: Case) -> Case:
    """x and y exchanged."""
    return replace(
        case,
        circles=tuple(Circle(c.key, c.y, c.x, c.r) for c in case.circles),
    )


TRANSFORMS: dict[str, Transform] = {
    "shuffled": shuffled,
    "shifted": shifted,
    "mirrored": mirrored,
    "swapped_axes": swapped_axes,
}
