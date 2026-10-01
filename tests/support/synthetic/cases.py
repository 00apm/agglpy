"""Hand-made circle sets with expected results worked out by hand.

All coordinates and radii are in pixels. Contacts lie on an axis or on
Pythagorean-triple offsets (3-4-5 scaled), so every distance in a case
is exact in floating point and an "exactly touching" pair really is
exactly touching.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Circle:
    """One primary particle, identified by a key that is unique per case."""

    key: str
    x: float
    y: float
    r: float


@dataclass(frozen=True)
class Case:
    """A circle set and the result any correct implementation must give.

    Attributes:
        name: Short id, used as the pytest id.
        circles: The particles of the case.
        groups: Expected agglomerates, as a partition of the circle keys.
        idj: Keys of the particles expected to be internally disjoint
            (lying completely inside another particle).
        note: What the case checks, in words.
    """

    name: str
    circles: tuple[Circle, ...]
    groups: frozenset[frozenset[str]]
    idj: frozenset[str] = field(default_factory=frozenset)
    note: str = ""


def make_case(
    name: str,
    circles: Iterable[tuple[str, float, float, float]],
    groups: Iterable[Iterable[str]],
    idj: Iterable[str] = (),
    note: str = "",
) -> Case:
    """Build a Case from plain tuples ``(key, x, y, r)`` and key groups."""
    circle_objs = tuple(Circle(*c) for c in circles)
    group_sets = frozenset(frozenset(g) for g in groups)
    keys = [c.key for c in circle_objs]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{name}: circle keys are not unique")
    grouped = [k for g in group_sets for k in g]
    if sorted(grouped) != sorted(keys):
        raise ValueError(f"{name}: groups are not a partition of the keys")
    return Case(name, circle_objs, group_sets, frozenset(idj), note)


SINGLE = make_case(
    "single",
    [("a", 0, 0, 10)],
    groups=[["a"]],
    note="One isolated primary particle is its own agglomerate.",
)


def _external_doublets(r1: float, r2: float, label: str) -> list[Case]:
    """Two circles side by side on the x axis, around the contact distance.

    The contact condition is ``dist <= r1 + r2``.
    """
    touch = r1 + r2
    return [
        make_case(
            f"doublet_{label}_gap_1px",
            [("a", 0, 0, r1), ("b", touch + 1, 0, r2)],
            groups=[["a"], ["b"]],
            note="1 px apart: no contact.",
        ),
        make_case(
            f"doublet_{label}_tangent",
            [("a", 0, 0, r1), ("b", touch, 0, r2)],
            groups=[["a", "b"]],
            note="dist == r1 + r2: touching counts as contact.",
        ),
        make_case(
            f"doublet_{label}_overlap_1px",
            [("a", 0, 0, r1), ("b", touch - 1, 0, r2)],
            groups=[["a", "b"]],
            note="1 px overlap: contact.",
        ),
    ]


# R = 10, r = 4: internal tangency at dist = R - r = 6
_R, _r = 10, 4

DOUBLETS: list[Case] = [
    *_external_doublets(10, 10, "equal"),
    *_external_doublets(10, 4, "unequal"),
    make_case(
        "doublet_tangent_diagonal",
        # dist = sqrt(15**2 + 20**2) = 25 = 10 + 15 exactly
        [("a", 0, 0, 10), ("b", 15, 20, 15)],
        groups=[["a", "b"]],
        note="Tangent along a 3-4-5 diagonal: contact.",
    ),
    make_case(
        "doublet_concentric",
        [("a", 0, 0, _R), ("b", 0, 0, _r)],
        groups=[["a", "b"]],
        idj=["b"],
        note="Smaller circle with the same centre: inside, idj.",
    ),
    make_case(
        "doublet_inside",
        [("a", 0, 0, _R), ("b", 2, 0, _r)],
        groups=[["a", "b"]],
        idj=["b"],
        note="dist = 2 < R - r: completely inside, idj.",
    ),
    make_case(
        "doublet_inside_1px_from_tangent",
        [("a", 0, 0, _R), ("b", _R - _r - 1, 0, _r)],
        groups=[["a", "b"]],
        idj=["b"],
        note="dist = R - r - 1: inside, idj.",
    ),
    make_case(
        "doublet_internally_tangent",
        [("a", 0, 0, _R), ("b", _R - _r, 0, _r)],
        groups=[["a", "b"]],
        idj=["b"],
        note="dist == R - r: touches the edge from inside, still idj.",
    ),
    make_case(
        "doublet_crossing_edge",
        [("a", 0, 0, _R), ("b", _R - _r + 1, 0, _r)],
        groups=[["a", "b"]],
        note=(
            "dist = R - r + 1: centre inside the big circle, but the small "
            "one crosses its edge. Grouped, not idj."
        ),
    ),
]
