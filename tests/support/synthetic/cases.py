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


STRUCTURES: list[Case] = [
    make_case(
        "triangle_tangent_plus_isolated",
        # right triangle with sides 30-40-50; radii 10, 20, 30 make every
        # pair exactly tangent: 10+20=30, 10+30=40, 20+30=50
        [
            ("a", 0, 0, 10),
            ("b", 30, 0, 20),
            ("c", 0, 40, 30),
            ("d", 200, 0, 5),
        ],
        groups=[["a", "b", "c"], ["d"]],
        note="Three mutually tangent particles + one isolated: 2 groups.",
    ),
    make_case(
        "star",
        # arms touch the centre only: neighbouring arms are
        # 20 * sqrt(2) = 28.3 > 20 apart
        [
            ("c", 0, 0, 10),
            ("e", 20, 0, 10),
            ("n", 0, 20, 10),
            ("w", -20, 0, 10),
            ("s", 0, -20, 10),
        ],
        groups=[["c", "e", "n", "w", "s"]],
        note="Arms connected only through the centre: one group.",
    ),
    make_case(
        "chain_equal_plus_neighbour_1px",
        [
            ("a", 0, 0, 10),
            ("b", 20, 0, 10),
            ("c", 40, 0, 10),
            ("d", 60, 0, 10),
            ("e", 80, 0, 10),
            ("f", 96, 0, 5),  # 1 px from e (touching at 95)
        ],
        groups=[["a", "b", "c", "d", "e"], ["f"]],
        note=(
            "Chain of 5: the ends don't touch but share one group "
            "(indirect contact). A particle 1 px from the end stays out."
        ),
    ),
    make_case(
        "chain_largest_at_end",
        [
            ("a", 0, 0, 20),
            ("b", 30, 0, 10),
            ("c", 50, 0, 10),
            ("d", 70, 0, 10),
            ("e", 90, 0, 10),
        ],
        groups=[["a", "b", "c", "d", "e"]],
        note=(
            "The legacy search starts from the largest particle; with it "
            "at one end the whole chain must still be found."
        ),
    ),
    make_case(
        "ring_square_plus_centre",
        # 8 particles on the outline of a 40 x 40 square, neighbours 20
        # apart (tangent); next-but-one neighbours are >= 28.3 apart.
        # The centre particle is 20 from the nearest ring centre, needs 15.
        [
            ("r1", 0, 0, 10),
            ("r2", 20, 0, 10),
            ("r3", 40, 0, 10),
            ("r4", 40, 20, 10),
            ("r5", 40, 40, 10),
            ("r6", 20, 40, 10),
            ("r7", 0, 40, 10),
            ("r8", 0, 20, 10),
            ("m", 20, 20, 5),
        ],
        groups=[["r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8"], ["m"]],
        note=(
            "Closed ring: the search must stop when it gets back to the "
            "first particle. The centre particle touches nothing: grouping "
            "is by contact, not by being surrounded."
        ),
    ),
    make_case(
        "polydisperse_collector",
        # collector R = 1000, small particles r = 0.5 (D ratio 2000)
        [
            ("big", 0, 0, 1000),
            ("tangent", 1000.5, 0, 0.5),
            ("gap_1px", -1001.5, 0, 0.5),
            ("inside", 500, 0, 0.5),
            ("on_edge_diag", -600, -800, 0.5),  # centre on the edge
            ("tangent_top", 0, 1000.5, 0.5),
            ("via_small", 0, 1001.5, 0.5),  # touches tangent_top only
        ],
        groups=[
            [
                "big",
                "tangent",
                "inside",
                "on_edge_diag",
                "tangent_top",
                "via_small",
            ],
            ["gap_1px"],
        ],
        idj=["inside"],
        note=(
            "Large collector with small particles: tangent, overlapping, "
            "inside (idj), 1 px away (separate), and one connected only "
            "through another small particle."
        ),
    ),
]


def long_chain(n: int) -> Case:
    """``n`` equal particles in a straight line, each touching the next."""
    keys = [f"p{i:05d}" for i in range(n)]
    return make_case(
        f"chain_{n}",
        [(k, 20 * i, 0, 10) for i, k in enumerate(keys)],
        groups=[keys],
        note=(
            f"A chain of {n} touching particles is one agglomerate, however "
            "long it is."
        ),
    )
