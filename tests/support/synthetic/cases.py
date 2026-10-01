"""Hand-made circle sets with expected results worked out by hand.

All coordinates and radii are in pixels. Contacts lie on an axis or on
Pythagorean-triple offsets (3-4-5 scaled), so every distance in a case
is exact in floating point and an "exactly touching" pair really is
exactly touching.
"""

import math
from collections.abc import Iterable, Mapping
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
        threshold: ``collector_threshold`` for classification.
        types: Expected particle type per key; empty if the case doesn't
            check classification.
        properties: Expected agglomerate properties (px, px³), keyed by
            the agglomerate's members; empty if not checked.
        summary: Expected per-image summary metrics (diameters in px);
            empty if not checked.
        note: What the case checks, in words.
    """

    name: str
    circles: tuple[Circle, ...]
    groups: frozenset[frozenset[str]]
    idj: frozenset[str] = field(default_factory=frozenset)
    threshold: float = 0.0
    types: Mapping[str, str] = field(default_factory=dict)
    properties: Mapping[frozenset[str], Mapping[str, float]] = field(
        default_factory=dict
    )
    summary: Mapping[str, float] = field(default_factory=dict)
    note: str = ""

    def agglomerate_types(self) -> dict[frozenset[str], str]:
        """Expected agglomerate type per group, implied by ``types``.

        A group with a collector particle is a collector agglomerate;
        otherwise all its members share the agglomerate's type
        (``similar`` or ``separate``).
        """
        result = {}
        for group in self.groups:
            member_types = {self.types[k] for k in group}
            if "collector" in member_types:
                result[group] = "collector"
            else:
                (result[group],) = member_types
        return result


def make_case(
    name: str,
    circles: Iterable[tuple[str, float, float, float]],
    groups: Iterable[Iterable[str]],
    idj: Iterable[str] = (),
    threshold: float = 0.0,
    types: Mapping[str, str] | None = None,
    properties: Mapping[Iterable[str], Mapping[str, float]] | None = None,
    summary: Mapping[str, float] | None = None,
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
    types = dict(types or {})
    if types and sorted(types) != sorted(keys):
        raise ValueError(f"{name}: types must be given for every key")
    props = {frozenset(k): dict(v) for k, v in (properties or {}).items()}
    if not set(props) <= group_sets:
        raise ValueError(f"{name}: properties given for a non-group")
    return Case(
        name,
        circle_objs,
        group_sets,
        frozenset(idj),
        threshold,
        types,
        props,
        dict(summary or {}),
        note,
    )


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


def _pair(
    name: str,
    r_big: float,
    r_small: float,
    threshold: float,
    types: tuple[str, str],
    note: str,
) -> Case:
    """Two overlapping particles on the x axis, the big one first."""
    return make_case(
        name,
        [("big", 0, 0, r_big), ("small", r_big + r_small - 1, 0, r_small)],
        groups=[["big", "small"]],
        threshold=threshold,
        types={"big": types[0], "small": types[1]},
        note=note,
    )


COLL = ("collector", "attached2coll")
SIM = ("similar", "similar")

# The rule: an agglomerate is a collector when D(2nd largest) / D(largest)
# <= threshold. Then the largest particle is the collector and all
# others are attached2coll; otherwise all are similar. A single particle
# is separate.
CLASSIFICATION: list[Case] = [
    make_case(
        "single_is_separate",
        [("a", 0, 0, 10)],
        groups=[["a"]],
        threshold=0.5,
        types={"a": "separate"},
        note="A particle alone is separate, whatever the threshold.",
    ),
    _pair("ratio_below", 10, 4, 0.5, COLL, "0.4 < 0.5: collector."),
    _pair("ratio_at_0.5", 10, 5, 0.5, COLL, "0.5 == 0.5: collector (<=)."),
    _pair("ratio_at_0.7", 10, 7, 0.7, COLL, "0.7 == 0.7: collector (<=)."),
    _pair(
        "ratio_at_0.8",
        10,
        8,
        0.8,
        COLL,
        "0.8 == 0.8: collector (<=); 0.8 is the golden-test threshold.",
    ),
    _pair("ratio_above", 10, 6, 0.5, SIM, "0.6 > 0.5: similar."),
    _pair(
        "threshold_zero",
        10,
        4,
        0.0,
        SIM,
        "Threshold 0 (the default): every agglomerate is similar.",
    ),
    _pair("equal_sizes", 10, 10, 0.8, SIM, "Ratio 1: similar."),
    make_case(
        "three_second_largest_decides_similar",
        # c is tangent to a on the right, b on the left
        [("a", 0, 0, 10), ("b", -14, 0, 4), ("c", 19, 0, 9)],
        groups=[["a", "b", "c"]],
        threshold=0.5,
        types={"a": "similar", "b": "similar", "c": "similar"},
        note=(
            "Only the two largest particles decide: 9/10 > 0.5 makes all "
            "similar, although b is small."
        ),
    ),
    make_case(
        "three_collector",
        [("a", 0, 0, 10), ("b", 14, 0, 4), ("c", -13, 0, 3)],
        groups=[["a", "b", "c"]],
        threshold=0.5,
        types={"a": "collector", "b": "attached2coll", "c": "attached2coll"},
        note="4/10 <= 0.5: a is the collector, b and c are attached.",
    ),
]

_POLY = STRUCTURES[-1]
assert _POLY.name == "polydisperse_collector"
CLASSIFICATION.append(
    make_case(
        "polydisperse_collector",
        [(c.key, c.x, c.y, c.r) for c in _POLY.circles],
        groups=_POLY.groups,
        idj=_POLY.idj,
        threshold=0.5,
        types={
            "big": "collector",
            "tangent": "attached2coll",
            "inside": "attached2coll",
            "on_edge_diag": "attached2coll",
            "tangent_top": "attached2coll",
            "via_small": "attached2coll",
            "gap_1px": "separate",
        },
        note=(
            "Collector with small particles (D ratio 2000); the one 1 px "
            "away is separate."
        ),
    )
)


def sphere(r: float) -> float:
    """Volume of a sphere of radius ``r``."""
    return 4 / 3 * math.pi * r**3


def volume_equivalent_d(volume: float) -> float:
    """Diameter of the sphere with the given volume."""
    return (6 * volume / math.pi) ** (1 / 3)


# Agglomerate properties, by definition:
#   volume             sum of the member sphere volumes
#   D                  diameter of one sphere with that volume
#   members_D_mean/std mean / sample std (n - 1) of the member diameters;
#                      std is NaN for a single member
#   idj_count          members lying completely inside another member
#   *_dsom             "dark side of the moon": a member seen completely
#                      inside another one's outline is assumed to have a
#                      twin hidden on the far side, so each idj member
#                      counts twice: volume_dsom = volume + idj volume,
#                      members_count_dsom = members_count + idj_count
PROPERTIES: list[Case] = [
    make_case(
        "single",
        [("a", 0, 0, 10)],
        groups=[["a"]],
        properties={
            ("a",): {
                "members_count": 1,
                "volume": sphere(10),
                "D": 20,
                "members_D_mean": 20,
                "members_D_std": math.nan,
                "idj_count": 0,
                "volume_dsom": sphere(10),
                "D_dsom": 20,
                "members_count_dsom": 1,
            }
        },
        note="One particle: D is its own diameter, std undefined.",
    ),
    make_case(
        "two_equal",
        [("a", 0, 0, 10), ("b", 20, 0, 10)],
        groups=[["a", "b"]],
        properties={
            ("a", "b"): {
                "members_count": 2,
                "volume": 2 * sphere(10),
                "D": 20 * 2 ** (1 / 3),  # twice the volume
                "members_D_mean": 20,
                "members_D_std": 0,
                "idj_count": 0,
                "volume_dsom": 2 * sphere(10),
                "D_dsom": 20 * 2 ** (1 / 3),
                "members_count_dsom": 2,
            }
        },
        note="Two equal spheres: D = d * 2^(1/3), not 2d.",
    ),
    make_case(
        "two_unequal",
        [("a", 0, 0, 10), ("b", 14, 0, 5)],
        groups=[["a", "b"]],
        properties={
            ("a", "b"): {
                "members_count": 2,
                "volume": sphere(10) + sphere(5),
                "D": 2 * 1125 ** (1 / 3),  # r³ = 1000 + 125
                "members_D_mean": 15,
                "members_D_std": 10 / math.sqrt(2),  # std(20, 10), n - 1
                "idj_count": 0,
                "volume_dsom": sphere(10) + sphere(5),
                "D_dsom": 2 * 1125 ** (1 / 3),
                "members_count_dsom": 2,
            }
        },
        note="Overlap doesn't reduce the volume: spheres are summed.",
    ),
    make_case(
        "one_inside",
        [("a", 0, 0, 10), ("b", 2, 0, 4)],
        groups=[["a", "b"]],
        idj=["b"],
        properties={
            ("a", "b"): {
                "members_count": 2,
                "volume": sphere(10) + sphere(4),
                "D": 2 * 1064 ** (1 / 3),  # r³ = 1000 + 64
                "members_D_mean": 14,
                "members_D_std": 12 / math.sqrt(2),  # std(20, 8), n - 1
                "idj_count": 1,
                "volume_dsom": sphere(10) + 2 * sphere(4),
                "D_dsom": 2 * 1128 ** (1 / 3),  # r³ = 1000 + 2 * 64
                "members_count_dsom": 3,
            }
        },
        note="The inside particle counts once more in the dsom values.",
    ),
    make_case(
        "polydisperse_small_members_count",
        [
            ("big", 0, 0, 1000),
            ("t1", 1000.5, 0, 0.5),
            ("t2", 0, 1000.5, 0.5),
            ("in", 500, 0, 0.5),
        ],
        groups=[["big", "t1", "t2", "in"]],
        idj=["in"],
        properties={
            ("big", "t1", "t2", "in"): {
                "members_count": 4,
                # small spheres add 3.75e-10 of the volume: rtol must be
                # well below that for this case to mean anything
                "volume": sphere(1000) + 3 * sphere(0.5),
                "D": 2 * (1000**3 + 3 * 0.5**3) ** (1 / 3),
                "members_D_mean": (2000 + 3 * 1) / 4,
                "members_D_std": math.sqrt(
                    ((2000 - 500.75) ** 2 + 3 * (1 - 500.75) ** 2) / 3
                ),
                "idj_count": 1,
                "volume_dsom": sphere(1000) + 4 * sphere(0.5),
                "D_dsom": 2 * (1000**3 + 4 * 0.5**3) ** (1 / 3),
                "members_count_dsom": 5,
            }
        },
        note="Tiny members (D ratio 2000) still add to volume and D.",
    ),
]


# Per-image summary metrics (names as in the legacy summary table):
#   N_primary_particle   particles
#   N_aerosol_particle   groups, single particles included
#   N_pp1                groups of one particle
#   N_ppA                particles in groups of two or more
#   N_agl                groups of two or more (true agglomerates)
#   N_collector_agl, N_similar_agl, N_pp1_separate   groups by type
#   ER       1 - N_pp1_separate / N_primary_particle
#   Ra       N_agl / N_primary_particle
#   sep2agl  N_pp1_separate / (N_collector_agl + N_similar_agl)
#   n_ppA    N_ppA / N_agl (particles per agglomerate)
#   n_ppP    N_primary_particle / N_aerosol_particle
#   particle_D*  mean, sample std, 10/50/90 % quantiles (linear
#                interpolation between sorted values), Sauter mean
#                diameter sum(D³) / sum(D²)
#   agl_member_count_mean / _std   over all groups, singles included
# Division by zero follows numpy: x / 0 = inf, 0 / 0 = NaN.
SUMMARY: list[Case] = [
    make_case(
        "composed_scene",
        [
            # similar triangle (radii 10/20/30, all tangent): 40/60 > 0.5
            ("t1", 0, 0, 10),
            ("t2", 30, 0, 20),
            ("t3", 0, 40, 30),
            # collector pair: 8/20 <= 0.5
            ("c_big", 0, 1000, 10),
            ("c_small", 13, 1000, 4),
            # two single particles
            ("s1", 1000, 0, 10),
            ("s2", 2000, 0, 5),
        ],
        groups=[["t1", "t2", "t3"], ["c_big", "c_small"], ["s1"], ["s2"]],
        threshold=0.5,
        # D (px) = 20, 40, 60, 20, 8, 20, 10; sorted 8 10 20 20 20 40 60
        # sum D = 178, sum D² = 6564, sum D³ = 305512
        summary={
            "N_primary_particle": 7,
            "N_aerosol_particle": 4,
            "N_pp1": 2,
            "N_ppA": 5,
            "N_agl": 2,
            "N_collector_agl": 1,
            "N_similar_agl": 1,
            "N_pp1_separate": 2,
            "ER": 1 - 2 / 7,
            "Ra": 2 / 7,
            "sep2agl": 2 / 2,
            "n_ppA": 5 / 2,
            "n_ppP": 7 / 4,
            "particle_Dmean": 178 / 7,
            # sum (D - mean)² = 6564 - 178² / 7 = 14264 / 7; / (n - 1)
            "particle_Dstd": math.sqrt(14264 / 42),
            "particle_D10": 8 + 0.6 * (10 - 8),  # position 0.1 * 6
            "particle_D50": 20,  # position 3
            "particle_D90": 40 + 0.4 * (60 - 40),  # position 5.4
            "particle_SMD": 305512 / 6564,
            "agl_member_count_mean": 7 / 4,  # counts 3, 2, 1, 1
            # sum (n - 1.75)² = 1.5625 + 0.0625 + 2 * 0.5625 = 2.75; / 3
            "agl_member_count_std": math.sqrt(11 / 12),
        },
        note="Similar triangle, collector pair, two singles; all by hand.",
    ),
    make_case(
        "singles_only",
        [("a", 0, 0, 10), ("b", 100, 0, 5), ("c", 200, 0, 8)],
        groups=[["a"], ["b"], ["c"]],
        threshold=0.5,
        # D = 20, 10, 16; sorted 10 16 20
        summary={
            "N_primary_particle": 3,
            "N_aerosol_particle": 3,
            "N_pp1": 3,
            "N_ppA": 0,
            "N_agl": 0,
            "N_collector_agl": 0,
            "N_similar_agl": 0,
            "N_pp1_separate": 3,
            "ER": 0,
            "Ra": 0,
            "sep2agl": math.inf,  # 3 separate / 0 agglomerates
            "n_ppA": math.nan,  # 0 particles / 0 agglomerates
            "n_ppP": 1,
            "particle_Dmean": 46 / 3,
            # sum (D - mean)² = 756 - 46² / 3 = 152 / 3; / (n - 1)
            "particle_Dstd": math.sqrt(76 / 3),
            "particle_D10": 10 + 0.2 * (16 - 10),  # position 0.2
            "particle_D50": 16,
            "particle_D90": 16 + 0.8 * (20 - 16),  # position 1.8
            "particle_SMD": (8000 + 1000 + 4096) / (400 + 100 + 256),
            "agl_member_count_mean": 1,
            "agl_member_count_std": 0,
        },
        note="No agglomerates: sep2agl is inf, n_ppA is NaN.",
    ),
    make_case(
        "empty",
        [],
        groups=[],
        threshold=0.5,
        summary={
            "N_primary_particle": 0,
            "N_aerosol_particle": 0,
            "N_pp1": 0,
            "N_ppA": 0,
            "N_agl": 0,
            "N_collector_agl": 0,
            "N_similar_agl": 0,
            "N_pp1_separate": 0,
            "ER": math.nan,
            "Ra": math.nan,
            "sep2agl": math.nan,
            "n_ppA": math.nan,
            "n_ppP": math.nan,
            "particle_Dmean": math.nan,
            "particle_Dstd": math.nan,
            "particle_D10": math.nan,
            "particle_D50": math.nan,
            "particle_D90": math.nan,
            "particle_SMD": math.nan,
            "agl_member_count_mean": math.nan,
            "agl_member_count_std": math.nan,
        },
        note=(
            "An image without particles gives zero counts and NaN ratios, "
            "so one empty image doesn't stop a batch."
        ),
    ),
]

SUMMARY.append(
    make_case(
        "even_count_quantiles",
        [
            ("a", 0, 0, 5),
            ("b", 100, 0, 6),
            ("c", 200, 0, 10),
            ("d", 300, 0, 15),
        ],
        groups=[["a"], ["b"], ["c"], ["d"]],
        # D = 10, 12, 20, 30: every quantile falls between two values
        summary={
            "particle_D10": 10 + 0.3 * (12 - 10),  # position 0.3
            "particle_D50": 12 + 0.5 * (20 - 12),  # position 1.5
            "particle_D90": 20 + 0.7 * (30 - 20),  # position 2.7
        },
        note="Even count: the median interpolates between two values.",
    )
)
