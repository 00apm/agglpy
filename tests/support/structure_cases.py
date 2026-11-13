"""Hand-built agglomerates with their segments read by eye.

Circles in px (``(x, y)`` with radius ``R``, or ``(x, y, r)``), named so
the expected segments can be written by name. ``run_case`` runs a case
through ``find_agglomerates``; ``CASES`` maps a case to its circles and
its by-eye segments (every particle not listed is ``core``).
``CONTACTS`` lists the contacts of the short-named particles that the
circles are built to give, so a change of the geometry shows.
``graph_tables`` builds the tables of a contact graph given as an edge
list; ``G1``, ``G4``, ``G5`` are hand graphs with worked answers.

Equal circles: radius 10 px, neighbours 19 px apart, so neighbours
overlap by 1 px and second neighbours do not touch.
"""

import warnings

import numpy as np
import pandas as pd

from agglpy.core.agglomerates import find_agglomerates

S, R = 19.0, 10.0  # centre spacing and radius
H = S * np.sqrt(3) / 2


def hexagon(cx=0.0, cy=0.0, tag="c"):
    """A dense core: 19 circles in a hexagon (corners 3 contacts)."""
    out = {}
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                out[f"{tag}{q},{r}"] = (cx + S * (q + r / 2), cy + H * r)
    return out


def unit(deg):
    return np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])


E = np.array([2 * S, 0.0])  # corner c2,0
M = np.array([1.5 * S, -H])  # edge particle c2,-1
BUMP = np.array([2.5 * S, -H])  # lattice place touching E and M


def line(start, step, names):
    return {
        n: tuple(np.asarray(start) + (i + 1) * np.asarray(step))
        for i, n in enumerate(names)
    }


def ring():
    """a, b, c, d: a on E, d on M, bulging away from the core."""
    a = E + S * unit(30)
    d = M + S * unit(-30)
    w = (d - a) / np.linalg.norm(d - a)
    n = np.array([-w[1], w[0]])
    if n[0] < 0:
        n = -n
    alpha = (np.linalg.norm(d - a) - S) / 2
    beta = np.sqrt(S * S - alpha * alpha)
    b = a + alpha * w + beta * n
    c = d - alpha * w + beta * n
    return {"a": tuple(a), "b": tuple(b), "c": tuple(c), "d": tuple(d)}, n


def _pol(c, deg, d=S):
    a = np.radians(deg)
    return (c[0] + d * np.cos(a), c[1] + d * np.sin(a))


def _wheel(c, tag):
    """A centre and the six circles around it (a core of 7)."""
    out = {f"{tag}c": c}
    for deg in range(0, 360, 60):
        out[f"{tag}{deg}"] = _pol(c, deg)
    return out


def two_links(with_tail, with_triangle):
    """Wheels A and B joined by two thin paths (s1-s3 and u1-u5).

    ``with_tail`` adds v on s2; ``with_triangle`` a thread on A
    (x, then the triangle x y z, then t on y). w always hangs on u3.
    """
    c = _wheel((0.0, 0.0), "A") | _wheel((114.0, 0.0), "B")
    for i, x in enumerate((38.0, 57.0, 76.0), 1):
        c[f"s{i}"] = (x, 0.0)
    if with_tail:
        c["v"] = (57.0, 19.0)
    for i, x in enumerate((19.0, 38.0, 57.0, 76.0, 95.0), 1):
        c[f"u{i}"] = (x, -32.9)
    c["w"] = (57.0, -51.9)
    if with_triangle:
        x = (-38.0, 0.0)
        c["x"] = x
        c["y"] = _pol(x, 150)
        c["z"] = _pol(x, 210)
        c["t"] = _pol(c["y"], 120)
    return c


def _first_cases():
    core = hexagon()
    two = core | hexagon(7 * S, 0, "d") | line(E, (S, 0), "st")
    ring_circles, normal = ring()
    out_dir = BUMP / np.linalg.norm(BUMP)
    b = np.array(ring_circles["b"])
    return {
        "1 plain thread": (
            core | line(E, (S, 0), "pqr"),
            {"dangling end": "p q r"},
        ),
        "2 tip triangle": (
            core | line(E, (S, 0), "pqru") | {"v": (5.5 * S, H)},
            {"dangling end": "p q r u v"},
        ),
        "3 link between two cores": (two, {"link": "s t"}),
        "4 thin ring on a core": (
            core | ring_circles,
            {"loop": "a b c d"},
        ),
        # 5 and 5b pin the rule's known limit: a thread along the rim
        # (by eye a dangling end p q r) reads as a loop / as core.
        "5 thread along the rim": (
            core
            | {
                "p": (0, 2 * H + S),
                "q": (S, 2 * H + S),
                "r": (2 * S, 2 * H + S),
            },
            {"loop": "p q", "dangling end": "r"},
        ),
        "5b thread lying on one core particle": (
            core
            | {
                "p": tuple(E + S * unit(30)),
                "q": tuple(E + S * unit(-30)),
                "r": tuple(E + S * unit(-30) + S * unit(-30)),
            },
            {"dangling end": "r"},
        ),
        "6 triangle inside a thread": (
            core | line(E, (S, 0), "pqruw") | {"v": (5.5 * S, H)},
            {"dangling end": "p q r u v w"},
        ),
        "7 thread on a rim particle X": (
            core | {"X": tuple(BUMP)} | line(BUMP, S * out_dir, "pqr"),
            {"dangling end": "p q r"},
        ),
        "8 thread on a link": (
            two | {"r": (3 * S, S)},
            {"link": "s t", "dangling end": "r"},
        ),
        "9 thread on a thin ring": (
            core | ring_circles | line(b, S * normal, "pq"),
            {"loop": "a b c d", "dangling end": "p q"},
        ),
        "10 one particle on two core particles": (
            core | {"X": tuple(BUMP)},
            {},
        ),
        "11 two links, thread with a triangle": (
            two_links(with_tail=False, with_triangle=True),
            {
                "link": "s1 s2 s3 u1 u2 u3 u4 u5",
                "dangling end": "w x y z t",
            },
        ),
        "12 two links, a tail on each": (
            two_links(with_tail=True, with_triangle=False),
            {"link": "s1 s2 s3 u1 u2 u3 u4 u5", "dangling end": "v w"},
        ),
    }


def _later_cases():
    core = hexagon()
    out = {}

    # Thread Q (q1..q5) from a corner; the tip of thread P (p1..p3,
    # from another corner) lies on q4: a ring through the core, q5
    # sticks out.
    k = np.array([S, 2 * H])
    q = [k + S * unit(60)]
    for turn in (20, -10, -40, -60):
        q.append(q[-1] + S * unit(turn))
    target = q[3] + 0.9 * S * unit(-140)
    step = (target - E) / 3
    assert 10.5 < np.linalg.norm(step) < 19.5
    out["13 thread tip on another thread"] = (
        core
        | {f"q{i + 1}": tuple(v) for i, v in enumerate(q)}
        | {f"p{i}": tuple(E + i * step) for i in (1, 2, 3)},
        {"loop": "p1 p2 p3 q1 q2 q3 q4", "dangling end": "q5"},
    )

    out["14 triangle with a tail"] = (
        {
            "a": (0, 0),
            "b": (S, 0),
            "c": (S / 2, H),
            "d": (2 * S, 0),
            "e": (3 * S, 0),
        },
        {"small core": "a b c", "dangling end": "d e"},
    )

    # A collector with a layer on its rim (l0..l7, neighbours
    # overlap), one dendrite (d1 d2 d3), two single particles lying
    # on it (i1 i2).
    circles = {"X": (0, 0, 60.0)}
    for i in range(8):
        circles[f"l{i}"] = tuple(62 * unit(18 * i))
    for i, dist in enumerate((68, 87, 106), 1):
        circles[f"d{i}"] = tuple(dist * unit(200))
    circles["i1"] = (10, -20)
    circles["i2"] = (-25, 10)
    out["15 collector: layer, dendrite, singles"] = (
        circles,
        {"dangling end": "d1 d2 d3 i1 i2"},
    )

    # A ring (a b c d) on the core and a second ring (e f g) on the
    # first one's b and c.
    ring_circles, normal = ring()
    b, c = np.array(ring_circles["b"]), np.array(ring_circles["c"])
    w = (c - b) / np.linalg.norm(c - b)
    th = np.radians(20)
    e = b + S * (np.cos(th) * normal - np.sin(th) * w)
    g = c + S * (np.cos(th) * normal + np.sin(th) * w)
    half = np.linalg.norm(g - e) / 2
    f = (e + g) / 2 + np.sqrt(S * S - half * half) * normal
    out["16 ring on a ring"] = (
        core | ring_circles | {"e": tuple(e), "f": tuple(f), "g": tuple(g)},
        {"loop": "a b c d e f g"},
    )

    out["17 ring alone"] = (
        {f"h{i}": tuple(S * unit(60 * i)) for i in range(6)},
        {"small core": "h0 h1 h2 h3 h4 h5"},
    )

    # A thread whose particles also overlap their second neighbours
    # (centres 9.5 px apart, r 10).
    strip = {"t1": tuple(E + np.array([S, 0]))}
    for i in range(2, 7):
        strip[f"t{i}"] = tuple(E + np.array([S + (i - 1) * S / 2, 0]))
    out["18 thread overlapping second neighbours"] = (
        core | strip,
        {"dangling end": "t1 t2 t3 t4 t5 t6"},
    )

    out["19 link overlapping second neighbours"] = (
        core
        | hexagon(8 * S, 0, "d")
        | {f"s{i}": (3 * S + (i - 1) * S / 2, 0) for i in range(1, 6)},
        {"link": "s1 s2 s3 s4 s5"},
    )

    # One particle j with three arms, each to its own core.
    arms = {"j": (0, 0)}
    cores = {}
    for tag, deg in zip("cde", (0, 120, 240), strict=True):
        arms[f"{tag}1"] = tuple(S * unit(deg))
        centre = 4 * S * unit(deg)
        cores |= hexagon(centre[0], centre[1], tag * 2)
    out["21 Y link between three cores"] = (
        cores | arms,
        {"link": "j c1 d1 e1"},
    )
    return out


CASES = _first_cases() | _later_cases()

# The contacts of the particles with names of 1-2 characters, as the
# circles are built to give them ("a-b", names sorted).
CONTACTS = {
    "1 plain thread": "c2,0-p, p-q, q-r",
    "2 tip triangle": "c2,0-p, p-q, q-r, r-u, r-v, u-v",
    "3 link between two cores": "c2,0-s, d-2,0-t, s-t",
    "4 thin ring on a core": "a-b, a-c2,0, b-c, c-d, c2,-1-d",
    "5 thread along the rim": "c-1,2-p, c0,2-q, p-q, q-r",
    "5b thread lying on one core particle": "c2,0-p, c2,0-q, p-q, q-r",
    "6 triangle inside a thread": "c2,0-p, p-q, q-r, r-u, r-v, u-v, u-w",
    "7 thread on a rim particle X": "X-c2,-1, X-c2,0, X-p, p-q, q-r",
    "8 thread on a link": "c2,0-s, d-2,0-t, r-s, s-t",
    "9 thread on a thin ring": "a-b, a-c2,0, b-c, b-p, c-d, c2,-1-d, p-q",
    "10 one particle on two core particles": "X-c2,-1, X-c2,0",
    "11 two links, thread with a triangle": (
        "A0-A300, A0-A60, A0-Ac, A0-s1, A120-Ac, A180-Ac, A180-x, "
        "A240-Ac, A300-Ac, A300-u1, A60-Ac, B0-B300, B0-B60, B0-Bc, "
        "B120-Bc, B180-Bc, B180-s3, B240-Bc, B240-u5, B300-Bc, B60-Bc, "
        "s1-s2, s2-s3, t-y, u1-u2, u2-u3, u3-u4, u3-w, u4-u5, x-y, x-z, "
        "y-z"
    ),
    "12 two links, a tail on each": (
        "A0-A300, A0-A60, A0-Ac, A0-s1, A120-Ac, A180-Ac, A240-Ac, "
        "A300-Ac, A300-u1, A60-Ac, B0-B300, B0-B60, B0-Bc, B120-Bc, "
        "B180-Bc, B180-s3, B240-Bc, B240-u5, B300-Bc, B60-Bc, s1-s2, "
        "s2-s3, s2-v, u1-u2, u2-u3, u3-u4, u3-w, u4-u5"
    ),
    "13 thread tip on another thread": (
        "c0,2-q1, c2,0-p1, p1-p2, p2-p3, p3-q4, q1-q2, q2-q3, q3-q4, q4-q5"
    ),
    "14 triangle with a tail": "a-b, a-c, b-c, b-d, d-e",
    "15 collector: layer, dendrite, singles": (
        "X-d1, X-i1, X-i2, X-l0, X-l1, X-l2, X-l3, X-l4, X-l5, X-l6, "
        "X-l7, d1-d2, d2-d3, l0-l1, l1-l2, l2-l3, l3-l4, l4-l5, l5-l6, "
        "l6-l7"
    ),
    "16 ring on a ring": (
        "a-b, a-c2,0, b-c, b-e, c-d, c-g, c2,-1-d, e-f, f-g"
    ),
    "17 ring alone": "h0-h1, h0-h5, h1-h2, h2-h3, h3-h4, h4-h5",
    "18 thread overlapping second neighbours": (
        "c2,0-t1, t1-t2, t1-t3, t2-t3, t2-t4, t3-t4, t3-t5, t4-t5, "
        "t4-t6, t5-t6"
    ),
    "19 link overlapping second neighbours": (
        "c2,0-s1, d-2,0-s5, s1-s2, s1-s3, s2-s3, s2-s4, s3-s4, s3-s5, s4-s5"
    ),
    "21 Y link between three cores": (
        "c1-cc-2,0, c1-j, d1-dd2,-2, d1-j, e1-ee0,2, e1-j"
    ),
}


def run_case(circles):
    """Find the agglomerate of a case: ``(names, particles, contacts)``.

    ``names[i]`` is the name of particle id ``i``; the case must form
    one agglomerate.
    """
    names = list(circles)
    table = pd.DataFrame(
        {
            "id": np.arange(len(names)),
            "x": [circles[n][0] + 300 for n in names],
            "y": [300 - circles[n][1] for n in names],
            "r": [circles[n][2] if len(circles[n]) > 2 else R for n in names],
        }
    )
    with warnings.catch_warnings():
        # enclosed particles (case 15) are warned about; not the point
        warnings.simplefilter("ignore")
        particles, contacts = find_agglomerates(table)
    assert particles["agglomerate_id"].nunique() == 1
    return names, particles, contacts


def named_contacts(names, contacts):
    """The contacts with a short-named end, as in ``CONTACTS``."""
    found = sorted(
        "-".join(sorted((names[a], names[b])))
        for a, b in zip(contacts["id_a"], contacts["id_b"], strict=True)
        if len(names[a]) <= 2 or len(names[b]) <= 2
    )
    return ", ".join(found)


def expected_segments(names, by_eye):
    """Every name with its by-eye segment; unlisted names are core."""
    want = dict.fromkeys(names, "core")
    for segment, listed in by_eye.items():
        for name in listed.split():
            assert name in want, name
            want[name] = segment
    return want


def graph_tables(edges, r, aggl=None):
    """Tables of a contact graph: ids 0 … n-1, the edges as contacts."""
    n = len(r)
    labels = [0] * n if aggl is None else list(aggl)
    particles = pd.DataFrame(
        {
            "id": np.arange(n, dtype=np.int64),
            "r": np.asarray(r, dtype=np.float64),
            "agglomerate_id": np.asarray(labels, dtype=np.int64),
        }
    )
    a = np.array([e[0] for e in edges], dtype=np.int64)
    b = np.array([e[1] for e in edges], dtype=np.int64)
    contacts = pd.DataFrame(
        {
            "agglomerate_id": particles["agglomerate_id"].to_numpy()[a],
            "id_a": a,
            "id_b": b,
            "inside": np.zeros(len(a), dtype=bool),
        }
    )
    return particles, agglomerate_table(particles), contacts


def agglomerate_table(particles):
    ids = np.unique(particles["agglomerate_id"].to_numpy(dtype=np.int64))
    return pd.DataFrame({"agglomerate_id": ids})


def wheel(centre, ring):
    """Edges of a centre joined to each ring particle, ring closed."""
    ring = list(ring)
    spokes = [(centre, i) for i in ring]
    rim = [(ring[k], ring[(k + 1) % len(ring)]) for k in range(len(ring))]
    return spokes + rim


# A core of 7 with two dangling ends: 7-8 on particle 1, 9 on 3.
G4 = ([*wheel(0, range(1, 7)), (1, 7), (7, 8), (3, 9)], [2.0] + [1.0] * 9)
# Two cores of 7, a link 14-15 between them, a loop 16-17 on the first.
G5 = (
    wheel(0, range(1, 7))
    + wheel(7, range(8, 14))
    + [(3, 14), (14, 15), (15, 10)]
    + [(1, 16), (16, 17), (17, 2)],
    [2.0] + [1.0] * 17,
)
# A tree, base 0.
G1 = ([(0, 1), (1, 2), (2, 3), (1, 4)], [3.0, 1, 1, 1, 1])
