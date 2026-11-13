"""Draw the pictures of docs/agglomerate-structure.md into this folder.

Usage: python make_pictures.py

The agglomerates are the hand-built circle sets of the structure tests
(tests/support/structure_cases.py), so the pictures show exactly what
the tests check.
"""

import math
import sys
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from agglpy import draw
from agglpy.core.structure import structure_properties

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tests"))

from support.structure_cases import (  # noqa: E402 (needs the path above)
    CASES,
    agglomerate_table,
    graph_tables,
    run_case,
    wheel,
)

COLORS = {
    "core": "#3b6fb6",
    "small core": "#7d5ba6",
    "link": "#d7263d",
    "loop": "#2a9d55",
    "dangling end": "#f08a24",
    "no core": "#9a9a9a",
}

SEGMENTS = {
    "1 plain thread": "a thread on a core",
    "3 link between two cores": "a link between two cores",
    "4 thin ring on a core": "a loop on a core",
    "8 thread on a link": "a dangling end on a link",
    "14 triangle with a tail": "a small core (no core of 7)",
    "15 collector: layer, dendrite, singles": (
        "a collector: core, dendrite, two singles"
    ),
}


def _polar(deg, d):
    a = math.radians(deg)
    return (d * math.cos(a), d * math.sin(a))


def _panel(ax, particles, contacts, title):
    draw.particles(
        ax,
        particles,
        color_by="segment_type",
        colors=COLORS,
        alpha=0.45,
        legend=False,
    )
    xy = particles.set_index("id")[["x", "y"]]
    for a, b in zip(contacts["id_a"], contacts["id_b"], strict=True):
        (x0, y0), (x1, y1) = xy.loc[a], xy.loc[b]
        ax.plot([x0, x1], [y0, y1], color="0.3", lw=0.6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def _legend(fig, labels):
    handles = [
        Line2D([], [], marker="o", ls="", color=COLORS[s], label=s)
        for s in labels
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(labels))


def segments():
    fig = Figure(figsize=(10.5, 6.6), dpi=90)
    FigureCanvasAgg(fig)
    for i, (name, title) in enumerate(SEGMENTS.items()):
        _, particles, contacts = run_case(CASES[name][0])
        p, _ = structure_properties(
            particles, agglomerate_table(particles), contacts
        )
        _panel(fig.add_subplot(2, 3, i + 1), p, contacts, title)
    _legend(fig, list(COLORS)[:-1])
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(HERE / "segments.png")


def cliff():
    # A wheel of 7 with a thread of 3: a core at 7, a small core at 8.
    edges = [*wheel(0, range(1, 7)), (1, 7), (7, 8), (8, 9)]
    xy = {0: (0.0, 0.0)}
    for k in range(6):
        xy[k + 1] = _polar(60 * k, 19.0)
    for k, x in enumerate((38.0, 57.0, 76.0), 7):
        xy[k] = (x, 0.0)
    particles, agglomerates, contacts = graph_tables(edges, [10.0] * 10)
    particles["x"] = [xy[i][0] + 100 for i in particles["id"]]
    particles["y"] = [100 - xy[i][1] for i in particles["id"]]
    fig = Figure(figsize=(7.5, 3.6), dpi=90)
    FigureCanvasAgg(fig)
    for i, m in enumerate((7, 8)):
        p, _ = structure_properties(
            particles, agglomerates, contacts, min_core_members=m
        )
        _panel(
            fig.add_subplot(1, 2, i + 1),
            p,
            contacts,
            f"min_core_members={m}",
        )
    _legend(fig, ["core", "small core", "dangling end"])
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(HERE / "cliff.png")


if __name__ == "__main__":
    segments()
    cliff()
