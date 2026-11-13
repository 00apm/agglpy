# %% [markdown]
# # Recipe: averages and ratios of agglomerate structure
#
# `res.with_structure()` gives every particle its branch depth and
# length and its segment, and every agglomerate its longest chain and
# maxima. Averages over branches or dangling ends need a choice the
# library can't make for you: which branches count. This recipe makes
# that choice visible in one line, so you can change it.
#
# Copy the functions and change them as you need. Run the cells in
# VS Code or Jupyter (with jupytext), or the whole file with Python.

# %%
import math

import pandas as pd

from agglpy.core.agglomerates import find_agglomerates
from agglpy.core.properties import agglomerate_properties, particle_properties
from agglpy.core.structure import structure_properties
from agglpy.tables import make_particles

KEY = ["image", "agglomerate_id"]


def _per_agglomerate(
    particles: pd.DataFrame, table: pd.DataFrame
) -> pd.DataFrame:
    """``table`` with a row for every agglomerate of ``particles``."""
    every = pd.MultiIndex.from_frame(particles[KEY].drop_duplicates())
    return table.reindex(every.sort_values())


# %% [markdown]
# ## Branches
#
# Every tip ends one branch. A single particle lying on an agglomerate
# is a tip of depth 1: on a large agglomerate there are many of them,
# and they pull the mean depth toward 1. `min_depth=2` leaves them out;
# set it to 0 to count every tip. An agglomerate without a branch that
# deep gets NaN, not 0: there is nothing to average.


# %%
def branch_statistics(
    particles: pd.DataFrame, min_depth: int = 2
) -> pd.DataFrame:
    """Mean and median branch depth and length, per agglomerate."""
    tips = particles[particles["tip"]]
    # The choice: which branches count.
    tips = tips[tips["branch_depth"] >= min_depth]
    table = tips.groupby(KEY).agg(
        branches=("branch_depth", "size"),
        branch_depth_mean=("branch_depth", "mean"),
        branch_depth_median=("branch_depth", "median"),
        branch_length_mean=("branch_length", "mean"),
        branch_length_median=("branch_length", "median"),
    )
    out = _per_agglomerate(particles, table)
    out["branches"] = out["branches"].fillna(0).astype(int)
    return out


# %% [markdown]
# ## Dangling ends
#
# A dangling end is one segment: its depth is the depth of its deepest
# particle, its length the length to its farthest one. The same filter
# leaves out ends of a single particle.


# %%
def dangling_end_statistics(
    particles: pd.DataFrame, min_depth: int = 2
) -> pd.DataFrame:
    """Mean and median dangling-end depth and length, per agglomerate."""
    rows = particles[particles["segment_type"] == "dangling end"]
    ends = rows.groupby([*KEY, "segment_id"]).agg(
        depth=("dangling_end_depth", "max"),
        length=("dangling_end_length", "max"),
        members=("id", "size"),
    )
    # The choice: which dangling ends count.
    ends = ends[ends["depth"] >= min_depth]
    table = ends.groupby(KEY).agg(
        dangling_ends=("depth", "size"),
        dangling_end_depth_mean=("depth", "mean"),
        dangling_end_depth_median=("depth", "median"),
        dangling_end_length_mean=("length", "mean"),
        dangling_end_length_median=("length", "median"),
    )
    out = _per_agglomerate(particles, table)
    out["dangling_ends"] = out["dangling_ends"].fillna(0).astype(int)
    return out


# %% [markdown]
# ## Volume beyond a depth
#
# The share of an agglomerate's volume that lies at `min_depth` contacts
# or more from its base: 0 for a compact agglomerate, growing as it
# spreads out.


# %%
def volume_share_beyond(
    particles: pd.DataFrame, min_depth: int = 2
) -> pd.Series:
    """Volume at ``branch_depth >= min_depth`` / total, per agglomerate."""
    beyond = particles["volume"].where(
        particles["branch_depth"] >= min_depth, 0
    )
    groups = particles[KEY]
    total = particles["volume"].groupby([groups[c] for c in KEY]).sum()
    return beyond.groupby([groups[c] for c in KEY]).sum() / total


# %% [markdown]
# ## Ratios of the agglomerate columns
#
# Lengths over the diameter of the largest member, and the fill: the
# projected area over the circle of the largest Feret diameter (1 for a
# single particle, small for a thin, spread-out agglomerate).


# %%
def relative_lengths(agglomerates: pd.DataFrame) -> pd.DataFrame:
    """Longest chain and longest branch in diameters of the largest."""
    d = agglomerates["D_largest"]
    chain = agglomerates["longest_chain_length"] / d
    branch = agglomerates["branch_length_max"] / d
    return pd.DataFrame(
        {
            "longest_chain_per_D_largest": chain,
            "branch_length_max_per_D_largest": branch,
        }
    )


def fill(agglomerates: pd.DataFrame) -> pd.Series:
    """Projected area over the area of the max-Feret circle."""
    circle = math.pi / 4 * agglomerates["D_feret_max"] ** 2
    return (agglomerates["area"] / circle).rename("fill")


# %% [markdown]
# ## Example
#
# One made-up image, in px, all particles of radius 10. Agglomerate 0:
# a core of 7 (a centre and six around it), a thread of 3 particles on
# one side and a single particle on the other. Agglomerate 1: a pair.
#
# The image goes through the core like `res.with_structure()` does it
# for each image; with `Results` use `res.particles` and
# `res.agglomerates` directly.

# %%
ring = [
    (19 * math.cos(math.radians(a)), 19 * math.sin(math.radians(a)))
    for a in range(0, 360, 60)
]
xy = [(0, 0), *ring, (38, 0), (57, 0), (76, 0), (-38, 0), (300, 0), (319, 0)]
found = make_particles([x for x, _ in xy], [y for _, y in xy], [10] * len(xy))
particles, contacts = find_agglomerates(found)
agglomerates = agglomerate_properties(particles, contacts)
particles, agglomerates = structure_properties(
    particle_properties(particles), agglomerates, contacts
)
particles = particles.assign(image="img1")
agglomerates = agglomerates.assign(image="img1")
particles[["id", "agglomerate_id", "segment_type", "branch_depth", "tip"]]

# %% [markdown]
# The deep tips of agglomerate 0 are the thread's end (depth 4) and
# the single particle (depth 2); the four free ring particles (depth 1)
# are left out. The pair has no deep branch: NaN.

# %%
branches = branch_statistics(particles)
branches

# %%
ends = dangling_end_statistics(particles)
share = volume_share_beyond(particles)
ratios = pd.concat(
    [agglomerates[KEY], relative_lengths(agglomerates), fill(agglomerates)],
    axis=1,
)
ratios
