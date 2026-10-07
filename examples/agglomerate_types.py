# %% [markdown]
# # Recipe: agglomerate types (collector / similar / separate)
#
# agglpy measures agglomerates; how to label them is your choice, so
# this rule is a recipe, not part of the library. It is the rule agglpy
# 0.4 used, applied to the `size_ratio` property: the diameter of the
# second largest member divided by the diameter of the largest one.
#
# - one member: `"separate"`;
# - `size_ratio <= threshold`: `"collector"` (one large particle has
#   collected smaller ones);
# - otherwise: `"similar"` (members of comparable size).
#
# Copy the functions and change the rule as you need. Run the cells in
# VS Code or Jupyter (with jupytext), or the whole file with Python.

# %%
import numpy as np
import pandas as pd

from agglpy.core.agglomerates import find_agglomerates
from agglpy.core.properties import agglomerate_properties
from agglpy.tables import make_particles

# %% [markdown]
# ## The rule
#
# `agglomerate_types` labels each row of an agglomerate table. It needs
# only two columns, `member_count` and `size_ratio`, so it works on any
# table agglpy gives you, for one image or many.

# %%
TYPES = ["collector", "similar", "separate"]


def agglomerate_types(
    agglomerates: pd.DataFrame, threshold: float
) -> pd.Series:
    """Type of each agglomerate, one value per row of the table."""
    # The first condition that holds wins; rows matching none are
    # "similar". Add or reorder conditions to change the rule.
    types = np.select(
        [
            agglomerates["member_count"].to_numpy() == 1,
            agglomerates["size_ratio"].to_numpy() <= threshold,
        ],
        ["separate", "collector"],
        default="similar",
    )
    return pd.Series(types, index=agglomerates.index, name="type")


# %% [markdown]
# ## Counting types per image
#
# The usual question is how many agglomerates of each type an image (or
# a sample, or any other column) has. `pd.crosstab` counts them; a type
# that occurs nowhere still gets its column, with zeros, so tables of
# different images line up.


# %%
def type_counts(
    agglomerates: pd.DataFrame, threshold: float, by: str = "image"
) -> pd.DataFrame:
    """Number of agglomerates of each type, one row per value of ``by``."""
    types = agglomerate_types(agglomerates, threshold)
    counts = pd.crosstab(agglomerates[by], types)
    return counts.reindex(columns=TYPES, fill_value=0)


# %% [markdown]
# ## Example
#
# Two small made-up images, in px. `img1` has a collector pair (r = 10
# with r = 4 touching it), a single particle and a pair of equal
# particles; `img2` has two single particles.
#
# Each image goes through the core: `find_agglomerates` groups the
# particles, `agglomerate_properties` measures each agglomerate (one
# row each, with `member_count` and `size_ratio`). The `image` column
# says where each row came from, so the tables can be stacked.

# %%
images = {
    "img1": make_particles(
        [0, 14, 100, 200, 220], [0, 0, 0, 0, 0], [10, 4, 5, 10, 10]
    ),
    "img2": make_particles([0, 300], [0, 0], [6, 6]),
}
agglomerates = pd.concat(
    [
        agglomerate_properties(find_agglomerates(p)).assign(image=name)
        for name, p in images.items()
    ],
    ignore_index=True,
)
agglomerates[["image", "member_count", "size_ratio"]]

# %% [markdown]
# With `threshold = 0.5` the collector pair (ratio 8 / 20 = 0.4) is a
# collector, the equal pair (ratio 1) is similar:

# %%
counts = type_counts(agglomerates, threshold=0.5)
counts
