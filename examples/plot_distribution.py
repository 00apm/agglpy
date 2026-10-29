# %% [markdown]
# # Recipe: plotting a size distribution
#
# agglpy computes distributions; how to plot them is your choice, so
# the plot is a recipe, not part of the library. `plot_distribution`
# draws the table that `distribution` (or `res.distribution`) returns:
#
# - one bar per size class, as wide as the class really is, on a log
#   axis;
# - bar height = `log_density`, the fraction per decade (dN / dlog D):
#   on a log axis the bar areas then compare like the fractions, even
#   where the classes have different widths;
# - the cumulative fraction as a curve on a second y axis (0 to 1);
# - one colour per group of `by`.
#
# Copy the function and change the plot as you need. Run the cells in
# VS Code or Jupyter (with jupytext), or the whole file with Python.

# %%
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from agglpy.core.agglomerates import find_agglomerates
from agglpy.core.distributions import distribution, size_classes
from agglpy.core.properties import agglomerate_properties
from agglpy.tables import make_particles

# %% [markdown]
# ## The plot
#
# The table has one row per class (and per group and basis) with the
# columns `left`, `right`, `width`, `basis`, `fraction`, `cumulative`
# and `log_density`. `basis` is `"number"` for counts, or the weight
# column given to `distribution(..., weights="volume")`.


# %%
def plot_distribution(
    table: pd.DataFrame,
    *,
    ax: Axes,
    by: str | None = None,
    basis: str = "number",
) -> Axes:
    """Bars per size class on a log axis, the cumulative curve on top."""
    rows = table[table["basis"] == basis]
    groups = [(None, rows)] if by is None else rows.groupby(by, sort=False)
    cumulative = ax.twinx()
    for i, (label, group) in enumerate(groups):
        colour = f"C{i}"
        ax.bar(
            group["left"],
            group["log_density"],
            width=group["width"],
            align="edge",  # a bar spans its class, left to right
            color=colour,
            alpha=0.4,
            edgecolor=colour,
            label=label,
        )
        # from 0 at the first class's left edge to 1 at the last right
        cumulative.plot(
            np.r_[group["left"].iloc[0], group["right"]],
            np.r_[0.0, group["cumulative"]],
            color=colour,
        )
    ax.set_xscale("log")
    ax.set_ylabel(f"fraction per decade ({basis})")
    cumulative.set_ylabel("cumulative fraction")
    cumulative.set_ylim(0, 1.05)
    if by is not None:
        ax.legend(title=by)
    return ax


# %% [markdown]
# ## Example
#
# Three made-up images in px, two of them in the series `"low flow"`
# and one in `"high flow"`: 150 particles each at random places, with
# lognormal radii (larger at high flow). Each image goes through the
# core (`find_agglomerates`, `agglomerate_properties`); the images
# table says which series each image belongs to, as the sheet's user
# columns do for `res.images`.

# %%
rng = np.random.default_rng(1)
images = pd.DataFrame(
    {
        "image": ["img1", "img2", "img3"],
        "series": ["low flow", "low flow", "high flow"],
    }
)
median_r = {"low flow": 6.0, "high flow": 10.0}
agglomerates = pd.concat(
    [
        agglomerate_properties(
            find_agglomerates(
                make_particles(
                    rng.uniform(0, 1000, 150),
                    rng.uniform(0, 1000, 150),
                    rng.lognormal(np.log(median_r[series]), 0.4, 150),
                )
            )
        ).assign(image=name)
        for name, series in zip(images.image, images.series, strict=True)
    ],
    ignore_index=True,
)

# %% [markdown]
# Size classes from 1 to 1000 px, four per decade; the distribution of
# the agglomerate diameter `D`, by series:

# %%
edges = size_classes(1, 1000, step=10**0.25, scale="log")
dist = distribution(images, agglomerates, "D", edges, by="series")
fig = Figure(figsize=(6, 4), layout="constrained")
plot_distribution(dist, ax=fig.add_subplot(), by="series")
fig.axes[0].set_xlabel("agglomerate diameter D [px]")
fig
