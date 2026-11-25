"""Detection on one image: ``detect_particles`` and its steps.

The steps are separate helpers so a tuning preview can rerun only what
a change touches; ``detect_particles`` runs exactly these, so both give
the same circles.
"""

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from agglpy.core._grey_image import check_grey_image
from agglpy.core.detect import _hough as hct_pass
from agglpy.core.detect import _merge_passes, _range_boundary_duplicates

# imported under another name: the keyword ``preprocess`` would hide it
from agglpy.core.preprocess import preprocess as core_preprocess
from agglpy.core.region import Bounds, region_bounds, region_slices
from agglpy.params import (
    HCTParams,
    HCTRange,
    PreprocessParams,
    RegionParams,
)
from agglpy.tables import validate_particles

__all__ = [
    "Stage",
    "configured_steps",
    "cut_region",
    "detect_particles",
    "hct_pass",
    "matching_stages",
    "particles_from_passes",
    "pass_of_particles",
    "preprocess_stages",
]

# Progress name of each preprocessing step, by PreprocessParams field
_STEP_NAMES = {
    "median_blur": "median",
    "rolling_ball": "rolling ball",
    "clahe": "clahe",
}


def detect_particles(
    image: NDArray[np.uint8],
    *,
    region: RegionParams = RegionParams(),
    preprocess: PreprocessParams = PreprocessParams(),
    hct: HCTParams,
    on_step: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Detect the particles on one image's analysed region.

    Cuts the region from the image, preprocesses it, runs one HCT pass
    per range and moves the circles back into the raw image's
    coordinates, so the table lines up with the image a person opens.

    Parameters
    ----------
    image : ndarray
        The raw 2-D ``uint8`` grey image (as from
        `agglpy.io.read_image`); not modified.
    region : RegionParams, default RegionParams()
        The analysed region.
    preprocess : PreprocessParams, default PreprocessParams()
        The preprocessing steps.
    hct : HCTParams
        The HCT ranges.
    on_step : callable, optional
        Called with the name of each step as it starts: ``"median"``,
        ``"rolling ball"``, ``"clahe"`` (those configured), then
        ``"hct"``; for a progress display.

    Returns
    -------
    DataFrame
        A particle table ``id, x, y, r`` in px, in the raw image's
        frame; empty when nothing is found.

    Raises
    ------
    ImageError
        If `image` is not a 2-D ``uint8`` array of at least 1 x 1 px.
    ParamsError
        If the region leaves no row.

    Warns
    -----
    NoParticlesWarning
        If nothing is found.
    DuplicateParticlesWarning
        If near-identical circles remain after the range-boundary
        duplicates are dropped.

    Notes
    -----
    Runs `cut_region`, `preprocess_stages`, one `hct_pass` per range
    and `particles_from_passes`, the steps a tuning preview reruns, so
    both give the same circles.
    """
    bounds, cut = cut_region(image, region)
    report = on_step or (lambda step: None)
    stages = preprocess_stages(cut, preprocess, on_step=report)
    final = stages[-1].image if stages else cut
    report("hct")
    passes = [hct_pass(final, hct_range) for hct_range in hct.ranges]
    return particles_from_passes(final, passes, hct.ranges, bounds)


@dataclasses.dataclass(frozen=True, eq=False)
class Stage:
    """One preprocessing step's result on the analysed region.

    Attributes
    ----------
    field : str
        The ``PreprocessParams`` field of the step.
    name : str
        The step's progress name (``"median"``, ...).
    value : object
        The field's value the step ran with.
    image : ndarray
        The image after the step (read-only), ``uint8``.
    """

    field: str
    name: str
    value: object
    image: NDArray[np.uint8]


def cut_region(
    image: NDArray[np.uint8], region: RegionParams
) -> tuple[Bounds, NDArray[np.uint8]]:
    """Return the analysed region's bounds and the cut (a view)."""
    # checked first, so a colour image fails here, not on its shape
    height, width = check_grey_image(image).shape
    bounds = region_bounds(region, width, height)
    rows, cols = region_slices(bounds)
    return bounds, image[rows, cols]


def configured_steps(preprocess: PreprocessParams) -> list[tuple[str, Any]]:
    """Return the configured steps in run order: ``(field, value)``."""
    return [
        (field.name, value)
        for field in dataclasses.fields(preprocess)
        if (value := getattr(preprocess, field.name)) is not None
    ]


def matching_stages(
    stages: Sequence[Stage], preprocess: PreprocessParams
) -> int:
    """Count the leading `stages` that ran with `preprocess`'s steps.

    Those can be reused on the same cut: equal field and value from the
    first step on; the first difference ends the match.
    """
    kept = 0
    for stage, (field, value) in zip(
        stages, configured_steps(preprocess), strict=False
    ):
        if stage.field != field or stage.value != value:
            break
        kept += 1
    return kept


def preprocess_stages(
    cut: NDArray[np.uint8],
    preprocess: PreprocessParams,
    *,
    reuse: Sequence[Stage] = (),
    on_step: Callable[[str], None] | None = None,
) -> tuple[Stage, ...]:
    """Run the configured preprocessing steps, one stage per step.

    The steps run in field order, one core call each; the last stage's
    image equals ``core.preprocess(cut, preprocess)``.

    Parameters
    ----------
    cut : ndarray
        The analysed region.
    preprocess : PreprocessParams
        The preprocessing steps.
    reuse : sequence of Stage, default ()
        Stages of an earlier run on the same ``cut``. Leading
        stages with the same field and value are taken as they are;
        from the first difference on, every step runs.
    on_step : callable, optional
        Called with the name of each step that runs.

    Returns
    -------
    tuple of Stage
        One ``Stage`` per configured step; empty when none is.
    """
    report = on_step or (lambda step: None)
    stages = list(reuse[: matching_stages(reuse, preprocess)])
    image = stages[-1].image if stages else cut
    for field, value in configured_steps(preprocess)[len(stages) :]:
        report(_STEP_NAMES[field])
        out = core_preprocess(image, PreprocessParams(**{field: value}))
        out.setflags(write=False)
        stages.append(Stage(field, _STEP_NAMES[field], value, out))
        image = out
    return tuple(stages)


def particles_from_passes(
    image: NDArray[np.uint8],
    passes: Sequence[NDArray[np.float32]],
    ranges: Sequence[HCTRange],
    bounds: Bounds,
) -> pd.DataFrame:
    """Return the HCT passes' particle table, in the raw image's frame.

    Parameters
    ----------
    image : ndarray
        The preprocessed region the passes ran on (its size and values
        go into the no-circles warning).
    passes : sequence of ndarray
        One ``hct_pass`` result per range, in range order.
    ranges : sequence of HCTRange
        The range of each pass, in the same order.
    bounds : Bounds
        The region's bounds.

    Warns
    -----
    NoParticlesWarning
        If no pass found a circle.
    DuplicateParticlesWarning
        If near-identical circles remain after the range-boundary
        duplicates are dropped.
    """
    found = _merge_passes(image, passes, ranges)
    # zero for a bottom cut; kept so a top or left cut stays right
    found["x"] += bounds.x_min + 0.5
    found["y"] += bounds.y_min + 0.5
    # _merge_passes already warned about duplicates
    return validate_particles(found, warn_duplicates=False)


def pass_of_particles(
    passes: Sequence[NDArray[np.float32]], ranges: Sequence[HCTRange]
) -> NDArray[np.intp]:
    """Return the pass of each row `particles_from_passes` returns.

    Parameters
    ----------
    passes : sequence of ndarray
        One ``hct_pass`` result per range, in range order.
    ranges : sequence of HCTRange
        The range of each pass, in the same order.

    Returns
    -------
    ndarray
        The 0-based pass of each kept circle, in row order.
    """
    counts = [len(found) for found in passes]
    every = np.repeat(np.arange(len(passes)), counts)
    return every[~_range_boundary_duplicates(passes, ranges)]
