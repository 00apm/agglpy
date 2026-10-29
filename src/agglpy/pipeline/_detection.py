"""Detection on one image: ``detect_particles`` and its steps.

The steps are separate helpers so a tuning preview can rerun only what
a change touches; ``detect_particles`` runs exactly these, so both give
the same circles.
"""

import dataclasses
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from agglpy.core._grey_image import check_grey_image
from agglpy.core.detect import _hough as hct_pass
from agglpy.core.detect import _table

# imported under another name: the keyword ``preprocess`` would hide it
from agglpy.core.preprocess import preprocess as core_preprocess
from agglpy.core.region import Bounds, region_bounds, region_slices
from agglpy.params import HCTParams, PreprocessParams, RegionParams
from agglpy.tables import validate_particles

__all__ = [
    "Stage",
    "cut_region",
    "detect_particles",
    "hct_pass",
    "particles_from_passes",
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
    Runs ``cut_region``, ``preprocess_stages``, one ``hct_pass`` per
    range and ``particles_from_passes``.

    Args:
        image: The raw 2-D ``uint8`` grey image (as from
            ``agglpy.io.read_image``); not modified.
        region: The analysed region.
        preprocess: The preprocessing steps.
        hct: The HCT ranges.
        on_step: Called with the name of each step as it starts:
            ``"median"``, ``"rolling ball"``, ``"clahe"`` (those
            configured), then ``"hct"``; for a progress display.

    Returns:
        A particle table ``id, x, y, r`` in px, in the raw image's
        frame; empty when nothing is found.

    Raises:
        ImageError: If ``image`` is not a 2-D ``uint8`` array of at
            least 1 x 1 px.
        ParamsError: If the region leaves no row.

    Warns:
        NoParticlesWarning: If nothing is found.
        DuplicateParticlesWarning: If two ranges found the same
            particle.
    """
    bounds, cut = cut_region(image, region)
    report = on_step or (lambda step: None)
    stages = preprocess_stages(cut, preprocess, on_step=report)
    final = stages[-1].image if stages else cut
    report("hct")
    passes = [hct_pass(final, hct_range) for hct_range in hct.ranges]
    return particles_from_passes(final, passes, bounds)


@dataclasses.dataclass(frozen=True, eq=False)
class Stage:
    """One preprocessing step's result on the analysed region.

    Attributes:
        field: The ``PreprocessParams`` field of the step.
        name: The step's progress name (``"median"``, ...).
        value: The field's value the step ran with.
        image: The image after the step (read-only).
    """

    field: str
    name: str
    value: object
    image: NDArray[np.uint8]


def cut_region(
    image: NDArray[np.uint8], region: RegionParams
) -> tuple[Bounds, NDArray[np.uint8]]:
    """The analysed region's bounds and the region cut (a view)."""
    # checked first, so a colour image fails here, not on its shape
    height, width = check_grey_image(image).shape
    bounds = region_bounds(region, width, height)
    rows, cols = region_slices(bounds)
    return bounds, image[rows, cols]


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

    Args:
        cut: The analysed region.
        preprocess: The preprocessing steps.
        reuse: Stages of an earlier run on the same ``cut``. Leading
            stages with the same field and value are taken as they are;
            from the first difference on, every step runs.
        on_step: Called with the name of each step that runs.

    Returns:
        One ``Stage`` per configured step; empty when none is.
    """
    report = on_step or (lambda step: None)
    stages: list[Stage] = []
    image = cut
    # reuse only while every earlier stage was reused too
    matching = True
    for field in dataclasses.fields(preprocess):
        value = getattr(preprocess, field.name)
        if value is None:
            continue
        i = len(stages)
        matching = (
            matching
            and i < len(reuse)
            and reuse[i].field == field.name
            and reuse[i].value == value
        )
        if matching:
            stage = reuse[i]
        else:
            report(_STEP_NAMES[field.name])
            out = core_preprocess(
                image, PreprocessParams(**{field.name: value})
            )
            out.setflags(write=False)
            stage = Stage(field.name, _STEP_NAMES[field.name], value, out)
        stages.append(stage)
        image = stage.image
    return tuple(stages)


def particles_from_passes(
    image: NDArray[np.uint8],
    passes: Sequence[NDArray[np.float32]],
    bounds: Bounds,
) -> pd.DataFrame:
    """The particle table of HCT passes, in the raw image's frame.

    Args:
        image: The preprocessed region the passes ran on (its size and
            values go into the no-circles warning).
        passes: One ``hct_pass`` result per range, in range order.
        bounds: The region's bounds.

    Warns:
        NoParticlesWarning: If no pass found a circle.
        DuplicateParticlesWarning: If two ranges found the same
            particle.
    """
    found = _table(image, passes)
    # zero for a bottom cut; kept so a top or left cut stays right
    found["x"] += bounds.x_min + 0.5
    found["y"] += bounds.y_min + 0.5
    # _table already warned about duplicates
    return validate_particles(found, warn_duplicates=False)
