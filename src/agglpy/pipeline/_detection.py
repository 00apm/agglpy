"""Detection on one image: ``detect_particles``."""

import dataclasses
from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from agglpy.core._grey_image import check_grey_image
from agglpy.core.detect import detect_hct

# imported under another name: the keyword ``preprocess`` would hide it
from agglpy.core.preprocess import preprocess as core_preprocess
from agglpy.core.region import region_bounds, region_slices
from agglpy.params import HCTParams, PreprocessParams, RegionParams
from agglpy.tables import validate_particles

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
    For previews and tuning (one range at a time, the preprocessed
    image), call the core steps directly.

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
    # checked first, so a colour image fails here, not on its shape
    height, width = check_grey_image(image).shape
    bounds = region_bounds(region, width, height)
    rows, cols = region_slices(bounds)
    report = on_step or (lambda step: None)
    cut = image[rows, cols]
    # one step per call: the steps run in field order and each only
    # transforms the image, so the result equals a single call
    for field in dataclasses.fields(preprocess):
        value = getattr(preprocess, field.name)
        if value is not None:
            report(_STEP_NAMES[field.name])
            cut = core_preprocess(cut, PreprocessParams(**{field.name: value}))
    report("hct")
    found = detect_hct(cut, hct)
    # zero for a bottom cut; kept so a top or left cut stays right
    found["x"] += bounds.x_min + 0.5
    found["y"] += bounds.y_min + 0.5
    # detect_hct already warned about duplicates
    return validate_particles(found, warn_duplicates=False)
