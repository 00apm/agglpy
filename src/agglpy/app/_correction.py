"""``Collection.import_particles``: a corrected particle file."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from agglpy.app._files import replace_file
from agglpy.app._records import (
    BaseInfo,
    CorrectionRecord,
    DetectionRecord,
    ImageInfo,
    ParticlesInfo,
    SourceInfo,
    csv_sha256,
    image_info,
    missing_image,
    now,
    sha256,
    versions,
    write_record,
)
from agglpy.app._report import class_names, report, warning_text
from agglpy.app._sheet import Entry
from agglpy.app._sources import resolve_stored, stored_path
from agglpy.app._work import (
    CORRECTED_CSV,
    CORRECTED_RECORD,
    HCT_CSV,
    HCT_RECORD,
    image_folder,
    inspect,
)
from agglpy.core.region import outside_region, region_bounds
from agglpy.errors import CorrectionMismatchWarning, OutputExistsError
from agglpy.io import particle_format, read_particles, write_particles
from agglpy.params import RegionParams
from agglpy.tables import _DUPLICATE_R_REL_TOL, _DUPLICATE_XY_TOL

__all__ = ["based_on_detection", "count_kept", "run_import"]

# A first guess, to calibrate on corrected images; changing it relabels
# existing corrections, no re-import needed.
_MIN_SHARED_FRACTION = 0.5

# How a person's particles_corrected.csv is named in the error
_CHANGED = {
    "edited outside agglpy": "was edited outside agglpy",
    "no record": "has no record",
    "file unreadable": "does not read",
}


def count_kept(detected: pd.DataFrame, corrected: pd.DataFrame) -> int:
    """Count the detected circles that have a matching corrected one.

    Matching as for duplicates: centres at most ``_DUPLICATE_XY_TOL``
    px apart, radii within ``_DUPLICATE_R_REL_TOL`` of the larger one.
    Each detected circle counts once.
    """
    if detected.empty or corrected.empty:
        return 0
    det_r = detected["r"].to_numpy(dtype=float)
    cor_r = corrected["r"].to_numpy(dtype=float)
    tree = cKDTree(corrected[["x", "y"]].to_numpy(dtype=float))
    near = tree.query_ball_point(
        detected[["x", "y"]].to_numpy(dtype=float), r=_DUPLICATE_XY_TOL
    )
    kept = 0
    for i, candidates in enumerate(near):
        r = cor_r[candidates]
        larger = np.maximum(r, det_r[i])
        if np.any(np.abs(r - det_r[i]) <= _DUPLICATE_R_REL_TOL * larger):
            kept += 1
    return kept


def based_on_detection(base: BaseInfo | None) -> bool:
    """Return whether a correction kept enough of its detection.

    The detection is the one it started from.
    """
    return (
        base is not None
        and base.detection_count > 0
        and base.kept >= _MIN_SHARED_FRACTION * base.detection_count
    )


def _stored_source(path: Path, folder: Path) -> str:
    """Give the source path as the record stores it, with ``/``.

    Relative to the collection when inside it, so the link survives
    moving the collection; absolute outside it, where the source stays
    put when the collection moves.
    """
    source, collection = path.resolve(), folder.resolve()
    if source.is_relative_to(collection):
        return stored_path(source, collection)
    return source.as_posix()


def run_import(
    folder: Path, entry: Entry, path: Path, *, format: str, force: bool
) -> tuple[pd.DataFrame, list[warnings.WarningMessage]]:
    """Import ``path`` as the correction of ``entry``'s image.

    Returns the one-row report and the warnings caught, for the caller
    to re-emit at the user's line.

    Raises
    ------
    FileNotFoundError
        If ``path`` doesn't exist.
    FileFormatError
        If ``path`` is not a particle file.
    OutputExistsError
        If the existing correction was changed by a person and
        ``force`` is False; nothing is written.
    """
    name = entry.image
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = particle_format(path) if format == "auto" else format
        table = read_particles(path, format=format)
        source = SourceInfo(
            _stored_source(path, folder), resolved, sha256(path)
        )
        read_end = len(caught)
        work = image_folder(folder, name)
        outcome, message = "imported", ""
        old = inspect(
            work / CORRECTED_CSV, work / CORRECTED_RECORD, CorrectionRecord
        )
        if old is not None:
            if old.state != "matching" and not force:
                raise OutputExistsError(
                    f"{CORRECTED_CSV} of {name} {_CHANGED[old.state]}; "
                    "force=True replaces it"
                )
            outcome = "replaced"
            if isinstance(old.record, CorrectionRecord):
                message = (
                    f"replaced correction from {old.record.source.path} "
                    f"({old.record.particles.count} particles, imported "
                    f"{old.record.created[:10]})"
                )
        base = _base(work, table)
        # warnings about the files already there belong to their records
        others_end = len(caught)
        image_path = resolve_stored(entry.file, folder)
        image = (
            image_info(entry.file, image_path)
            if image_path.is_file()
            else missing_image(entry.file)
        )
        signs = _mismatch_signs(table, base, image)
        if signs:
            warnings.warn(
                f"check that this is the correction of {name}: "
                + "; ".join(signs),
                CorrectionMismatchWarning,
                stacklevel=2,
            )
        own = caught[:read_end] + caught[others_end:]
        texts = tuple(warning_text(w) for w in own)
        work.mkdir(parents=True, exist_ok=True)
        csv = work / CORRECTED_CSV
        replace_file(
            csv, lambda tmp: write_particles(table, tmp, overwrite=True)
        )
        agglpy, dependencies = versions()
        write_record(
            work / CORRECTED_RECORD,
            CorrectionRecord(
                created=now(),
                agglpy=agglpy,
                dependencies=dependencies,
                image=image,
                particles=ParticlesInfo(
                    CORRECTED_CSV, len(table), csv_sha256(csv)
                ),
                warnings=texts,
                source=source,
                base=base,
            ),
        )
    row = {
        "image": name,
        "outcome": outcome,
        "particle_count": len(table),
        "message": message,
        "warnings": class_names(texts),
    }
    return report([row]), list(caught)


def _mismatch_signs(
    corrected: pd.DataFrame, base: BaseInfo | None, image: ImageInfo
) -> list[str]:
    """Return the signs that the file is not the image's correction.

    No outside check when the image size is unknown.
    """
    signs = []
    if base is not None and not based_on_detection(base):
        signs.append(
            f"it keeps only {base.kept} of {base.detection_count} detected "
            "circles (status() shows correction_base: no detection)"
        )
    if image.width is not None and image.height is not None:
        bounds = region_bounds(RegionParams(), image.width, image.height)
        outside = int(outside_region(corrected, bounds).sum())
        if outside == 1:
            signs.append("1 circle lies entirely outside the image")
        elif outside:
            signs.append(f"{outside} circles lie entirely outside the image")
    return signs


def _base(work: Path, corrected: pd.DataFrame) -> BaseInfo | None:
    """Return the detection present now, if its CSV reads."""
    hct = inspect(work / HCT_CSV, work / HCT_RECORD, DetectionRecord)
    if hct is None or hct.particles is None:
        return None
    # the record's date only describes the file it matches
    matching = hct.state == "matching" and hct.record is not None
    return BaseInfo(
        sha256=hct.sha256,
        detected=hct.record.created if matching and hct.record else None,
        kept=count_kept(hct.particles, corrected),
        detection_count=len(hct.particles),
    )
