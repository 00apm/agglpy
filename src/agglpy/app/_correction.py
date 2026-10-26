"""``Collection.import_particles``: a corrected particle file."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from agglpy.app._files import replace_file
from agglpy.app._records import (
    CorrectionRecord,
    DetectionRecord,
    ParticlesInfo,
    SourceInfo,
    StartedFrom,
    image_info,
    missing_image,
    now,
    sha256,
    versions,
    write_record,
)
from agglpy.app._report import class_names, report, warning_text
from agglpy.app._sheet import Entry
from agglpy.app._sources import resolve_stored
from agglpy.app._work import (
    CORRECTED_CSV,
    CORRECTED_RECORD,
    HCT_CSV,
    HCT_RECORD,
    image_folder,
    inspect,
)
from agglpy.errors import OutputExistsError
from agglpy.io import particle_format, read_particles, write_particles
from agglpy.tables import _DUPLICATE_R_REL_TOL, _DUPLICATE_XY_TOL

__all__ = ["based_on_detection", "count_kept", "run_import"]

# A first guess, to calibrate on corrected images; changing it relabels
# existing corrections, no re-import needed.
_MIN_SHARED_FRACTION = 0.5

# How a person's particles_corrected.csv is named in the error
_CHANGED = {
    "edited by hand": "was edited by hand",
    "no record": "has no record",
    "unreadable": "does not read",
}


def count_kept(detected: pd.DataFrame, corrected: pd.DataFrame) -> int:
    """How many detected circles have a matching corrected circle.

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


def based_on_detection(started: StartedFrom | None) -> bool:
    """Whether a correction kept enough of the detection it started from."""
    return (
        started is not None
        and started.detection_count > 0
        and started.kept >= _MIN_SHARED_FRACTION * started.detection_count
    )


def run_import(
    folder: Path, entry: Entry, path: Path, *, format: str, force: bool
) -> tuple[pd.DataFrame, list[warnings.WarningMessage]]:
    """Import ``path`` as the correction of ``entry``'s image.

    Returns the one-row report and the warnings caught, for the caller
    to re-emit at the user's line.

    Raises:
        FileNotFoundError: If ``path`` doesn't exist.
        FileFormatError: If ``path`` is not a particle file.
        OutputExistsError: If the existing correction was changed by a
            person and ``force`` is False; nothing is written.
    """
    name = entry.image
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = particle_format(path) if format == "auto" else format
        table = read_particles(path, format=format)
        # absolute and with "/": the notebook's working folder and the OS
        # mean nothing to whoever reads the record later
        source = SourceInfo(path.resolve().as_posix(), resolved, sha256(path))
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
        started = _started_from(work, table)
        # warnings about the files already there belong to their records
        others_end = len(caught)
        if started is not None and not based_on_detection(started):
            warnings.warn(
                f"correction keeps only {started.kept} of "
                f"{started.detection_count} detected circles; shown as not "
                "based on the detection",
                UserWarning,
                stacklevel=2,
            )
        image_path = resolve_stored(entry.file, folder)
        image = (
            image_info(entry.file, image_path)
            if image_path.is_file()
            else missing_image(entry.file)
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
                    CORRECTED_CSV, len(table), sha256(csv)
                ),
                warnings=texts,
                source=source,
                started_from=started,
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


def _started_from(work: Path, corrected: pd.DataFrame) -> StartedFrom | None:
    """The detection present now, if its CSV reads."""
    hct = inspect(work / HCT_CSV, work / HCT_RECORD, DetectionRecord)
    if hct is None or hct.particles is None:
        return None
    # the record's date only describes the file it matches
    matching = hct.state == "matching" and hct.record is not None
    return StartedFrom(
        sha256=hct.sha256,
        detected=hct.record.created if matching and hct.record else None,
        kept=count_kept(hct.particles, corrected),
        detection_count=len(hct.particles),
    )
