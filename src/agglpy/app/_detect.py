"""``Collection.detect``: HCT detection into the work folders."""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from agglpy.app._config import ImageParams
from agglpy.app._config_file import ConfigFile
from agglpy.app._files import replace_file
from agglpy.app._records import (
    DetectionRecord,
    ParticlesInfo,
    image_info,
    now,
    same_image,
    sha256,
    versions,
    write_record,
)
from agglpy.app._report import report
from agglpy.app._sheet import Entry, Sheet
from agglpy.app._sources import resolve_stored
from agglpy.app._work import (
    CORRECTED_CSV,
    HCT_CSV,
    HCT_RECORD,
    image_folder,
    inspect,
)
from agglpy.errors import AgglpyError
from agglpy.io import read_image, write_particles
from agglpy.pipeline import detect_particles

__all__ = ["run"]

# How a person's particles_hct.csv is named in the report: now / after
_KEPT = {
    "edited by hand": ("was edited by hand", "was edited by hand"),
    "no record": ("has no record", "had no record"),
    "unreadable": ("does not read", "did not read"),
}


def run(
    folder: Path,
    sheet: Sheet,
    config: ConfigFile,
    names: Sequence[str],
    *,
    force: bool,
    progress: bool,
) -> tuple[pd.DataFrame, list[tuple[type[Warning], str]]]:
    """Detect the images ``names`` (in sheet order, checked).

    Returns the report and the summary warnings for the caller to emit.
    """
    entries = {e.image: e for e in sheet.entries}
    rows = [
        _one(folder, entries[name], config.params(name), force)
        for name in names
    ]
    return report(rows), []


def _one(
    folder: Path, entry: Entry, params: ImageParams, force: bool
) -> dict[str, object]:
    """Decide and, if due, detect one image: its report row."""
    name = entry.image
    if params.hct is None:
        return _row(
            name,
            "skipped",
            "detection not set: hct.ranges is empty (set_params)",
        )
    path = resolve_stored(entry.file, folder)
    try:
        if not path.is_file():
            return _row(name, "error", f"image file not found: {entry.file}")
        work = image_folder(folder, name)
        outcome, message = "detected", ""
        # first match decides: a correction, a person's file, up to date
        if (work / CORRECTED_CSV).is_file():
            if not force:
                return _row(
                    name,
                    "skipped",
                    "has a correction; force=True re-detects (the "
                    "correction is kept)",
                )
            message = "the correction is kept and stays in use"
        else:
            hct = inspect(work / HCT_CSV, work / HCT_RECORD, DetectionRecord)
            if hct is not None and hct.state != "matching":
                state, past = _KEPT[hct.state]
                if not force:
                    return _row(
                        name,
                        "kept",
                        f"{HCT_CSV} {state}; force=True replaces it",
                    )
                outcome = "replaced"
                message = f"replaced {HCT_CSV}, which {past}"
            elif (
                hct is not None
                and isinstance(hct.record, DetectionRecord)
                and hct.record.params == params
                and same_image(hct.record.image, path)
            ):
                if not force:
                    return _row(name, "skipped", "up to date")
                message = "was up to date"
        count = _detect(entry, path, params, work)
    except Exception as exc:
        # one image's failure is a row; Ctrl+C still stops the batch
        return _row(name, "error", _error_text(exc))
    return _row(name, outcome, message, count)


def _detect(entry: Entry, path: Path, params: ImageParams, work: Path) -> int:
    """Detect one image and write its CSV, then its record."""
    assert params.hct is not None
    image = read_image(path)
    info = image_info(entry.file, path)
    table = detect_particles(
        image,
        region=params.region,
        preprocess=params.preprocess,
        hct=params.hct,
    )
    work.mkdir(parents=True, exist_ok=True)
    csv = work / HCT_CSV
    # overwrite: a temp file left by a hard stop must not block the write
    replace_file(csv, lambda tmp: write_particles(table, tmp, overwrite=True))
    agglpy, dependencies = versions()
    write_record(
        work / HCT_RECORD,
        DetectionRecord(
            created=now(),
            agglpy=agglpy,
            dependencies=dependencies,
            image=info,
            params=params,
            particles=ParticlesInfo(HCT_CSV, len(table), sha256(csv)),
            warnings=(),
        ),
    )
    return len(table)


def _row(
    name: str, outcome: str, message: str, count: int | None = None
) -> dict[str, object]:
    return {
        "image": name,
        "outcome": outcome,
        "particle_count": count,
        "message": message,
        "warnings": "",
    }


def _error_text(exc: Exception) -> str:
    """agglpy's messages as they are; others with their class name."""
    if isinstance(exc, AgglpyError):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"
