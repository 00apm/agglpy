"""``Collection.status``: where each image's particles come from.

Read-only: every value is worked out from the files as they are now.
"""

import unicodedata
import warnings
from pathlib import Path

import pandas as pd

from agglpy.app._config import ImageParams
from agglpy.app._config_file import ConfigFile
from agglpy.app._correction import based_on_detection
from agglpy.app._records import (
    CorrectionRecord,
    DetectionRecord,
    nanometres,
    read_record,
    same_image,
)
from agglpy.app._report import class_names
from agglpy.app._sheet import Entry, Sheet
from agglpy.app._sources import resolve_stored
from agglpy.app._work import (
    CORRECTED_CSV,
    CORRECTED_RECORD,
    HCT_CSV,
    HCT_RECORD,
    WORK_FOLDER,
    ParticleFile,
    image_folder,
    inspect,
)
from agglpy.core.region import outside_region, region_bounds
from agglpy.errors import FileFormatError, LeftoverFoldersWarning
from agglpy.io import ImageMetadata, read_image_metadata
from agglpy.params import RegionParams

__all__ = ["STATUS_COLUMNS", "table"]

STATUS_COLUMNS = (
    "image",
    "include",
    "file",
    "pixel_size_nm",
    "pixel_size_source",
    "particle_set",
    "particle_count",
    "outside_image",
    "detection",
    "correction",
    "kept",
    "warnings",
)
_DTYPES = {
    "include": "bool",
    "pixel_size_nm": "Float64",
    "particle_count": "Int64",
    "outside_image": "Int64",
}


def table(folder: Path, sheet: Sheet, config: ConfigFile) -> pd.DataFrame:
    """One row per sheet image, in sheet order.

    Warns:
        UserWarning: For each particle CSV that does not read.
        UnreadableRecordWarning: For each record that does not read.
        LeftoverFoldersWarning: Once, if ``agglpy/`` holds folders of
            images the sheet does not list.
    """
    rows = [
        _row(folder, sheet, entry, config.params(entry.image))
        for entry in sheet.entries
    ]
    _warn_leftovers(folder, sheet)
    result = pd.DataFrame(rows, columns=list(STATUS_COLUMNS))
    return result.astype({c: _DTYPES.get(c, object) for c in STATUS_COLUMNS})


def _row(
    folder: Path, sheet: Sheet, entry: Entry, params: ImageParams
) -> dict[str, object]:
    path = resolve_stored(entry.file, folder)
    meta: ImageMetadata | None = None
    if not path.is_file():
        file = "missing"
    else:
        try:
            # the pixel-size warnings belong to the records of the runs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                meta = read_image_metadata(path)
            file = "ok"
        except FileFormatError:
            file = "unreadable"
    work = image_folder(folder, entry.image)
    hct = inspect(work / HCT_CSV, work / HCT_RECORD, DetectionRecord)
    corr = inspect(
        work / CORRECTED_CSV, work / CORRECTED_RECORD, CorrectionRecord
    )
    used = corr if corr is not None else hct
    used_set = (
        "none"
        if used is None
        else ("corrected" if corr is not None else "hct")
    )
    records = [f.record for f in (used, hct, corr) if f and f.record]
    nm, source = _pixel_size(sheet, entry, meta, records)
    count: int | None = 0
    outside: int | None = 0
    if used is not None:
        size = _size(meta, records)
        if used.particles is None:
            count = outside = None
        else:
            count = len(used.particles)
            outside = (
                None
                if size is None
                else int(
                    outside_region(
                        used.particles, region_bounds(RegionParams(), *size)
                    ).sum()
                )
            )
    started = (
        corr.record.started_from
        if corr is not None and isinstance(corr.record, CorrectionRecord)
        else None
    )
    return {
        "image": entry.image,
        "include": entry.include,
        "file": file,
        "pixel_size_nm": nm,
        "pixel_size_source": source,
        "particle_set": used_set,
        "particle_count": count,
        "outside_image": outside,
        "detection": _detection(hct, params, path, file),
        "correction": _correction(corr, hct),
        "kept": (
            ""
            if started is None
            else f"{started.kept} of {started.detection_count} detected"
        ),
        "warnings": (
            class_names(used.record.warnings)
            if used is not None and used.record is not None
            else ""
        ),
    }


def _pixel_size(
    sheet: Sheet,
    entry: Entry,
    meta: ImageMetadata | None,
    records: list[DetectionRecord | CorrectionRecord],
) -> tuple[float | None, str]:
    """Sheet > image metadata > a record's copy of it > none."""
    if entry.pixel_size_nm is not None:
        return entry.pixel_size_nm, sheet.path.name
    if meta is not None:
        nm = nanometres(meta.pixel_size)
        return (nm, "image file") if nm is not None else (None, "none")
    # the image can't be read now: its metadata as the record stored it
    for record in records:
        if record.image.pixel_size_nm is not None:
            return record.image.pixel_size_nm, "image file"
    return None, "none"


def _size(
    meta: ImageMetadata | None,
    records: list[DetectionRecord | CorrectionRecord],
) -> tuple[int, int] | None:
    if meta is not None:
        return meta.width, meta.height
    for record in records:
        width, height = record.image.width, record.image.height
        if width is not None and height is not None:
            return width, height
    return None


def _detection(
    hct: ParticleFile | None, params: ImageParams, path: Path, file: str
) -> str:
    if hct is None:
        return "not set" if params.hct is None else "not detected"
    if hct.state != "matching" or not isinstance(hct.record, DetectionRecord):
        return hct.state
    # a missing image is shown by the file column; params decide then
    if file != "missing" and not same_image(hct.record.image, path):
        return "image changed"
    if hct.record.params != params:
        return "params changed"
    return "up to date"


def _correction(corr: ParticleFile | None, hct: ParticleFile | None) -> str:
    if corr is None:
        return ""
    if corr.state != "matching" or not isinstance(
        corr.record, CorrectionRecord
    ):
        return corr.state
    started = corr.record.started_from
    if started is None or not based_on_detection(started):
        return "not from detection"
    if hct is None or started.sha256 != hct.sha256:
        return "older detection"
    return "current detection"


def _warn_leftovers(folder: Path, sheet: Sheet) -> None:
    """Name the folders in ``agglpy/`` that no sheet row owns."""
    work = folder / WORK_FOLDER
    if not work.is_dir():
        return
    by_fold: dict[str, list[str]] = {}
    for name in sheet.names:
        by_fold.setdefault(_fold(name), []).append(name)
    by_file = {e.file: e.image for e in sheet.entries}
    listed = []
    for path in sorted(work.iterdir()):
        if not path.is_dir() or _owned(folder, path, by_fold):
            continue
        listed.append(path.name + _hint(path, by_file))
    if listed:
        warnings.warn(
            f"{WORK_FOLDER}/ has folders for images not in "
            f"{sheet.path.name}: {', '.join(listed)}",
            LeftoverFoldersWarning,
            stacklevel=2,
        )


def _owned(folder: Path, path: Path, by_fold: dict[str, list[str]]) -> bool:
    """Whether ``path`` is the folder agglpy opens for a sheet image.

    Decided by the file system: on Windows and macOS ``agglpy/d7-017/``
    is the folder of ``D7-017``; on Linux it is another folder, never
    read, so it is listed.
    """
    for name in by_fold.get(_fold(path.name), []):
        own = image_folder(folder, name)
        if path.name == name or (own.is_dir() and own.samefile(path)):
            return True
    return False


def _hint(path: Path, by_file: dict[str, str]) -> str:
    """Point at the sheet row whose image file the folder's record names."""
    records: list[CorrectionRecord | DetectionRecord | None] = []
    # a record whose CSV is gone is ignored, as everywhere else
    if (path / CORRECTED_CSV).is_file():
        records.append(read_record(path / CORRECTED_RECORD, CorrectionRecord))
    if (path / HCT_CSV).is_file():
        records.append(read_record(path / HCT_RECORD, DetectionRecord))
    for record in records:
        if record is not None and record.image.file in by_file:
            name = by_file[record.image.file]
            return (
                f" (same image file as {name}: rename the folder to {name} "
                "to keep its particles)"
            )
    return ""


def _fold(name: str) -> str:
    return unicodedata.normalize("NFC", name).casefold()
