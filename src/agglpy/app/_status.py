"""``Collection.status``: where each image's particles come from.

Read-only: every value is worked out from the files as they are now.
"""

import unicodedata
import warnings
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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

__all__ = [
    "STATUS_COLUMNS",
    "ImageFacts",
    "image_facts",
    "pixel_size",
    "table",
]

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


@dataclass(frozen=True)
class ImageFacts:
    """What the files say about one sheet image now.

    Read once by ``image_facts``, so ``status()`` and ``analyze()`` see
    the same facts.

    Attributes:
        entry: The sheet row.
        path: The image file, resolved.
        file: ``ok``, ``missing`` or ``unreadable``.
        meta: The image metadata; None unless ``file`` is ``ok``.
        meta_warnings: The metadata reader's warnings (e.g.
            ``PixelSizeWarning``), caught.
        hct: The detection's particle file; None without its CSV.
        corrected: The correction's particle file; None without its
            CSV.
        pixel_size: In m as read (the sheet's nm · 1e-9, the image
            metadata, or a record's copy of it), or None.
        pixel_size_source: ``images.xlsx`` / ``images.csv``, ``image
            file`` or ``none``.
        size: ``(width, height)`` in px from the image or a record;
            None when neither gives it.
        detection: The ``status()`` label of the detection.
        correction: The ``status()`` label of the correction.
    """

    entry: Entry
    path: Path
    file: Literal["ok", "missing", "unreadable"]
    meta: ImageMetadata | None
    meta_warnings: tuple[warnings.WarningMessage, ...]
    hct: ParticleFile | None
    corrected: ParticleFile | None
    pixel_size: float | None
    pixel_size_source: str
    size: tuple[int, int] | None
    detection: str
    correction: str

    @property
    def used(self) -> ParticleFile | None:
        """The particle file in use: the correction if there is one."""
        return self.corrected if self.corrected is not None else self.hct

    @property
    def particle_set(self) -> Literal["corrected", "hct"] | None:
        """The set in use; None without particles."""
        if self.corrected is not None:
            return "corrected"
        return "hct" if self.hct is not None else None


def table(folder: Path, sheet: Sheet, config: ConfigFile) -> pd.DataFrame:
    """One row per sheet image, in sheet order.

    Warns:
        UserWarning: For each particle CSV that does not read.
        UnreadableRecordWarning: For each record that does not read.
        LeftoverFoldersWarning: Once, if ``agglpy/`` holds folders of
            images the sheet does not list.
    """
    rows = [
        _row(image_facts(folder, sheet, entry, config.params(entry.image)))
        for entry in sheet.entries
    ]
    _warn_leftovers(folder, sheet)
    result = pd.DataFrame(rows, columns=list(STATUS_COLUMNS))
    return result.astype({c: _DTYPES.get(c, object) for c in STATUS_COLUMNS})


def image_facts(
    folder: Path, sheet: Sheet, entry: Entry, params: ImageParams
) -> ImageFacts:
    """Read the facts of one image from its files.

    Warns:
        UserWarning: If a particle CSV does not read.
        UnreadableRecordWarning: For each record that does not read.
    """
    path = resolve_stored(entry.file, folder)
    meta: ImageMetadata | None = None
    caught: list[warnings.WarningMessage] = []
    file: Literal["ok", "missing", "unreadable"]
    if not path.is_file():
        file = "missing"
    else:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
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
    records = [f.record for f in (used, hct, corr) if f and f.record]
    metres, source = _pixel_size(sheet, entry, meta, records)
    return ImageFacts(
        entry=entry,
        path=path,
        file=file,
        meta=meta,
        meta_warnings=tuple(caught),
        hct=hct,
        corrected=corr,
        pixel_size=metres,
        pixel_size_source=source,
        size=_size(meta, records),
        detection=_detection(hct, params, path, file),
        correction=_correction(corr, hct),
    )


def pixel_size(
    folder: Path, sheet: Sheet, entry: Entry
) -> tuple[float | None, str]:
    """The pixel size in m and its source, reading no more than needed.

    The precedence of ``image_facts``; nothing is hashed and the
    readers' warnings are dropped.
    """
    meta: ImageMetadata | None = None
    path = resolve_stored(entry.file, folder)
    if entry.pixel_size_nm is None and path.is_file():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                meta = read_image_metadata(path)
        except FileFormatError:
            pass
    return _pixel_size(sheet, entry, meta, _records(folder, entry.image))


def _records(
    folder: Path, name: str
) -> Iterator[DetectionRecord | CorrectionRecord]:
    """The records next to their CSVs, correction first, read lazily."""
    work = image_folder(folder, name)
    files: tuple[tuple[str, str, type[Any]], ...] = (
        (CORRECTED_CSV, CORRECTED_RECORD, CorrectionRecord),
        (HCT_CSV, HCT_RECORD, DetectionRecord),
    )
    for csv, path, kind in files:
        if not (work / csv).is_file():
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            record = read_record(work / path, kind)
        if record is not None:
            yield record


def _row(facts: ImageFacts) -> dict[str, object]:
    used = facts.used
    count: int | None = 0
    outside: int | None = 0
    if used is not None:
        if used.particles is None:
            count = outside = None
        else:
            count = len(used.particles)
            outside = (
                None
                if facts.size is None
                else int(
                    outside_region(
                        used.particles,
                        region_bounds(RegionParams(), *facts.size),
                    ).sum()
                )
            )
    corr = facts.corrected
    started = (
        corr.record.started_from
        if corr is not None and isinstance(corr.record, CorrectionRecord)
        else None
    )
    return {
        "image": facts.entry.image,
        "include": facts.entry.include,
        "file": facts.file,
        "pixel_size_nm": nanometres(facts.pixel_size),
        "pixel_size_source": facts.pixel_size_source,
        "particle_set": facts.particle_set or "none",
        "particle_count": count,
        "outside_image": outside,
        "detection": facts.detection,
        "correction": facts.correction,
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
    records: Iterable[DetectionRecord | CorrectionRecord],
) -> tuple[float | None, str]:
    """Sheet > image metadata > a record's copy of it > none, in m."""
    if entry.pixel_size_nm is not None:
        return entry.pixel_size_nm * 1e-9, sheet.path.name
    if meta is not None:
        if meta.pixel_size is None:
            return None, "none"
        return meta.pixel_size, "image file"
    # the image can't be read now: its metadata as the record stored it
    for record in records:
        if record.image.pixel_size_nm is not None:
            return record.image.pixel_size_nm * 1e-9, "image file"
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
