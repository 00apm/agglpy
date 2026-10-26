"""``Collection.detect``: HCT detection into the work folders."""

import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

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
from agglpy.app._report import class_names, report, warning_text
from agglpy.app._sheet import Entry, Sheet
from agglpy.app._sources import resolve_stored
from agglpy.app._work import (
    CORRECTED_CSV,
    HCT_CSV,
    HCT_RECORD,
    image_folder,
    inspect,
)
from agglpy.errors import (
    AgglpyError,
    IncompleteRunWarning,
    LossyImageWarning,
)
from agglpy.io import read_image, write_particles
from agglpy.logger import logger
from agglpy.pipeline import detect_particles

__all__ = ["run"]

# How a person's particles_hct.csv is named in the report: now / after
_KEPT = {
    "edited by hand": ("was edited by hand", "was edited by hand"),
    "no record": ("has no record", "had no record"),
    "unreadable": ("does not read", "did not read"),
}
_JPEG_START = b"\xff\xd8\xff"

Caught = list[warnings.WarningMessage]


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

    Each image's warnings are caught, so the batch never stops on a
    warning filter set to "error". Returns the report and the summary
    warnings for the caller to emit at the user's line.
    """
    entries = {e.image: e for e in sheet.entries}
    rows: list[dict[str, object]] = []
    seen: dict[type[Warning], list[str]] = {}
    bar = tqdm(total=len(names), disable=not progress, unit="image")
    try:
        for name in names:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                row = _one(
                    folder,
                    entries[name],
                    config.params(name),
                    force,
                    caught,
                    _describe(bar, name),
                )
            row["warnings"] = class_names(warning_text(w) for w in caught)
            for w in caught:
                images = seen.setdefault(w.category, [])
                if name not in images:
                    images.append(name)
            logger.info("%s: %s %s", name, row["outcome"], row["message"])
            rows.append(row)
            bar.update(1)
    finally:
        bar.close()
    return report(rows), _summaries(seen, rows)


def _describe(bar: tqdm, name: str) -> Callable[[str], None]:
    """Show the image and its current step on the progress bar."""

    def show(step: str) -> None:
        bar.set_description(f"{name} · {step}")

    return show


def _summaries(
    seen: dict[type[Warning], list[str]], rows: list[dict[str, object]]
) -> list[tuple[type[Warning], str]]:
    """One warning per caught class, then the count of images not done."""
    found: list[tuple[type[Warning], str]] = [
        (
            cls,
            f"{cls.__name__} in {', '.join(images)} (messages in the "
            "returned table and the records)",
        )
        for cls, images in seen.items()
    ]
    counts = {"skipped": 0, "kept": 0, "error": 0}
    for row in rows:
        outcome = row["outcome"]
        # up-to-date skips are a normal resume, not something left undone
        if outcome in counts and row["message"] != "up to date":
            counts[str(outcome)] += 1
    if any(counts.values()):
        listed = ", ".join(f"{n} {o}" for o, n in counts.items() if n)
        found.append(
            (IncompleteRunWarning, f"{listed}; see the returned table")
        )
    return found


def _one(
    folder: Path,
    entry: Entry,
    params: ImageParams,
    force: bool,
    caught: Caught,
    on_step: Callable[[str], None],
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
        count = _detect(entry, path, params, work, caught, on_step)
    except Exception as exc:
        # one image's failure is a row; Ctrl+C still stops the batch
        text = _error_text(exc)
        logger.exception("%s: error %s", name, text)
        return _row(name, "error", text)
    return _row(name, outcome, message, count)


def _detect(
    entry: Entry,
    path: Path,
    params: ImageParams,
    work: Path,
    caught: Caught,
    on_step: Callable[[str], None],
) -> int:
    """Detect one image and write its CSV, then its record."""
    assert params.hct is not None
    start = len(caught)  # the record keeps what this detection warned
    with path.open("rb") as fh:
        if fh.read(3) == _JPEG_START:
            warnings.warn(
                f"{entry.file} is JPEG-compressed; edges may shift by the "
                "compression artefacts",
                LossyImageWarning,
                stacklevel=2,
            )
    on_step("reading")
    image = read_image(path)
    info = image_info(entry.file, path)
    table = detect_particles(
        image,
        region=params.region,
        preprocess=params.preprocess,
        hct=params.hct,
        on_step=on_step,
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
            warnings=tuple(warning_text(w) for w in caught[start:]),
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
