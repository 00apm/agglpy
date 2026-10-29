"""``Collection.analyze``: the particle sets into one ``Results``."""

import dataclasses
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from agglpy.app._config import ImageParams, plain
from agglpy.app._config_file import ConfigFile
from agglpy.app._records import (
    CorrectionRecord,
    DetectionRecord,
    nanometres,
    now,
)
from agglpy.app._report import warning_text
from agglpy.app._results import Origin, Results
from agglpy.app._sheet import Entry, Sheet, typed_user_columns
from agglpy.app._status import ImageFacts, image_facts
from agglpy.app._work import CORRECTED_CSV, HCT_CSV
from agglpy.core.properties import to_physical
from agglpy.errors import NotReadyError, OutdatedParticlesWarning
from agglpy.logger import logger
from agglpy.pipeline import ImageResult
from agglpy.pipeline import analyze as analyze_image

__all__ = ["run", "summary_text"]

# Images named in a summary warning; the export record has them all.
_LISTED = 10

_DTYPES: dict[str, Any] = {
    "image": object,
    "pixel_size": "float64",
    "pixel_size_source": object,
    "fov_area": "float64",
    "width": "int64",
    "height": "int64",
    "particle_set": object,
    "outside_region": "int64",
}


def run(
    folder: Path, sheet: Sheet, config: ConfigFile, *, progress: bool
) -> tuple[Results, list[tuple[type[Warning], str]]]:
    """Analyse every included image of ``sheet``, in sheet order.

    Every image is checked before any is analysed. Each image's
    warnings are caught; returns the results and the summary warnings
    (outdated detections first, then one per class) for the caller to
    emit at the user's line.

    Raises:
        NotReadyError: If no image is included or an included image is
            not ready; the message lists them all.
    """
    included = [e for e in sheet.entries if e.include]
    if not sheet.entries:
        raise NotReadyError(f"{sheet.path.name} has no images")
    if not included:
        raise NotReadyError(
            "no image to analyse: every row of "
            f"{sheet.path.name} has include false"
        )
    params = {e.image: config.params(e.image) for e in included}
    facts = [image_facts(folder, sheet, e, params[e.image]) for e in included]
    _check_ready(sheet, facts)

    rows: list[dict[str, object]] = []
    particles: list[pd.DataFrame] = []
    agglomerates: list[pd.DataFrame] = []
    records: dict[str, dict[str, object]] = {}
    outdated: list[tuple[str, str]] = []
    seen: dict[type[Warning], list[tuple[str, str]]] = {}
    bar = tqdm(total=len(facts), disable=not progress, unit="image")
    try:
        for f in facts:
            name = f.entry.image
            bar.set_description(name)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    result, row = _one(f, params[name])
                    p = row["pixel_size"]
                    assert isinstance(p, float)
                    for tables, table in (
                        (particles, result.particles),
                        (agglomerates, result.agglomerates),
                    ):
                        physical = to_physical(table, p)
                        physical.insert(0, "image", name)
                        tables.append(physical)
                except Exception as exc:
                    exc.add_note(f"while analysing image {name}")
                    raise
            # a metadata warning counts only when the metadata gave the
            # pixel size
            found = [
                *(f.meta_warnings if f.entry.pixel_size_nm is None else ()),
                *caught,
            ]
            for w in found:
                seen.setdefault(w.category, []).append((name, str(w.message)))
            rows.append(row)
            records[name] = _record(
                f, params[name], row, [warning_text(w) for w in found]
            )
            reason = _outdated(f, params[name])
            if reason is not None:
                outdated.append((name, reason))
                logger.warning("%s: %s", name, reason)
            logger.info(
                "%s: %d particles (%s), %d agglomerates, %d outside the "
                "region; pixel size %s nm from %s",
                name,
                len(result.particles),
                f.particle_set,
                len(result.agglomerates),
                row["outside_region"],
                nanometres(p),
                f.pixel_size_source,
            )
            bar.update(1)
    finally:
        bar.close()
    logger.info("analysed %d images", len(facts))

    images = _images_table(sheet, included, rows)
    origin = Origin(
        analyzed=now(),
        collection=folder,
        sheet=sheet.path.name,
        images=records,
        files={f.entry.image: f.path for f in facts},
        excluded=tuple(
            {"image": e.image, "note": e.note}
            for e in sheet.entries
            if not e.include
        ),
        steps=(),
        names=sheet.names,
    )
    results = Results(
        images,
        pd.concat(particles, ignore_index=True),
        pd.concat(agglomerates, ignore_index=True),
        origin,
    )
    summaries = [(cls, summary_text(found)) for cls, found in seen.items()]
    if outdated:
        summaries.insert(
            0, (OutdatedParticlesWarning, _outdated_text(outdated))
        )
    return results, summaries


def summary_text(messages: list[tuple[str, str]]) -> str:
    """One warning for a class: the images and their messages.

    ``messages`` are ``(image, message)`` in order; the first
    ``_LISTED`` images are listed, then how many more.
    """
    images = list(dict.fromkeys(name for name, _ in messages))
    listed = set(images[:_LISTED])
    noun = "image" if len(images) == 1 else "images"
    lines = [
        f"{len(images)} {noun} (each message is in res.export's record.yml):",
        *(f"  {name}: {text}" for name, text in messages if name in listed),
    ]
    if len(images) > _LISTED:
        lines.append(f"  and {len(images) - _LISTED} more")
    return "\n".join(lines)


def _outdated(f: ImageFacts, params: ImageParams) -> str | None:
    """Why the detection in use no longer matches; None if it does."""
    if f.particle_set != "hct":
        return None  # a correction is the user's final word
    if f.detection == "image changed":
        return "image file changed since detect()"
    if f.detection != "params changed":
        return None
    if params.hct is None:
        return (
            "detection was switched off in agglpy.yml after detect(); "
            "set hct again or import_particles()"
        )
    return "params changed since detect()"


def _outdated_text(outdated: list[tuple[str, str]]) -> str:
    count = len(outdated)
    were = "image was" if count == 1 else "images were"
    return "\n".join(
        [
            f"{count} {were} analysed with particles from an outdated "
            "detection:",
            *(f"  {name}: {reason}" for name, reason in outdated),
            "These results do not reflect the current params. Run "
            "col.detect() to update them, then analyze() again.",
        ]
    )


def _check_ready(sheet: Sheet, facts: list[ImageFacts]) -> None:
    """Raise one ``NotReadyError`` naming every image not ready."""
    lines: list[str] = []
    fixes: dict[str, None] = {}
    where = sheet.path.name
    for f in facts:
        problems: list[str] = []
        if f.pixel_size is None:
            problems.append("no pixel size")
            fixes[f"fill pixel_size_nm in {where}"] = None
        used = f.used
        if used is None:
            problems.append("no particles")
            fixes["run detect() or import_particles()"] = None
        if f.file != "ok":
            state = "not found" if f.file == "missing" else "does not read"
            problems.append(f"image file {state}: {f.entry.file}")
            fixes[f"fix the image file or its path in {where}"] = None
        if used is not None and used.particles is None:
            csv = HCT_CSV if f.particle_set == "hct" else CORRECTED_CSV
            problems.append(f"{csv} does not read")
            fixes["import the particles again or delete the file"] = None
        if problems:
            lines.append(f"  {f.entry.image}: {'; '.join(problems)}")
    if not lines:
        return
    advice = "; ".join(
        [
            *fixes,
            "set include to false to leave an image out. col.status() "
            "shows every image.",
        ]
    )
    raise NotReadyError(
        "\n".join(
            [
                f"{len(lines)} of {len(facts)} images are not ready for "
                "analyze():",
                *lines,
                advice[0].upper() + advice[1:],
            ]
        )
    )


def _one(
    f: ImageFacts, params: ImageParams
) -> tuple[ImageResult, dict[str, object]]:
    """Analyse one ready image: the result in px and its images row."""
    assert f.meta is not None and f.pixel_size is not None
    used = f.used
    assert used is not None and used.particles is not None
    result = analyze_image(
        used.particles,
        width=f.meta.width,
        height=f.meta.height,
        region=params.region,
    )
    assert result.fov_area is not None
    p = f.pixel_size
    row: dict[str, object] = {
        "image": f.entry.image,
        "pixel_size": p,
        "pixel_size_source": f.pixel_size_source,
        "fov_area": result.fov_area * p**2,
        "width": f.meta.width,
        "height": f.meta.height,
        "particle_set": f.particle_set,
        "outside_region": len(result.outside_ids),
    }
    return result, row


def _images_table(
    sheet: Sheet, included: list[Entry], rows: list[dict[str, object]]
) -> pd.DataFrame:
    """The agglpy columns, then the user columns, then ``note``."""
    table = pd.DataFrame(rows, columns=list(_DTYPES)).astype(_DTYPES)
    # typed over the analysed rows only: an empty cell on an excluded
    # row doesn't turn an int column into float
    user = typed_user_columns(
        dataclasses.replace(sheet, entries=tuple(included))
    ).drop(columns="image")
    table = pd.concat([table, user], axis=1)
    table["note"] = pd.Series([e.note for e in included], dtype=object)
    return table


def _record(
    f: ImageFacts,
    params: ImageParams,
    row: dict[str, object],
    texts: list[str],
) -> dict[str, object]:
    """The image's entry in the export record, as plain data."""
    used = f.used
    assert used is not None and used.particles is not None
    return {
        "file": f.entry.file,
        "width": row["width"],
        "height": row["height"],
        "pixel_size": row["pixel_size"],
        "pixel_size_source": row["pixel_size_source"],
        "region": plain(params.region, "region"),
        "outside_region": row["outside_region"],
        "particle_set": row["particle_set"],
        "particles": {
            "file": HCT_CSV if f.particle_set == "hct" else CORRECTED_CSV,
            "count": len(used.particles),
            "sha256": used.sha256,
            "state": used.state,
        },
        "detection": f.detection,
        "correction": f.correction,
        "made_by": _made_by(used.record),
        "warnings": texts,
    }


def _made_by(
    record: DetectionRecord | CorrectionRecord | None,
) -> dict[str, object] | None:
    """The key facts of the particle record used; None without one."""
    if isinstance(record, DetectionRecord):
        return {
            "kind": "detection",
            "created": record.created,
            "agglpy": record.agglpy,
            "params": plain(record.params, "params"),
        }
    if isinstance(record, CorrectionRecord):
        started = record.started_from
        return {
            "kind": "correction",
            "created": record.created,
            "agglpy": record.agglpy,
            "source": {
                "path": record.source.path,
                "format": record.source.format,
            },
            "started_from": (
                None
                if started is None
                else {
                    "detected": started.detected,
                    "kept": started.kept,
                    "detection_count": started.detection_count,
                }
            ),
        }
    return None
