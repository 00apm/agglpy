"""The work folder of one image and the state of its particle files."""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar

import pandas as pd

from agglpy.app._records import (
    CorrectionRecord,
    DetectionRecord,
    read_record,
    sha256,
)
from agglpy.errors import FileFormatError
from agglpy.io import read_particles

__all__ = [
    "CORRECTED_CSV",
    "CORRECTED_RECORD",
    "HCT_CSV",
    "HCT_RECORD",
    "WORK_FOLDER",
    "ParticleFile",
    "image_folder",
    "inspect",
    "particle_set",
]

WORK_FOLDER = "agglpy"
HCT_CSV = "particles_hct.csv"
HCT_RECORD = "particles_hct_record.yml"
CORRECTED_CSV = "particles_corrected.csv"
CORRECTED_RECORD = "particles_corrected_record.yml"

State = Literal["matching", "edited by hand", "no record", "unreadable"]
R = TypeVar("R", DetectionRecord, CorrectionRecord)


@dataclass(frozen=True)
class ParticleFile:
    """A particle CSV against its record.

    Attributes:
        state: ``matching``, ``edited by hand`` (its hash differs from
            the record's), ``no record`` (missing or unreadable) or
            ``unreadable`` (the CSV itself).
        record: The record, None when missing or unreadable.
        particles: The particle table, None when the CSV is unreadable.
        sha256: The CSV's hash.
    """

    state: State
    record: DetectionRecord | CorrectionRecord | None
    particles: pd.DataFrame | None
    sha256: str


def image_folder(collection: Path, name: str) -> Path:
    """The work folder of image ``name``: ``agglpy/<name>/``."""
    return collection / WORK_FOLDER / name


def inspect(
    csv: Path, record_path: Path, kind: type[R]
) -> ParticleFile | None:
    """The state of ``csv`` against its record; None without the CSV.

    The reader's own warnings (duplicates, ImageJ shapes) are dropped:
    they belong to the record of the run that wrote the file.

    Warns:
        UserWarning: If the CSV does not read.
        UnreadableRecordWarning: If the record does not read.
    """
    if not csv.is_file():
        return None
    digest = sha256(csv)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            particles = read_particles(csv)
    except FileFormatError as exc:
        warnings.warn(
            f"{csv.name} in {csv.parent.name}/ does not read: {exc}",
            UserWarning,
            stacklevel=2,
        )
        record = read_record(record_path, kind)
        return ParticleFile("unreadable", record, None, digest)
    record = read_record(record_path, kind)
    if record is None:
        return ParticleFile("no record", None, particles, digest)
    state: State = (
        "matching" if record.particles.sha256 == digest else "edited by hand"
    )
    return ParticleFile(state, record, particles, digest)


def particle_set(folder: Path) -> Literal["corrected", "hct"] | None:
    """The particle set in use, decided by the CSV files alone."""
    if (folder / CORRECTED_CSV).is_file():
        return "corrected"
    if (folder / HCT_CSV).is_file():
        return "hct"
    return None
