"""``Image``: a handle on one image of a collection."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import pandas as pd

from agglpy import _yaml
from agglpy.app._config import ImageParams
from agglpy.app._records import CorrectionRecord, DetectionRecord, read_record
from agglpy.app._work import (
    CORRECTED_CSV,
    CORRECTED_RECORD,
    HCT_CSV,
    HCT_RECORD,
    image_folder,
    particle_set,
)
from agglpy.errors import NoParticleSetError
from agglpy.io import read_particles

if TYPE_CHECKING:
    from agglpy.app._collection import Collection

__all__ = ["Image"]

R = TypeVar("R", DetectionRecord, CorrectionRecord)


class Image:
    """One image of a collection, made by ``col[name]``.

    Holds only the collection and the name: every attribute reads the
    files again, so the handle always shows what is on disk.
    """

    def __init__(self, collection: "Collection", name: str) -> None:
        self._collection = collection
        self._name = name

    @property
    def name(self) -> str:
        """The image name, as in the sheet."""
        return self._name

    @property
    def params(self) -> ImageParams:
        """The image's params (``col.params(name)``)."""
        return self._collection.params(self._name)

    @property
    def particle_set(self) -> Literal["corrected", "hct"] | None:
        """The particle set in use: the correction if there is one."""
        return particle_set(self._folder)

    @property
    def particles(self) -> pd.DataFrame:
        """The particle table (px) of the set in use.

        Raises:
            NoParticleSetError: If the image has no particles yet.
        """
        used = self.particle_set
        if used is None:
            raise NoParticleSetError(
                f"{self._name} has no particles: run detect() or "
                "import_particles()"
            )
        name = CORRECTED_CSV if used == "corrected" else HCT_CSV
        return read_particles(self._folder / name)

    @property
    def detection(self) -> dict[str, Any] | None:
        """The record of ``particles_hct.csv`` as plain data, or None.

        None when there is no detection or its record does not read.
        """
        return self._record(HCT_CSV, HCT_RECORD, DetectionRecord)

    @property
    def correction(self) -> dict[str, Any] | None:
        """The record of ``particles_corrected.csv`` as plain data, or
        None."""
        return self._record(CORRECTED_CSV, CORRECTED_RECORD, CorrectionRecord)

    def __repr__(self) -> str:
        return f"Image({self._name!r}, particle_set={self.particle_set!r})"

    @property
    def _folder(self) -> Path:
        return image_folder(self._collection.folder, self._name)

    def _record(
        self,
        csv: str,
        record: str,
        kind: type[R],
    ) -> dict[str, Any] | None:
        path = self._folder / record
        # a record counts only next to its CSV and when agglpy reads it
        if not (self._folder / csv).is_file():
            return None
        if read_record(path, kind) is None:
            return None
        data: dict[str, Any] = _plain(_yaml.load(path))
        return data


def _plain(value: Any) -> Any:
    """YAML data as built-in dicts, lists, str, int, float, bool."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, bool) or value is None:
        return value
    for kind in (str, int, float):
        if isinstance(value, kind):
            return kind(value)
    return value
