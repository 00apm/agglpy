"""``Image``: a handle on one image of a collection."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from agglpy import _yaml
from agglpy.app._config import ImageParams
from agglpy.app._preview import Preview, make
from agglpy.app._records import CorrectionRecord, DetectionRecord, read_record
from agglpy.app._sheet import Entry, read_grid, validate
from agglpy.app._sources import resolve_stored
from agglpy.app._status import pixel_size
from agglpy.app._work import (
    CORRECTED_CSV,
    CORRECTED_RECORD,
    HCT_CSV,
    HCT_RECORD,
    image_folder,
    particle_set,
)
from agglpy.errors import NoParticleSetError
from agglpy.io import read_image, read_particles

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
    def array(self) -> NDArray[np.uint8]:
        """The image file as ``io.read_image`` returns it: whole, uncut.

        Read on each access.

        Raises:
            FileNotFoundError: If the image file is missing.
            FileFormatError: If OpenCV can't decode it.
        """
        return read_image(self._path())

    def preview(
        self, *, params: ImageParams | None = None, progress: bool = True
    ) -> Preview:
        """Detect this image in memory, to look at and tune the params.

        Reads the image file, cuts the region, runs the preprocessing
        steps, then one HCT pass per range: the circles ``detect()``
        would store with the same params. Writes nothing and emits no
        warning: what ``detect()`` would warn is in ``p.warnings``.
        Works for every image of the sheet, excluded ones too.

        Args:
            params: The params to use, e.g. another image's
                (``col["other"].params``) or another preview's
                (``p.params``); None for this image's params in
                ``agglpy.yml``. Change them with the preview's change
                methods.
            progress: Show a progress bar of the steps.

        Returns:
            The preview.

        Raises:
            ParamsError: If ``params`` is not an ``ImageParams`` or
                ``agglpy.yml`` has problems.
            FileNotFoundError: If the image file is missing.
            FileFormatError: If OpenCV can't decode it.
        """
        return make(self._collection, self._name, params, progress)

    @property
    def pixel_size(self) -> float | None:
        """The pixel size in m used for the results.

        The sheet's ``pixel_size_nm``, else the image file's, else the
        copy in a particle record (image missing or unreadable), else
        None.
        """
        collection = self._collection
        sheet = validate(read_grid(collection.sheet_path))
        for entry in sheet.entries:
            if entry.image == self._name:
                return pixel_size(collection.folder, sheet, entry)[0]
        raise KeyError(self._name)

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

    def _entry(self) -> Entry:
        sheet = validate(read_grid(self._collection.sheet_path))
        for entry in sheet.entries:
            if entry.image == self._name:
                return entry
        raise KeyError(self._name)

    def _path(self) -> Path:
        """The image file of the sheet entry, resolved."""
        return resolve_stored(self._entry().file, self._collection.folder)

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
