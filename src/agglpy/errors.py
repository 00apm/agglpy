"""Errors and warnings of agglpy.

The errors agglpy raises on purpose subclass `AgglpyError`, most also a
built-in exception (``ValueError``, ``FileExistsError`` ...), so a
handler for the built-in one catches them too; the warnings subclass
``UserWarning``. The classes before `AgglpyError` serve the legacy API.
"""

from typing import List


class MultipleFilesFoundError(Exception):
    """Several files are found where one with the name is expected.

    Parameters
    ----------
    matches : list of str
        The files found.
    filename : str, default ""
        The name searched for; ``""`` leaves it out of the message.
    """

    def __init__(
        self,
        matches: List[str],
        filename: str = "",
    ) -> None:
        self.filename = filename
        self.matches = matches
        if filename == "":
            super().__init__(f"Multiple files found: {matches}")
        else:
            super().__init__(f"Multiple files found for {filename}: {matches}")


class SettingsStructureError(Exception):
    """A legacy settings dict does not match the settings schema."""

    pass


class DirectoryStructureError(Exception):
    """A legacy working directory does not have the required layout."""

    pass


class ParticleCsvStructureError(Exception):
    """A primary particle CSV file has a structure not recognised."""

    pass


class ImgDataSetBufferError(Exception):
    """The primary particle source buffer is not configured properly."""

    pass


class ImgDataSetStructureError(Exception):
    """The structure of an ``ImgDataSet`` object is not correct."""

    pass


class ImgDataSetStateError(Exception):
    """An ``ImgDataSet`` is not in the state an operation needs."""

    pass


class AgglomerateStructureError(Exception):
    """The structure of an ``Agglomerate`` object is not correct."""

    pass


# --- New code; the legacy exceptions above go with the legacy API. ---


class AgglpyError(Exception):
    """Base class of the errors agglpy raises on purpose."""


class ParamsError(AgglpyError, ValueError):
    """A parameter value is invalid."""


class TableError(AgglpyError, ValueError):
    """A table lacks what a function needs from it."""


class ParticleTableError(TableError):
    """A particle table does not match its schema."""


class ImageError(AgglpyError, ValueError):
    """An image array is not what a function needs.

    Preprocessing and detection take a 2-D ``uint8`` grey image, as
    `agglpy.io.read_image` returns it; other arrays are not converted,
    since scaling a float or 16-bit image is a choice that changes what
    is detected.
    """


class SheetError(AgglpyError, ValueError):
    """The image sheet of a collection has invalid content.

    The message lists every problem by row and column.
    """


class OutputExistsError(AgglpyError, FileExistsError):
    """A folder already holds an export and overwrite is off.

    The message names the files of the earlier export; nothing is
    written.
    """


class OutputLockedError(AgglpyError, PermissionError):
    """A file agglpy writes can't be replaced.

    Usually the file is open in another program (on Windows, Excel
    locks an open workbook). The message names the file; nothing is
    changed.
    """


class DuplicateParticlesWarning(UserWarning):
    """Near-identical circles are found: a detection mistake."""


class NoParticlesWarning(UserWarning):
    """A detection finds no particles at all.

    The result is still a valid, empty particle table; the message
    states the image size and its range of pixel values.
    """


class ValuesNotCountedWarning(UserWarning):
    """A distribution leaves values out of its classes.

    The values lie outside the size classes or are NaN; the fractions
    refer to the counted values only.
    """


class MissingImagesWarning(UserWarning):
    """Images of the images table have no per-image value."""


class FileFormatError(AgglpyError, ValueError):
    """A file exists but its content can't be read.

    No or an ambiguous particle-file format, missing columns, values
    that break the particle table schema (the `ParticleTableError` is
    chained as the cause), an image OpenCV can't decode, an unreadable
    TIFF header. The message names the file.
    """


class PixelSizeWarning(UserWarning):
    """The pixel size read from a file may be wrong.

    The Zeiss text field and the raw value differ by more than 1 %, or
    the pixels are not square (x and y differ by more than 1 %).
    """


class NonCircularParticlesWarning(UserWarning):
    """A reader approximates or drops non-circular shapes.

    Near-circular ovals were read as circles, or shapes that are not
    circles were left out.
    """


class ParticlesOutsideRegionWarning(UserWarning):
    """Particles outside the analysed region are left out.

    Each of them lies entirely outside the region.
    """


class UnmatchedPathsWarning(UserWarning):
    """Image paths do not match the filename rule.

    Their columns were left empty.
    """


class UnreadableRecordWarning(UserWarning):
    """A particle record cannot be read.

    agglpy treats it as missing.
    """


class LossyImageWarning(UserWarning):
    """An image is JPEG-compressed.

    Compression artefacts can shift detected edges.
    """


class IncompleteRunWarning(UserWarning):
    """A batch leaves images not done.

    The returned table says which and why.
    """


class LeftoverFoldersWarning(UserWarning):
    """``agglpy/`` holds folders of images the sheet no longer lists.

    They are never moved or deleted.
    """


class NoParticleSetError(AgglpyError, FileNotFoundError):
    """An image has no particles yet.

    Neither a detection nor an imported correction.
    """


class NotReadyError(AgglpyError, ValueError):
    """Images of a collection are not ready for a step.

    The message lists every image and how to fix it.
    """


class OutdatedParticlesWarning(UserWarning):
    """Particles from an outdated detection are used.

    The detection was made with other params or another image file.
    """
