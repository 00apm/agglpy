from typing import List


class MultipleFilesFoundError(Exception):
    """Raised when multiple files with the specified name are found."""

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
    """Raised when settings dict does no match defined schema"""

    pass


class DirectoryStructureError(Exception):
    """Raised when directory structure for Agglomerate analysis is incorrect"""

    pass


class ParticleCsvStructureError(Exception):
    """
    Raised when the structure of primary particle data .csv structure
    is not recognized
    """

    pass


class ImgDataSetBufferError(Exception):
    """
    Raised when the Primary Particle source buffer is not properly configured
    """

    pass


class ImgDataSetStructureError(Exception):
    """
    Raised when the structure of ImgDataSet object is not correct.
    """

    pass


class ImgDataSetStateError(Exception):
    """
    Raised when the state of ImgDataSet object is not correct for current operation.
    """

    pass


class AgglomerateStructureError(Exception):
    """
    Raised when the structure of Agglomerate object is not correct.
    """

    pass


# --- New code (Phase 2). Legacy exceptions above are deleted in 2.9. ---


class AgglpyError(Exception):
    """Base class of every error the new agglpy code raises on purpose."""


class ParamsError(AgglpyError, ValueError):
    """Raised when a parameter value is invalid."""


class TableError(AgglpyError, ValueError):
    """Raised when a table lacks what a function needs from it."""


class ParticleTableError(TableError):
    """Raised when a particle table does not match its schema."""


class ImageError(AgglpyError, ValueError):
    """Raised when an image array is not what a function needs.

    Preprocessing and detection take a 2-D ``uint8`` grey image, as
    ``agglpy.io.read_image`` returns it; other arrays are not converted,
    since scaling a float or 16-bit image is a choice that changes what
    is detected.
    """


class OutputExistsError(AgglpyError, FileExistsError):
    """Raised when a folder already holds an export and overwrite is off.

    The message names the files of the earlier export; nothing is
    written.
    """


class OutputLockedError(AgglpyError, PermissionError):
    """Raised when a file of an earlier export can't be replaced.

    Usually the file is open in another program (on Windows, Excel
    locks an open workbook). The message names the file; nothing is
    changed.
    """


class DuplicateParticlesWarning(UserWarning):
    """Warns about near-identical circles: a detection mistake."""


class NoParticlesWarning(UserWarning):
    """Warns that a detection found no particles at all.

    The result is still a valid, empty particle table; the message
    states the image size and its range of pixel values.
    """


class ValuesNotCountedWarning(UserWarning):
    """Warns that a distribution left values out of its classes.

    The values lie outside the size classes or are NaN; the fractions
    refer to the counted values only.
    """


class MissingImagesWarning(UserWarning):
    """Warns that images of the images table have no per-image value."""


class FileFormatError(AgglpyError, ValueError):
    """Raised when a file exists but its content can't be read.

    No or an ambiguous particle-file format, missing columns, values
    that break the particle table schema (the ``ParticleTableError`` is
    chained as the cause), an image OpenCV can't decode, an unreadable
    TIFF header. The message names the file.
    """


class PixelSizeWarning(UserWarning):
    """Warns that the pixel size read from a file may be wrong.

    The Zeiss text field and the raw value differ by more than 1 %, or
    the pixels are not square (x and y differ by more than 1 %).
    """


class NonCircularParticlesWarning(UserWarning):
    """Warns that a reader approximated or dropped non-circular shapes.

    Near-circular ovals were read as circles, or shapes that are not
    circles were left out.
    """


class ParticlesOutsideImageWarning(UserWarning):
    """Warns that particles lie entirely outside their image."""
