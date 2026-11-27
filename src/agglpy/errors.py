"""Errors and warnings of agglpy.

The errors agglpy raises on purpose subclass `AgglpyError`, most also a
built-in exception (``ValueError``, ``FileExistsError`` ...), so a
handler for the built-in one catches them too; the warnings subclass
``UserWarning``.
"""


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
    """A file to be written exists and may not be replaced.

    Raised by an export without ``overwrite=True``, by
    `Collection.create` in a folder that already has a sheet or
    ``agglpy.yml``, and where a person's correction would be replaced or
    removed without ``force=True``. The message names the files; nothing
    is written.
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
    """The particles in use no longer match their input.

    A detection made with other params or another image file, or a
    correction whose source file changed since it was imported.
    """


class CorrectionMismatchWarning(UserWarning):
    """An imported correction may not belong to its image.

    It keeps fewer than half of the detected circles, or has circles
    lying entirely outside the image.
    """
