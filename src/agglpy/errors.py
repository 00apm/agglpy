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


class ParticleTableError(AgglpyError, ValueError):
    """Raised when a particle table does not match its schema."""


class DuplicateParticlesWarning(UserWarning):
    """Warns about near-identical circles: a detection mistake."""
