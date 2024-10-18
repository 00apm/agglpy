from typing import List


class MultipleFilesFoundError(Exception):
    """Raised when multiple files with the specified name are found."""

    def __init__(
        self,
        matches: List[str],
        filename: str = "",
    ):
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


class AgglomerateStructureError(Exception):
    """
    Raised when the structure of Agglomerate object is not correct.
    """

    pass
