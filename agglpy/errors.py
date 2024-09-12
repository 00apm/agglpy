from typing import List
class MultipleFilesFoundError(Exception):
    """Raised when multiple files with the specified name are found."""
    def __init__(self, matches: List[str], filename: str ="", ):
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