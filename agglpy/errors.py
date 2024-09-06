class MultipleFilesFoundError(Exception):
    """Raised when multiple files with the specified name are found."""
    def __init__(self, filename, matches):
        self.filename = filename
        self.matches = matches
        super().__init__(f"Multiple files found for {filename}: {matches}")