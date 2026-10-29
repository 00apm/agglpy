"""The application layer: collections of images on disk.

A collection is a folder with an image sheet (what to analyse) and
``agglpy.yml`` (how). The files are the only state: every call reads
them again.
"""

from agglpy.app._collection import Collection
from agglpy.app._config import ImageParams
from agglpy.app._image import Image
from agglpy.app._results import Drawings, Results

__all__ = ["Collection", "Drawings", "Image", "ImageParams", "Results"]
