"""Pipeline: the core steps joined for one image.

Plain functions over one image's data, in px: no loops over images, no
files, no figures. Loops, progress and file handling belong to the
layer above.
"""

from agglpy.pipeline._analysis import ImageResult, analyze

__all__ = ["ImageResult", "analyze"]
