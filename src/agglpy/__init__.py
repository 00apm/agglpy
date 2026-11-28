"""Primary particles and their agglomerates on microscope images.

agglpy detects spherical primary particles on SEM and TEM images with
the Hough Circle Transform, groups touching particles into agglomerates
and measures both. The entry point is `Collection`: a folder of images
with their params, detected or corrected particles and the analysis.
"""

from agglpy.app import (
    Collection,
    Drawings,
    Image,
    Preview,
    Results,
    metric_names,
)
from agglpy.core.distributions import size_classes

__all__ = [
    "Collection",
    "Drawings",
    "Image",
    "Preview",
    "Results",
    "metric_names",
    "size_classes",
]
