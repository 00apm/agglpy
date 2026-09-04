# agglpy: particle and agglomerate detection
**agglpy** is a tool for primary particle and agglomerate detection on digital microscopic (SEM/TEM) images.
It combines several digital image processing and particle classification algorithms (including image filtering, edge/shape detection, and Hough transform, DFS and Union-Find) to describe the structure of each individual agglomerate.

## Methodology

**agglpy** detects spherical primary particles and its agglomerates in several steps:

1. Input image preprocessing,
2. Edge detection and binarization,
3. Primary particle detection using the Hough Circle Transform (HCT),
4. (Optional) Manual correction of missed and incorrectly detected primary particles,
5. Primary particle connectivity determination,
6. Grouping of connected particles into agglomerates.

See [docs/methodology.md](docs/methodology.md) for the full description of the method.

## Documentation

See [docs/index.md](docs/index.md).


## Installation


## Dependencies




## Citing


