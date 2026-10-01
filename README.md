# agglpy: particle and agglomerate detection

> **Status: under development.** `v0.4.0` is the last release of the original API (`Manager` + `settings.yml`).
> Version 0.5 is a redesign and will break that API. The documentation is incomplete and will be written as the new
> interface stabilises.

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

Not yet on PyPI or conda-forge (planned). To use the legacy version `v0.4.0`:

```
pip install git+https://github.com/00apm/agglpy@v0.4.0
```

or create its conda environment from
[`environment-v0.4.0.yml`](environment-v0.4.0.yml) (`conda env create -f environment-v0.4.0.yml`).

## Dependencies

See [`pyproject.toml`](pyproject.toml). A full description is to be written.

## Citing

A citation file (`CITATION.cff`) and a DOI will be added. Until then, please cite the repository URL:
https://github.com/00apm/agglpy

