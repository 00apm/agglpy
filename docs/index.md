# agglpy documentation

agglpy detects spherical primary particles on microscope (SEM/TEM) images with the Hough Circle Transform, groups
touching particles into agglomerates and measures both: sizes, shapes, contacts, structure and population metrics.
These pages describe agglpy 0.5 (development version); the parameters of every function are in its docstring
(`help(col.detect)`).

- [Getting started](getting-started.md) — install, then one walk-through from images to exported results.
- [Collection](collection.md) — the folder agglpy works in: the image sheet, `agglpy.yml`, particle sets, status
  labels, correcting in Fiji, which calls write files.
- [Results](results.md) — what `col.analyze()` returns: the tables, their columns and units, metrics, size
  distributions, subsets, drawings, export.
- [Methodology](methodology.md) — the method: preprocessing, detection, manual correction, connectivity, hidden
  particles, sizes and population metrics.
- [Structure of agglomerates](agglomerate-structure.md) — cores, links, loops, dangling ends, branches and the
  longest chain of each agglomerate (`res.with_structure()`).

Recipes (tested scripts in [`examples/`](../examples)): agglomerate types, distribution plots, structure averages.

For contributors:

- [Docstring standard](contributing/docstrings.md) — numpydoc and the rules agglpy adds.
