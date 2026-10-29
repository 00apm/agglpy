# Tests

| Folder | What it tests | Kind | Lifetime |
|---|---|---|---|
| `foundation/` | the foundation modules of the new architecture: `test_params.py` → `agglpy.params`, `test_tables.py` → `agglpy.tables`, `test_logger.py` → `agglpy.logger`, `test_errors.py` → `agglpy.errors`, `test_yaml.py` → `agglpy._yaml` (comment-keeping YAML) | unit tests | permanent |
| `core/` | one file per core module of the new architecture (`test_agglomerates.py` → `agglpy.core.agglomerates`, `test_properties.py` → `agglpy.core.properties`, `test_metrics.py` → `agglpy.core.metrics`, `test_distributions.py` → `agglpy.core.distributions`, `test_region.py` → `agglpy.core.region`, `test_preprocess.py` → `agglpy.core.preprocess`, `test_detect.py` → `agglpy.core.detect` (with a contract test on drawn discs), `test_classify.py` → the recipe `examples/agglomerate_types.py` (no classification in the library)) | unit tests on hand-made data, answers worked out by hand | permanent |
| `pipeline/` | the pipeline layer `agglpy.pipeline`: `test_analyze.py` → `analyze`, `ImageResult`; `test_detect_particles.py` → `detect_particles` | unit tests on hand-made tables and drawn images | permanent |
| `io/` | the io edge `agglpy.io`: `test_images.py` → `read_image`, `test_metadata.py` → `read_image_metadata` and the pixel-size readers, `test_particle_files.py` → `read_particles`, `write_particles` | unit tests on the real images and CSVs in `data/` and on tiny synthetic files (`support/tiffs.py`, `tmp_path`) | permanent |
| `draw/` | the draw edge `agglpy.draw`: `test_figure.py` → `figure`, `test_particles.py` → `particles` with numeric colours, `test_labels.py` → label colours and legends, `test_agglomerates.py` → `agglomerates`, `test_color_scale.py` → `color_scale` (one scale over several images) | pixel checks on figures saved as PNG (`support/render.py`) | permanent |
| `export/` | the export edge `agglpy.export`: `test_write_results.py` → `write_results` | files written to `tmp_path` and read back | permanent |
| `app/` | the application layer `agglpy.app`: `test_config.py` → params conversion and the `agglpy.yml` template, `test_config_file.py` → reading, checking and editing `agglpy.yml`, `test_sheet.py` → reading and validating the image sheet, `test_sheet_writer.py` → writing it back in place, `test_sources.py` → image files and the filename rule, `test_collection.py` → `Collection`, `test_records.py` → particle records, `test_detect.py` → `detect`, `test_import.py` → `import_particles`, `test_status.py` → `status`, `test_image.py` → `col[name]`, `test_analyze.py` → `analyze` (readiness, the tables in SI units, warnings, progress), `test_results.py` → `Results` statistics, `subset` and `export`, `test_drawings.py` → `res.draw`, `Drawings` and their PNG export; `conftest.py` → collections of drawn disc TIFFs (`analysis_col`: four images, three ready for `analyze`) | unit tests on files in `tmp_path` (sheets from `support/sheets.py`, disc images from `support/collection.py`) | permanent |
| `recipes/` | the recipes in `examples/` (repo root), their functions and each one run as a script: `test_agglomerate_types.py` → `agglomerate_types.py`, `test_plot_distribution.py` → `plot_distribution.py` | unit tests on hand-made data | permanent |
| `golden/` | the whole analysis on two real SEM images (D7-017, D7-019), compared with recorded results: the pipeline on the automatic and the corrected path (`test_pipeline_golden.py`, expected files in `data/golden/expected/pipeline/`, px) and the legacy automatic path (`test_golden.py`, until the legacy code is removed) | end-to-end, marked `slow` | permanent |
| `legacy/` | the v0.4 API: `Manager`, `ImgDataSet`, settings, folder layout, CSV importers | unit / integration | deleted with the legacy code (roadmap 2.9); the CSV importer tests have their new-API version in `io/test_particle_files.py` |
| `support/` | helpers, **no tests**: data paths, synthetic TIFFs (`tiffs.py`), images of drawn discs (`discs.py`) and collections of them (`collection.py`), image sheets as xlsx / CSV (`sheets.py`), PNG round trip of a figure (`render.py`), synthetic cases, the neutral `Result`, adapters (legacy and core) | — | `synthetic/legacy_adapter.py` deleted in 2.9 |
| `data/` | input images, settings, CSVs, golden expected files | — | — |

`conftest.py` holds fixtures shared by all folders (paths to the input data sets).

## Synthetic cases (`support/synthetic/`)

- `cases.py`: circle sets in px with the expected result, written as plain data.
- `result.py`: the neutral result every adapter returns (planned particle table columns).
- `transforms.py`: order / position / mirror / axis-swap transformations that must not change a result.
- `legacy_adapter.py`: runs a case through the v0.4 `ImgDataSet` code.
- `core_adapter.py`: runs a case through the new core (agglomerates, properties, the summary in physical units; types through the recipe).
- `adapters.py`: `ADAPTERS` (run a circle case through an implementation), `STATS` (its size-class functions, called
  directly) and `KNOWN_FAILURES`.

The `adapter` fixture (`core/conftest.py`) runs every core test once per entry in `ADAPTERS`. Each adapter
declares the checks it supports (`SUPPORTED_CHECKS`); a test of a check that an adapter doesn't support yet is
skipped with the roadmap item that brings it (`pytest -rs` lists them). Known bugs of an implementation are listed
in `KNOWN_FAILURES` and become strict xfails for that implementation only.

## Running

```
pytest                         # everything
pytest -m "not slow"           # skip the golden tests while working
pytest tests/core              # one folder
pytest tests/core -rx          # also list expected failures and their reasons
pytest tests/golden --force-regen   # after an INTENDED change of results; review the diff
```

Test files live in folders without `__init__.py`; `pyproject.toml` sets `--import-mode=importlib` and
`pythonpath = ["tests", "."]`, so helpers are imported as `from support.… import …` and recipes as
`from examples.… import …`.
