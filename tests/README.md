# Tests

| Folder | What it tests | Kind |
|---|---|---|
| `foundation/` | the foundation modules: `test_params.py` → `agglpy.params`, `test_tables.py` → `agglpy.tables`, `test_logger.py` → `agglpy.logger`, `test_errors.py` → `agglpy.errors`, `test_yaml.py` → `agglpy._yaml` (comment-keeping YAML) | unit tests |
| `core/` | one file per core module: `test_agglomerates.py` → `agglpy.core.agglomerates`, `test_properties.py` → `agglpy.core.properties`, `test_structure.py` → `agglpy.core.structure`, `test_metrics.py` → `agglpy.core.metrics`, `test_distributions.py` → `agglpy.core.distributions`, `test_region.py` → `agglpy.core.region`, `test_preprocess.py` → `agglpy.core.preprocess`, `test_detect.py` → `agglpy.core.detect`: `detect_hct` (with a contract test on drawn discs) and `hct_edges` | unit tests on hand-made data, answers worked out by hand |
| `pipeline/` | the pipeline layer `agglpy.pipeline`: `test_analyze.py` → `analyze`, `ImageResult`; `test_detect_particles.py` → `detect_particles` and the detection steps a preview reuses (`cut_region`, `preprocess_stages`, `hct_pass`, `particles_from_passes`) | unit tests on hand-made tables and drawn images |
| `io/` | the io edge `agglpy.io`: `test_images.py` → `read_image`, `test_metadata.py` → `read_image_metadata` and the pixel-size readers, `test_particle_files.py` → `read_particles`, `write_particles` | unit tests on the real images and particle files in `data/` and on tiny synthetic files (`support/tiffs.py`, `tmp_path`) |
| `draw/` | the draw edge `agglpy.draw`: `test_figure.py` → `figure`, `figure_pair`, `image`, `test_particles.py` → `particles` with numeric colours, `test_labels.py` → label colours and legends (also `legend`), `test_agglomerates.py` → `agglomerates`, `test_color_scale.py` → `color_scale` (one scale over several images), `test_notebook.py` → `agglpy._notebook` (figures shown in Jupyter: PNG, ipympl canvas); `conftest.py` → the Agg backend | pixel checks on figures saved as PNG (`support/render.py`) |
| `export/` | the export edge `agglpy.export`: `test_write_results.py` → `write_results` | files written to `tmp_path` and read back |
| `app/` | the application layer `agglpy.app`: `test_config.py` → params conversion and the `agglpy.yml` template, `test_config_file.py` → reading, checking and editing `agglpy.yml`, `test_sheet.py` → reading and validating the image sheet, `test_sheet_writer.py` → writing it back in place, `test_sources.py` → image files and the filename rule, `test_collection.py` → `Collection`, `test_records.py` → particle records, `test_detect.py` → `detect`, `test_import.py` → `import_particles`, `test_status.py` → `status`, `test_image.py` → `col[name]`, `test_analyze.py` → `analyze` (readiness, the tables in SI units, warnings, progress), `test_results.py` → `Results` statistics, `subset` and `export` (with the default export folders), `test_drawings.py` → `res.draw`, `Drawings` and their PNG export, `test_preview.py` → `img.preview`, `Preview` (params, change methods and what they rerun, the card, `keep`, `export`), `test_preview_views.py` → its views and comparisons; `conftest.py` → collections of drawn disc TIFFs (`analysis_col`: four images, three ready for `analyze`; `tuning_col`: three images with two HCT ranges, for previews) | unit tests on files in `tmp_path` (sheets from `support/sheets.py`, disc images from `support/collection.py`) |
| `recipes/` | the recipes in `examples/` (repo root), their functions and each one run as a script: `test_agglomerate_types.py` → `agglomerate_types.py` (also on the synthetic classification cases), `test_plot_distribution.py` → `plot_distribution.py` | unit tests on hand-made data |
| `golden/` | the whole analysis on two real SEM images (D7-017, D7-019), compared with recorded results: the pipeline on the automatic and the corrected path (`test_pipeline_golden.py`, expected files in `data/golden/expected/pipeline/`, px) | end-to-end, marked `slow` |
| `support/` | helpers, **no tests**: data paths (`paths.py`), synthetic TIFFs (`tiffs.py`), images of drawn discs (`discs.py`) and collections of them (`collection.py`), image sheets as xlsx / CSV (`sheets.py`), PNG round trip of a figure (`render.py`), hand-built agglomerates for the structure tests (`structure_cases.py`), the synthetic cases (`synthetic/`, below) | — |
| `data/` | `images/`: four real SEM images (D5-015, D6-027, D7-017, D7-019); `particles/`: particle files in the formats `read_particles` reads (Fiji CSVs, the agglpy 0.4 and older agglpy formats); `golden/expected/pipeline/`: the golden tests' expected files | — |

## Synthetic cases (`support/synthetic/`)

- `cases.py`: circle sets in px with the expected result, written as plain data.
- `run.py`: `run_case` runs a case through the core (agglomerates, properties, summary; types through the recipe) and
  keys the answers by the case's circle names.
- `transforms.py`: order / position / mirror / axis-swap transformations that must not change a result.

## Running

```
pytest                         # everything
pytest -m "not slow"           # skip the golden tests while working
pytest tests/core              # one folder
pytest tests/golden --force-regen   # after an INTENDED change of results; review the diff
```

Test files live in folders without `__init__.py`; `pyproject.toml` sets `--import-mode=importlib` and
`pythonpath = ["tests", "."]`, so helpers are imported as `from support.… import …` and recipes as
`from examples.… import …`.
