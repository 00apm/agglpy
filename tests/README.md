# Tests

| Folder | What it tests | Kind | Lifetime |
|---|---|---|---|
| `core/` | one file per core module of the new architecture (`test_group.py` → `agglpy.group`, `test_classify.py` → `agglpy.classify`, later `test_properties.py`, `test_stats.py`, …) | unit tests on hand-made data, answers worked out by hand | permanent |
| `golden/` | the whole analysis on two real SEM images (D7-017, D7-019), compared with recorded results | end-to-end, marked `slow` | permanent (runner changes in Phase 2) |
| `legacy/` | the v0.4 API: `Manager`, `ImgDataSet`, settings, folder layout, CSV importers | unit / integration | deleted with the legacy code (roadmap 2.9); CSV importer tests move to the new `io` module (2.2) |
| `support/` | helpers, **no tests**: data paths, synthetic cases, the neutral `Result`, adapters | — | `synthetic/legacy_adapter.py` deleted in 2.9 |
| `data/` | input images, settings, CSVs, golden expected files | — | — |

`conftest.py` holds fixtures shared by all folders (paths to the input data sets).

## Synthetic cases (`support/synthetic/`)

- `cases.py`: circle sets in px with the expected result, written as plain data.
- `result.py`: the neutral result every adapter returns (planned particle table columns).
- `transforms.py`: order / position / mirror / axis-swap transformations that must not change a result.
- `legacy_adapter.py`: runs a case through the v0.4 `ImgDataSet` code.
- `adapters.py`: `ADAPTERS` (the implementations every core test runs against) and `KNOWN_FAILURES`.

The `adapter` fixture (`core/conftest.py`) runs every core test once per entry in `ADAPTERS`. Known bugs of an
implementation are listed in `KNOWN_FAILURES` and become strict xfails for that implementation only.

## Running

```
pytest                         # everything
pytest -m "not slow"           # skip the golden tests while working
pytest tests/core              # one folder
pytest tests/core -rx          # also list expected failures and their reasons
pytest tests/golden --force-regen   # after an INTENDED change of results; review the diff
```

Test files live in folders without `__init__.py`; `pyproject.toml` sets `--import-mode=importlib` and
`pythonpath = ["tests"]`, so helpers are imported as `from support.… import …`.
