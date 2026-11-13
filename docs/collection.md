# Collection

A collection is the folder agglpy works in: a sheet listing the images (what to analyse), a params file (how)
and a work folder with the particles of each image. `agglpy.Collection` is a handle on that folder.

**The files are the only state.** The `Collection` object holds only the folder path; every call reads the files
again. A sheet edited in Excel and a call from the notebook never conflict, and a collection opened after a
restart is exactly as it was left. The walk-through is in [Getting started](getting-started.md); this page is the
reference.

```python
import agglpy
col = agglpy.Collection.create("ash-2026", images="raw/")   # new folder
col = agglpy.Collection.open("ash-2026")                    # existing one: checks both files
```

## The folder

| Path | Written by | Content |
|---|---|---|
| `images.xlsx` or `images.csv` | `create`, sheet calls, you | [The image sheet](#the-image-sheet): one row per image |
| `agglpy.yml` | `create`, `set_params`, `reset_params`, `p.keep()`, you | [Params](#params-agglpyyml): defaults and per-image differences |
| `agglpy/<image>/` | `detect`, `import_particles` | The [particle sets](#particle-sets-detection-and-correction) of one image and their records |
| `exports/results`, `exports/drawings`, `exports/tuning` | `export(...)` without a folder | Default export folders |

The image files stay where they are: the sheet stores their paths. A folder holds one collection: `create`
refuses a folder that already has a sheet or `agglpy.yml` (`OutputExistsError`). Keep exactly one sheet: both
`images.xlsx` and `images.csv` in one folder is a `SheetError`. `agglpy/` belongs to agglpy; an export into it is
refused.

## The image sheet

One row per image, one column per fact. Five columns are fixed; any other column is a *user column* (series,
temperature, operator, ...): it is carried into the results' `images` table, where it can group metrics.

| Column | Meaning | Example |
|---|---|---|
| `image` | Name of the image: unique (ignoring case), usable as a folder name | `D7-017` |
| `file` | Path of the image file, relative to the collection folder, `/`-separated | `../raw/D7-017.tif` |
| `include` | Whether `analyze()` uses the image; empty = included | `TRUE`, `false`, `no`, `0` |
| `pixel_size_nm` | Pixel size in **nm**; empty = read from the image file ([Pixel size](#pixel-size)) | `2.5` |
| `note` | Free text, e.g. why an image is excluded | `charging streaks` |
| *any other* | User column | `series` = `S1` |

Only `image` and `file` are required. Rules checked on every read (all problems are listed in one `SheetError`,
by row and column):

- `image`: no `< > : " / \ | ? *` or control characters, no trailing dot or space, no Windows device name (`CON`,
  `NUL`, `COM1`, ...).
- `include`: `TRUE`/`FALSE` cells, `true`/`false`, `yes`/`no`, `1`/`0` or empty.
- `pixel_size_nm`: a number from 0.001 to 10 000, `.` as decimal point.
- `images.xlsx`: one worksheet, no formulas (paste values instead). `images.csv`: UTF-8.
- Reserved names, not allowed for user columns: `pixel_size`, `pixel_size_source`, `fov_area`, `width`,
  `height`, `particle_set`, `outside_region` (agglpy adds them to the results' `images` table).

**Adding images.** `create(..., images=)` and `col.add_images(...)` take folders or files. A folder gives its
image files (`.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`, `.bmp`, any case) directly inside it; `pattern=` takes
globs instead, e.g. `"**/*.tif"` for subfolders. The name is the file stem; `columns_from_path=` (a regular
expression with named groups) fills user columns from the path and can set the name with the group `image`. Files
already in the sheet are skipped, and so are the collection's own `agglpy/` and `exports/` folders. See `help(col.add_images)`.

```python
col.add_images("raw/batch2", columns_from_path=r"(?P<series>S\d+)_(?P<image>[^/]+)\.tif$")
```

**Editing.** Edit the sheet by hand (close it in Excel before agglpy writes it), or with `read_sheet()` →
change the DataFrame → `write_sheet(table)`. agglpy edits the file in place: only changed cells are set, so
formatting, widths and filters stay; charts and pictures embedded in `images.xlsx` are not kept.
`col.include(...)` and `col.exclude(..., note=)` set the `include` cell. `col.remove_images(...)` removes the row,
the image's entry in `agglpy.yml` and its work folder, never the image file.

**Renaming an image** by hand means renaming its work folder `agglpy/<name>/` and its entry under `images:` in
`agglpy.yml` too; `status()` warns about a work folder no row owns (`LeftoverFoldersWarning`).

## Params (`agglpy.yml`)

The params say how each image is detected: the analysed region, the preprocessing and the Hough Circle Transform
(HCT) ranges. All lengths are in **px**. `defaults` apply to every image; `images` holds, per image, only what
differs from the defaults.

```yaml
defaults:
  region:
    bottom_fraction: 0.1
  preprocess:
    median_blur: 3
    rolling_ball: {radius: 50}
    clahe: null
  hct:
    ranges:
    - {d_min: 8, d_max: 30}
    - {d_min: 30, d_max: 120, accumulator_threshold: 25}
images:
  D7-019:
    preprocess:
      rolling_ball: null
```

Merging: an image's mapping is merged into the defaults key by key; a list (`ranges`, `tile_grid`) replaces the
default list as a whole; `null` switches a step off. `hct: {ranges: []}` (the template's value) means detection
is not set: `detect()` skips the image. Image names YAML reads as numbers must be quoted (`'007':`).

| Key | Neutral value | Meaning (details: `help()` of the class) |
|---|---|---|
| `region.bottom_fraction` | `0.0` | Fraction of the height cut off at the bottom (SEM data bar); `RegionParams` |
| `preprocess.median_blur` | `null` | Odd kernel size ≥ 3; `PreprocessParams` |
| `preprocess.rolling_ball` | `null` | `radius`, `light_background` (false), `presmooth` (true); `RollingBallParams` |
| `preprocess.clahe` | `null` | `clip_limit`, `tile_grid` `[columns, rows]`; `ClaheParams` |
| `hct.ranges` | `[]` | One HCT pass per entry: `d_min`, `d_max` (required), `min_dist_ratio` (0.4), `canny_threshold` (200), `accumulator_threshold` (15); `HCTRange` |

The classes are in `agglpy.params`; the method behind each step is in [Methodology](methodology.md), sections
1–3. Write params by hand (comments are kept when agglpy writes the file), with `col.set_params(...)` (a mapping
merges into what is there; `None` as the image sets the defaults) and `col.reset_params(...)`, or from a tuning
preview with `p.keep()`. `col.params(name)` and `col[name].params` give the merged params of one image.

```python
col.set_params(None, hct={"ranges": [{"d_min": 8, "d_max": 30}]})
col.set_params("D7-019", preprocess={"rolling_ball": None})
```

Every image is checked before anything is written; a problem is a `ParamsError` naming its key path, e.g.
`images.D7-019.hct.ranges[0]`.

## Pixel size

Particles are detected and stored in px; `analyze()` converts each image with its own pixel size, so every
results table is in SI units (see [Results](results.md)). The pixel size of an image comes from the first of:

| Source | `pixel_size_source` in `status()` | Note |
|---|---|---|
| Sheet column `pixel_size_nm` | `images.xlsx` / `images.csv` | In nm; overrides the file |
| Image file tags | `image file` | Zeiss `CZ_SEM`, Thermo/FEI `PixelWidth`, or an ImageJ calibration ([Methodology](methodology.md#9-population-metrics-and-size-distributions), "Units") |
| The copy in a particle record | `image file` | Only when the image file is missing or does not read |
| None | `none` | `analyze()` refuses the image |

`PixelSizeWarning` marks a doubtful value read from a file (Zeiss
text and raw value differ, non-square pixels). `col[name].pixel_size` gives the value in **m**.

## Particle sets: detection and correction

Each image has up to two particle sets in its work folder, each a CSV with a record next to it:

| File in `agglpy/<image>/` | Written by | Content |
|---|---|---|
| `particles_hct.csv` | `col.detect()` | Circles found by HCT: `id, x, y, r` in px |
| `particles_hct_record.yml` | `col.detect()` | Image (hash, size, time, width, height, pixel size), params, versions, warnings |
| `particles_corrected.csv` | `col.import_particles(...)` | The corrected circles, rewritten as `id, x, y, r` in px |
| `particles_corrected_record.yml` | `col.import_particles(...)` | Image, source file (path, format, hash), the detection it started from, versions, warnings |

**The set in use** is decided by the CSV files alone: `particles_corrected.csv` if it exists, else
`particles_hct.csv`. `col[name].particle_set` names it, `col[name].particles` reads it. `detect()` never touches a
correction; a correction is never reported as outdated.

**Records** are what agglpy compares the CSV and the image with. Do not edit them. A CSV whose hash no longer
matches its record is *edited by hand*: agglpy keeps it and does not overwrite it without `force=True`. Read a
record with `col[name].detection` / `col[name].correction` (plain dicts); `col[name].detection_params` gives the
params of the stored detection, for `col[name].preview(params=...)`.

**Detection outcomes** (the `outcome` column of the table `detect()` returns; see `help(col.detect)`):

| Outcome | Meaning |
|---|---|
| `detected` | Detected and written |
| `replaced` | `force=True` replaced a `particles_hct.csv` edited by hand, without a record or unreadable |
| `skipped` | Detection not set (`ranges: []`), the image has a correction, or up to date (same params, same image file) |
| `kept` | `particles_hct.csv` is edited by hand, has no record or does not read; kept |
| `error` | This image failed (message in the table); its files are unchanged, the other images go on |

Warnings of each image go into its record and the `warnings` column, and each warning class is raised once after
the batch. `IncompleteRunWarning` says that images ended `kept`, `error` or `skipped` for a reason other than
"up to date". JPEG images give `LossyImageWarning`.

## Coordinates

Particle files hold `id, x, y, r` in **px**, with pixel centres at integers: pixel `i` covers `[i - 0.5, i + 0.5)`,
as in OpenCV, NumPy and scikit-image. `x` runs to the right, `y` down, and the origin is the top-left corner of the
whole image file, data bar included, whatever region detection used. A circle that crosses the image edge, or has
its centre outside, is a partly visible particle and stays in the file; `analyze()` leaves out only circles lying
entirely outside the analysed region (`ParticlesOutsideRegionWarning`, counted in `outside_region`).

ImageJ and Fiji count from pixel edges, so a circle read from Fiji is shifted by −0.5 px. `agglpy.io.read_particles`
does the conversion; the details are in [Methodology](methodology.md#4-manual-correction), "Coordinates" and
"Reading a Fiji correction".

## Correcting in Fiji

The manual correction itself (what to add, what to remove) is described in
[Methodology](methodology.md#4-manual-correction). In the collection:

1. Open the image file itself (uncut, so the coordinates match) in Fiji and draw one oval ROI per particle in the
   ROI Manager.
2. List the ROIs in the ROI Manager and save the list as CSV. The columns agglpy reads are
   `Index, Type, X, Y, Width, Height`; other columns are ignored.
3. Import it:

```python
col.import_particles("D7-017", "fiji/D7-017.csv")
```

The file is read (format recognised from the header; `format=` chooses one), rewritten as
`agglpy/D7-017/particles_corrected.csv` and used from then on; the source file is only read. Ovals whose width
and height differ by at most 1 px are read as circles (`r = (Width + Height)/4`), other shapes are left out, both
reported in one `NonCircularParticlesWarning`. Ids are the numbers Fiji shows. An agglpy particle file
(`id, x, y, r`) imports the same way. Importing again replaces the correction; a correction edited by hand needs
`force=True`. See `help(col.import_particles)`.

The record notes how many detected circles the correction kept (centres within 2 px, radii within 10 %). A
correction that keeps fewer than half of them is shown as `not from detection`.

## Status

`col.status()` returns one row per sheet image, worked out from the files now. It writes nothing.

| Column | Content |
|---|---|
| `image`, `include` | From the sheet |
| `file` | `ok`, `missing` or `unreadable` |
| `pixel_size_nm`, `pixel_size_source` | The value used, in nm, and where it comes from ([Pixel size](#pixel-size)) |
| `particle_set`, `particle_count` | `corrected`, `hct` or `none`, and its number of circles |
| `outside_image` | Circles lying entirely outside the image |
| `detection`, `correction` | Labels below |
| `kept` | For a correction: `k of n detected` |
| `warnings` | Warning classes in the record of the set in use |

| `detection` | Meaning | What to do |
|---|---|---|
| `not set` | `hct.ranges` is empty | Set ranges (`set_params`, or tune and `p.keep()`) |
| `not detected` | Ranges set, no `particles_hct.csv` | `col.detect()` |
| `up to date` | Same params, same image file | Nothing |
| `params changed` | `agglpy.yml` changed since the detection | `col.detect()`; `analyze()` warns with `OutdatedParticlesWarning` until then |
| `image changed` | The image file is not the one detected | `col.detect()`; check the `file` path |
| `edited by hand` | `particles_hct.csv` differs from its record | Keep it, or `detect(force=True)` |
| `no record` | The record is missing or does not read | `detect(force=True)` |
| `unreadable` | `particles_hct.csv` does not read | `detect(force=True)` |

| `correction` | Meaning | What to do |
|---|---|---|
| *(empty)* | No correction | — |
| `current detection` | Started from the `particles_hct.csv` present now | Nothing |
| `older detection` | Started from a detection since replaced or removed | Check the correction still fits |
| `not from detection` | No detection at import, or it kept fewer than half of the detected circles | Check it is the right file |
| `edited by hand` | `particles_corrected.csv` differs from its record | Import the corrected file again |
| `no record` | The record is missing or does not read | Import again |
| `unreadable` | `particles_corrected.csv` does not read | Import again, or delete it |

`status()` and `analyze()` read the same facts: `analyze()` refuses (`NotReadyError`, listing every image and the fix) while an
included image has no pixel size, no particles, or a missing or unreadable file. Select images from the table:

```python
st = col.status()
col.detect(st.detection == "params changed")
```

## Which calls write files

Only methods that change the collection, and `export()` of a returned object, write files. Each checks
everything first and writes nothing when a check fails; a file held open by another program gives
`OutputLockedError` and stays as it was.

| Call | Writes |
|---|---|
| `Collection.create` | The sheet, `agglpy.yml` (template), empty `agglpy/` |
| `add_images`, `write_sheet`, `include`, `exclude` | The sheet |
| `remove_images` | The sheet, `agglpy.yml`, deletes `agglpy/<image>/` (a correction only with `force=True`) |
| `set_params`, `reset_params`, `p.keep()` | `agglpy.yml` |
| `detect` | `agglpy/<image>/particles_hct.csv` + record |
| `import_particles` | `agglpy/<image>/particles_corrected.csv` + record |
| `res.export(folder)` | `results.xlsx`, `csv/`, `record.yml` (default `exports/results`) |
| `res.draw(...).export(folder)` | One PNG per image (default `exports/drawings`) |
| `p.export(folder)` | One PNG per preview view + `<image> preview.yml` (default `exports/tuning`) |

Read only: `Collection.open`, `read_sheet`, `params`, `status`, `analyze`, `col[name]` and its attributes,
`col[name].preview(...)` and its change methods, `res.draw(...)`. An export never overwrites an earlier one
without `overwrite=True`; a relative export folder is taken from the notebook's working folder.
