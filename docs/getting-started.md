# Getting started

From a folder of SEM images to exported tables, in the order of the work. Each step links to the page that
describes it in full; the parameters of every call are in its docstring (`help(col.detect)`).

## Install

agglpy 0.5 is not on PyPI yet; it is installed from the `develop` branch. With conda (the environment file pins
Python 3.11 and the dependencies from conda-forge):

```
git clone https://github.com/00apm/agglpy
cd agglpy
git checkout develop
conda env create
conda activate agglpy
pip install -e . --no-deps
```

`--no-deps` keeps pip from adding a second OpenCV next to conda's. With pip only:

```
pip install "agglpy[notebook] @ git+https://github.com/00apm/agglpy@develop"
```

The `notebook` extra adds zoomable figures in Jupyter (`%matplotlib widget`); without it, figures show as PNGs.

## 1. Create a collection

A collection is a folder holding the image sheet (`images.xlsx`, one row per image), `agglpy.yml` (the
detection params) and `agglpy/` (agglpy's work). The images stay where they are. `columns_from_path` turns parts
of each file's path into sheet columns, here the subfolder as `series`:

```python
from agglpy import Collection

col = Collection.create(
    "study", images="sem", pattern="**/*.tif",
    columns_from_path=r"(?P<series>[^/]+)/",
)
```

Edit `images.xlsx` in Excel, or from Python: fill `pixel_size_nm` for images whose files carry no pixel size,
and add any columns to group by later.

```python
sheet = col.read_sheet()
sheet["pixel_size_nm"] = 2.5
col.write_sheet(sheet)
```

Later, `col.add_images(...)` (same arguments) appends only files not yet in the sheet, and
`Collection.open("study")` reopens the folder. The collection object holds no state: every call reads the files
again, so editing them by hand between calls is safe. See [Collection](collection.md#the-image-sheet).

## 2. Tune the detection

A new `agglpy.yml` has no HCT ranges, so nothing is detected yet. A preview detects one image in memory and
writes nothing; each change method returns a new preview and reruns only the steps it changes.

```python
p = col["A-01"].preview()
p = p.change_region(bottom_fraction=0.1)              # cut off the data bar
p = p.change_preprocess(median_blur=3, rolling_ball={"radius": 40})
p = p.set_ranges([(16, 44)])                          # circle diameters in px
p.circles()                                           # the circles over the image
```

Other views: `p.image()`, `p.stage("preprocessed", side_by_side=True)` (what HCT sees), `p.edges(1)` (the edges
of range 1). To judge a change, compare two previews: new circles green, lost ones red.

```python
q = p.change_hct(range=1, accumulator_threshold=10)
q.circles(compare=p)
```

Store the result as the defaults for every image, then check another image with them:

```python
col.set_params(None, region=p.params.region,
               preprocess=p.params.preprocess, hct=p.params.hct)
col["B-02"].preview().circles()
```

`q.keep()` stores a preview's params for its own image only (only what differs from the defaults is written).
Params and their layout: [Collection](collection.md#params-agglpyyml) and the docstrings of
`agglpy.params.HCTRange` and `PreprocessParams`; the method: [Methodology](methodology.md).

## 3. Detect

`detect()` runs the detection on every included image and writes `agglpy/<name>/particles_hct.csv` with a record
of the params and the image it came from. An image whose record still matches is skipped, so after a change of
params only the affected images run again.

```python
report = col.detect()      # one row per image: outcome, particle_count, warnings
col.status()               # pixel size, particle set and detection state of each image
```

`status()` writes nothing; its `detection` column says whether each detection is `up to date` or why not
(`params changed`, `image changed`, ...). See [Collection](collection.md#status).

## 4. Correct (optional)

Automatic detection misses or adds circles. Correct them in Fiji's ROI Manager, save the ROI list as CSV and
import it; from then on the image uses the correction instead of the detection, and `detect()` never touches it.

```python
col.import_particles("A-02", "fiji/A-02.csv")
col.status()[["image", "particle_set", "correction"]]
```

How to draw the ROIs and what the import checks: [Collection](collection.md#correcting-in-fiji).

## 5. Choose the images

The sheet's `include` column decides which images `analyze()` uses. `exclude` sets it false and records the
reason in `note`; both accept a name, a list of names, or a True / False Series on the rows of `status()`.

```python
col.exclude("B-04", note="charging")
status = col.status()
col.exclude(status.particle_count < 5, note="too few particles")
col.include("B-04")        # back in; its note stays
```

## 6. Analyze

`analyze()` groups the particles of every included image into agglomerates and measures both. It first checks
that every included image has a pixel size and particles, and otherwise lists the fixes without computing
anything. It writes nothing; run it again after any change.

```python
res = col.analyze()
res.images          # one row per image: pixel size, field of view, sheet columns
res.particles       # one row per particle: image, id, x, y, r, D, agglomerate_id, ...
res.agglomerates    # one row per agglomerate: member_count, D, area, touches_edge, ...
res.contacts        # one row per possible contact between two particles
```

All tables are in SI units (m, m², m³). For other units add a column:
`res.particles["D_nm"] = res.particles.D * 1e9`. Every column: [Results](results.md#units-and-keys).

## 7. Summarize

`summary` pools the images of each group (a sheet column such as `series`, or `"image"`);
`summary_across_images` computes each metric per image, then the mean and confidence interval over the images.

```python
res.summary(by="series")
res.summary_across_images(by="series")
```

Size distributions take the class edges in m; `size_classes` builds the usual ones.

```python
import agglpy

edges = agglpy.size_classes(20e-9, 160e-9, step=2, scale="log")
res.distribution("particles", "D", edges, by="series")
res.distribution("agglomerates", "D", edges, weights="volume")
```

`subset` leaves out images or agglomerates and is noted in the export record; `with_structure` adds cores,
links, branches and the longest chain of each agglomerate. Both return new results.

```python
inner = res.subset(agglomerates=~res.agglomerates.touches_edge, label="not on the edge")
st = res.with_structure()
st.summary(by="series", metrics="structure")
```

Metrics and their definitions: [Results](results.md#metrics); structure:
[Structure of agglomerates](agglomerate-structure.md).

## 8. Draw and export

`draw` colours the agglomerates (or particles, `what="particles"`) by a column, with one colour scale over all
images; `d["A-01"]` makes the figure. `export` writes the tables to `results.xlsx` and `csv/`, with the units and
a `record.yml` of where every number came from. Use one folder per export:

```python
d = res.draw(color_by="member_count")
d["A-01"]
folder = col.folder / "exports" / "2026-11 series A vs B"
res.export(folder, tables={"summary": res.summary(by="series")})
d.export(folder / "drawings")
```

Exporting into a folder that already holds an export needs `overwrite=True`. A relative folder is taken from the
current working directory, not from the collection. See [Results](results.md#export).

## Next

- [Collection](collection.md): the folder, the sheet, `agglpy.yml`, particle sets, status labels, Fiji
  correction, and which calls write files.
- [Results](results.md): the tables, units, metrics, distributions, subsets, drawings and export.
- [Methodology](methodology.md): how particles are detected and grouped.
- [Structure of agglomerates](agglomerate-structure.md): cores, links, loops, dangling ends and branches.
- Recipes in [`examples/`](../examples): [agglomerate types](../examples/agglomerate_types.py) (collector /
  similar / separate), [plotting a size distribution](../examples/plot_distribution.py),
  [averages of agglomerate structure](../examples/agglomerate_structure.py).
