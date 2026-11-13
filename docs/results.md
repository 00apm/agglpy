# Results

`col.analyze()` returns a `Results`: four tables (`images`, `particles`, `agglomerates`, `contacts`) and the methods
that summarise, subset, draw and export them. It is a snapshot of the collection's files at the time of the call:
change a particle file or the sheet, and `analyze()` again. Lengths, areas and volumes are in SI units (m, m², m³).
`analyze()` and every method of `Results` write nothing, except `export()` and `Drawings.export()`.

```python
res = col.analyze()
res.summary(by="series")
```

How the particles and agglomerates are found is in [Methodology](methodology.md); the collection folder, the
sheet and the parameters in [Collection](collection.md).

## Units and keys

- **Units.** Each image is converted from px to m with its own `pixel_size` before the images are pooled: lengths
  × p, areas × p², volumes × p³. Counts, ratios and flags have no unit. Other units: add a column,
  `res.particles["D_nm"] = res.particles.D * 1e9`.
- **Keys.** `image` joins every table. A particle is `(image, id)`, an agglomerate `(image, agglomerate_id)`. Ids
  are unique only within an image and change with a new detection or correction. The index is `0 … n-1`.
- **Own columns.** The tables are the stored frames, not copies: columns you add are seen by every method,
  `subset`, `draw` and `export`. Don't change rows or agglpy's columns; take a `subset` instead.
- **Joins.** User columns onto particles: `res.particles.merge(res.images[["image", "series"]], on="image")`;
  each particle with its agglomerate: `res.particles.merge(res.agglomerates, on=["image", "agglomerate_id"],
  suffixes=("", "_agglomerate"))`.

**Circle model.** Every particle is a sphere whose projection is its detected circle; sizes, volumes and surfaces
are those of the spheres. Nothing is measured on the grey values after detection.

**In 2D.** The column tables below say what kind of number each value is against the true 3D value:

| kind | meaning |
|---|---|
| exact | the quantity itself: a 2D quantity, a count of rows, or a size under the sphere model |
| lower bound | the true value can only be at or above it |
| upper bound | the true value can only be at or below it |
| model | rests on a stated assumption (spheres; deposition from no preferred direction) |
| description | biased in either direction; for comparing samples measured alike, not for quoting as the truth |
| label | bookkeeping (ids, names, flags); nothing to compare with |

No kind includes detection errors (missed or false circles, radius errors, a contact decided within about 1 px):
every number inherits them. Two effects of the projection run through all tables:

- **Possible contacts.** Overlapping or touching circles may touch in 3D, or lie one behind the other. agglpy
  counts them all as *possible contacts* ([methodology §6](methodology.md#6-connectivity-criterion-and-agglomerate-classification)):
  contact counts are upper bounds. Two agglomerates that overlap in the projection are one agglomerate.
- **Hidden particles.** A particle completely behind a larger one is not in the image (more often behind large
  collectors). Counts and volumes miss them; only the `*_with_hidden`
  columns correct for them, by a model
  ([methodology §7](methodology.md#7-hidden-primary-particles-enclosed-particles-and-the-with-hidden-values)).

## Images table

`res.images`: one row per analysed image (rows with `include` true), in sheet order: agglpy's columns, the
sheet's user columns, then `note`.

| column | unit | meaning | in 2D |
|---|---|---|---|
| `image` | text | the image's name in the sheet | label |
| `pixel_size` | m | length of one pixel side | exact (as calibrated) |
| `pixel_size_source` | text | `images.xlsx` / `images.csv` (the sheet's `pixel_size_nm`) or `image file` (its tags) | label |
| `fov_area` | m² | field of view: area of the analysed region | exact |
| `width`, `height` | px | size of the whole image file | exact |
| `particle_set` | text | `hct` (detection) or `corrected` (imported correction) | label |
| `outside_region` | count | circles of the particle file lying entirely outside the region, left out | exact |
| user columns, `note` | as in the sheet | copied from the sheet; group by them with `by=` | label |

A wrong pixel size scales every length, area and volume of its image: check `pixel_size_source`. The analysed
region cuts the data bar off the bottom ([methodology §1](methodology.md#1-input-image-preprocessing)); a large
`outside_region` hints at a wrong `bottom_fraction`. Comparing `hct` with `corrected` images compares detection
quality too.

## Particle table

`res.particles`: one row per primary particle, `image` first.

| column | unit | meaning | in 2D |
|---|---|---|---|
| `id` | — | the particle's number in its particle file | label |
| `x`, `y` | m | circle centre; origin at the centre of the top-left pixel | exact |
| `r` | m | circle radius = sphere radius | exact (sphere model) |
| `agglomerate_id` | — | the agglomerate the particle belongs to; a single particle has its own | label |
| `enclosed` | True / False | the circle lies completely inside a larger circle: the particle is on its visible side | label |
| `coordination` | count | the particle's possible contacts | upper bound |
| `D` | m | diameter, $2r$; the size of the particle size distribution | exact (sphere model) |
| `area` | m² | circle area, $\pi r^2$ (projected) | exact |
| `volume` | m³ | sphere volume, $\frac{4}{3}\pi r^3$ | model |
| `surface` | m² | sphere surface, $\pi D^2$ (not the projected `area`) | model |
| `touches_edge` | True / False | the circle reaches the analysed region's edge | exact |

`x`, `y` back in px: divide by the image's `pixel_size`. A particle at the edge counts in full everywhere,
though it may be cut off. Enclosed particles are members everywhere (every count, sum and statistic); only the
`*_with_hidden` columns count them twice. An enclosed particle inside a nearly identical circle is a duplicate
detection (warned at detection), still counted.

**Structure columns.** `res.with_structure()` returns new results with `segment_type`, `segment_id`,
`branch_depth`, `branch_length`, `tip`, `dangling_end_depth` and `dangling_end_length` added: see
[Structure of agglomerates](agglomerate-structure.md#columns).

## Agglomerate table

`res.agglomerates`: one row per agglomerate, single particles included (an agglomerate of one member), `image`
first. Definitions of size and shape: [methodology §8](methodology.md#8-agglomerate-size-and-shape).

| column | unit | meaning | in 2D |
|---|---|---|---|
| `agglomerate_id` | — | the agglomerate's number within its image | label |
| `member_count` | count | primary particles, enclosed ones included | description |
| `enclosed_count` | count | enclosed members | label |
| `contact_count` | count | possible contacts between members; at least `member_count − 1` | upper bound |
| `loop_count` | count | `contact_count − member_count + 1`: contacts beyond those that hold it together | upper bound (loose) |
| `volume` | m³ | sum of member sphere volumes | model; low by hidden particles |
| `D` | m | volume-equivalent diameter, $(6V/\pi)^{1/3}$; the size of the aerosol size distribution | model; low by hidden particles |
| `surface` | m² | sum of member sphere surfaces | model |
| `D_mean`, `D_std` | m | mean and sample std of member diameters (`D_std` NaN for one member) | exact for the visible members |
| `D_largest` | m | largest member diameter (the collector, if any) | exact for the visible members |
| `size_ratio` | — | second largest / largest member diameter (NaN for one member) | exact for the visible members |
| `volume_with_hidden` | m³ | `volume` with enclosed members counted twice | model |
| `D_with_hidden` | m | volume-equivalent diameter of `volume_with_hidden` | model |
| `member_count_with_hidden` | count | `member_count + enclosed_count` | model |
| `surface_with_hidden` | m² | `surface` with enclosed members counted twice | model |
| `x_com`, `y_com` | m | volume-weighted centre of mass | exact (projection) |
| `rg` | m | radius of gyration of the member spheres, heights unknown | lower bound |
| `area` | m² | projected area: exact area of the union of member circles | exact |
| `D_pa` | m | diameter of the circle with that area, $\sqrt{4A/\pi}$ | exact |
| `D_feret_x`, `D_feret_y` | m | Feret (caliper) diameter along the image x / y axis | exact |
| `D_feret_max` | m | largest Feret diameter | exact; lower bound of the 3D extent |
| `touches_edge` | True / False | any member reaches the analysed region's edge | exact |

- `member_count` is low by the hidden particles and high where separate agglomerates overlap in the image.
- `surface` loses nothing where spheres touch: for sintered or fused particles it is an upper bound.
- `loop_count` is a loose bound: every false contact closes a loop, so a collector carrying separate chains can
  show many loops it does not have.
- `area` and the Feret diameters describe the circle model, not the outline the image shows. `D_feret_x` /
  `_y` depend on how the agglomerate lies; for shape, prefer `D_feret_max`.
- The `*_with_hidden` columns assume deposition from no preferred direction and ignore particles hidden behind
  the outline. No metric uses them.
- An agglomerate that `touches_edge` may continue outside the image. Large agglomerates touch the edge more
  often, so leaving them out shifts the size distribution towards small sizes:
  `res.subset(agglomerates=~res.agglomerates.touches_edge)`.
- agglpy does not classify agglomerates (collector, similar, …); `size_ratio` is the building block of such a
  rule (recipe `examples/agglomerate_types.py`).

**Structure columns.** `res.with_structure()` adds `longest_chain`, `longest_chain_length`, `branch_count`,
`branch_depth_max`, `branch_length_max`, `core_count`, `core_member_count`, `small_core_member_count`,
`link_count`, `dangling_end_count`, `dangling_end_member_count`, `dangling_end_depth_max` and
`dangling_end_length_max`: see [Structure of agglomerates](agglomerate-structure.md#columns).

## Contacts table

`res.contacts`: one row per possible contact, images in the order of the results, then sorted by
`agglomerate_id, id_a, id_b`; an agglomerate has
`contact_count` rows. No units.

| column | unit | meaning | in 2D |
|---|---|---|---|
| `image`, `agglomerate_id` | — | where the pair lies; both particles are in this agglomerate | label |
| `id_a`, `id_b` | — | the two particle ids; `id_a` the larger (equal radii: the lower id) | upper-bound set |
| `inside` | True / False | `id_b` lies inside `id_a` | label |

Every true contact is a row; many rows are not true contacts. A pair that cannot touch (an inner particle inside
a middle one inside an outer one) is left out
([methodology §6](methodology.md#6-connectivity-criterion-and-agglomerate-classification)). Pairs per image:
`res.contacts.groupby("image").size()`.

## Metrics

`res.summary(by=None)` gives one row per group of images: the `by` columns, `n_images` (images in the group,
those without particles included), then the metric families in the order below. `by` is `None` (all images),
`"image"` or columns of `res.images`; `metrics=` picks families (see `help(res.summary)`).

`summary` **pools** the images of a group: counts are summed, ratios computed from the summed counts, sizes from
the pooled rows. Ratios are never means of per-image ratios
([methodology §9](methodology.md#9-population-metrics-and-size-distributions)). Every metric uses the visible
values only. A group without particles gives zero counts and NaN ratios and sizes.

```python
res.summary(by="series", metrics=["counts", "ratios"])
```

**counts** (no unit)

| column | meaning | formula | in 2D |
|---|---|---|---|
| `N_primary` | primary particles | particle rows | lower bound |
| `N_aerosol` | aerosol particles: agglomerates and single particles | agglomerate rows | description |
| `N_pp1` | single particles | agglomerates with `member_count` 1 | description |
| `N_ppA` | primary particles in agglomerates | $N_{primary} - N_{pp1}$ | description |
| `N_aggl` | agglomerates of two or more particles | $N_{aerosol} - N_{pp1}$ | description |
| `N_merges` | merges that joined the primaries into aerosol particles | $N_{primary} - N_{aerosol}$ | description |

`N_merges` depends only on the grouping: false contacts inside an agglomerate do not change it.

**ratios** (from the pooled counts)

| column | meaning | formula | in 2D |
|---|---|---|---|
| `Ra` | agglomeration ratio of Gotoh et al. ([1996](methodology.md#references)); 0 … 0.5, falls when agglomerates merge | $N_{aggl} / N_{primary}$ | description |
| `agglomerated_fraction` | number fraction of primaries bound in agglomerates | $1 - N_{pp1} / N_{primary}$ | description |
| `merge_ratio` | degree of agglomeration: 0 all single, 1 all in one; grows with every merge; NaN below 2 particles | $N_{merges} / (N_{primary} - 1)$ | description |
| `n_ppA` | mean primaries per agglomerate (singles left out) | $N_{ppA} / N_{aggl}$ | description |
| `n_ppP` | mean primaries per aerosol particle (singles included) | $N_{primary} / N_{aerosol}$ | description |

Exactly: `agglomerated_fraction = merge_ratio · (N_primary − 1) / N_primary + Ra`.

**contacts** (from `contact_count`; no unit)

| column | meaning | formula | in 2D |
|---|---|---|---|
| `N_contacts` | possible contacts | $\sum$ `contact_count` | upper bound |
| `N_loops` | contacts beyond those that hold the agglomerates together | $N_{contacts} - N_{merges}$ | upper bound (loose) |
| `coordination_mean` | mean possible contacts per particle | $2 N_{contacts} / N_{primary}$ | upper bound |

**structure** (only on results from `res.with_structure()`; without it, asking for the family raises an error)

| column | meaning | formula | in 2D |
|---|---|---|---|
| `N_cores` | cores | $\sum$ `core_count` | description |
| `N_small_cores` | agglomerates with a small core | agglomerates with `small_core_member_count` > 0 | description |
| `N_links` | link runs | $\sum$ `link_count` | description |
| `N_dangling_ends` | dangling ends | $\sum$ `dangling_end_count` | description (usually low) |
| `core_fraction`, `small_core_fraction` | share of primaries in cores / small cores | particles of that `segment_type` / $N_{primary}$ | upper bound |
| `link_fraction`, `loop_fraction` | share of primaries in links / loops | as above | description |
| `dangling_end_fraction` | share of primaries in dangling ends | as above | lower bound |
| `no_core_fraction` | share of primaries in agglomerates with nothing looped | as above | description |

The six fractions add up to 1. Segments and their caveats: [Structure of agglomerates](agglomerate-structure.md).

**per_area** (over the summed `fov_area` of the group)

| column | unit | meaning | formula |
|---|---|---|---|
| `coverage` | — | covered fraction of the substrate | $\sum$ agglomerate `area` / $\sum$ `fov_area` |
| `N_primary_per_area` | m⁻² | primary particles per area | $N_{primary} / \sum$ `fov_area` |
| `N_aerosol_per_area` | m⁻² | aerosol particles per area | $N_{aerosol} / \sum$ `fov_area` |
| `N_aggl_per_area` | m⁻² | agglomerates per area | $N_{aggl} / \sum$ `fov_area` |

Agglomerates never share area, so `coverage` is the exact union of all circles (exact in 2D). Circles crossing
the region's edge count in full, so all four are biased high for large agglomerates on small images. A
deposition rate: divide by a sampling time column of the sheet.

**particle_size** (over the particle `D`; m)

| column | meaning | formula |
|---|---|---|
| `particle_D_mean`, `particle_D_std` | number mean and sample std of the diameters | |
| `particle_D10`, `particle_D50`, `particle_D90` | number-based 10 / 50 / 90 % quantiles | linear interpolation between sorted values |
| `particle_SMD` | Sauter mean diameter | $\sum D^3 / \sum D^2$ |

Of the visible particles (exact under the sphere model); the hidden ones are not in it.

**aerosol_size** (over the agglomerate `D`, single particles included; m)

| column | meaning | formula |
|---|---|---|
| `aerosol_D_mean`, `aerosol_D_std` | number mean and sample std of the volume-equivalent diameters | |
| `aerosol_D10`, `aerosol_D50`, `aerosol_D90` | number-based 10 / 50 / 90 % quantiles | linear interpolation |

Model: low by the hidden particles, high where overlapping agglomerates are joined.

**member_count** (over `member_count` of every aerosol particle; no unit)

| column | meaning | formula |
|---|---|---|
| `aerosol_member_count_std` | sample std of the member counts | the mean is `n_ppP` |
| `aerosol_member_count_q10`, `_q50`, `_q90` | 10 / 50 / 90 % quantiles of the member counts | linear interpolation |

Description, as `member_count`.

## Across images

`res.summary_across_images(by="series")` computes every metric per image, then for each group and metric the
mean over the images, its sample std, the number of images with a value and a t-based confidence interval
(95 % by default; see `help(res.summary_across_images)`). Each image weighs equally: this describes the typical
image and how well it is known; `summary` describes the pooled population. A ratio pooled and a ratio averaged
over images differ when the images differ in size.

Long table, one row per group and metric:

| column | meaning |
|---|---|
| the `by` columns | the group |
| `metric` | the metric's name |
| `mean`, `std` | mean and sample std over the images |
| `n` | images with a value (NaN values left out, per metric) |
| `ci_low`, `ci_high` | $\bar{x} \pm t_{(1+c)/2,\,n-1}\, s/\sqrt{n}$; NaN when `n` is 1 |

`res.values_across_images(per_image, by=...)` does the same for your own values: a table with `image` and one
row per image (more keys with `keys=`; see `help(res.values_across_images)`). An image without a row gets NaN,
with a warning; when the value is a count, add those images with 0.

```python
per_image = res.agglomerates.groupby("image", as_index=False).touches_edge.mean()
res.values_across_images(per_image, by="series")
```

## Size distributions

`res.distribution(table, column, edges, by=None)` counts any numeric column of `"particles"` or `"agglomerates"`
over size classes, per group of images: the particle size distribution (particle `D`), the aerosol size
distribution (agglomerate `D`), member counts or any other column. `agglpy.size_classes` builds linear or log
edges; any increasing array works. Options (closed side, weights): `help(res.distribution)`.

```python
edges = agglpy.size_classes(10e-9, 640e-9, step=2, scale="log")
res.distribution("particles", "D", edges, by="series", weights="volume")
```

Long table, one row per group, basis and class:

| column | meaning |
|---|---|
| the `by` columns | the group |
| `left`, `mid`, `right`, `width` | the class; `(left, right]` by default, the outer edge always counted |
| `basis` | `number`, then each weight column (e.g. `volume`) |
| `amount` | count per class, or sum of the weight column |
| `fraction` | amount / the group's counted total; adds up to 1 |
| `cumulative` | running sum of `fraction`; ends at 1 |
| `density` | `fraction / width`, per unit of the column (per m for `D`) |
| `log_density` | `fraction / (log10 right − log10 left)`, $dN/d\log D$; NaN where `left` ≤ 0 |

Values outside the classes and NaN values are not counted (warned); fractions refer to the counted values. A
value within $10^{-9}$ (relative) of an edge counts as on it. Every class appears in every group; a group
without values has zero amounts and NaN fractions. Plotting: recipe `examples/plot_distribution.py`.

`res.distribution_across_images(...)` gives each class's fraction per image, then mean ± CI over the images of
each group: the class columns, `basis`, `metric` (`fraction`) and the columns of [Across images](#across-images).
An image without counted values is left out of the means and of `n`.

## Subsets

`res.subset(images=..., agglomerates=...)` returns new results with part of the rows; `res` stays as it is.

```python
res.subset(images=res.images.series == "a")
res.subset(agglomerates=~res.agglomerates.touches_edge, label="no edge")
```

- `images` keeps those images with their particles, agglomerates and contacts.
- `agglomerates` keeps whole agglomerates with their members and contacts, and every image: an image with nothing
  left counts as zeros in the metrics.
- `images` takes a name, a list of names or a True / False Series on `res.images`' index; `agglomerates` a
  True / False Series on `res.agglomerates`' index. The new tables are numbered `0 … n-1`.
- Each step (with its `label`) goes into the export record.
- Structure columns stay valid after a subset (agglomerates are kept whole); `segment_id` keeps its gaps.
  `subset` first, then `with_structure()`, numbers the segments from 0.

Options and errors: `help(res.subset)`. A subset is a view of one run; which images `analyze()` uses at all is
the sheet's `include` column ([Collection](collection.md)).

## Drawings

`res.draw(what="agglomerates", color_by="D")` returns `Drawings`: a read-only mapping image name → matplotlib
`Figure`, in the order of the results. `what` is `"particles"` (coloured by a particle column) or
`"agglomerates"` (by an agglomerate column); a column of `res.images` has to be merged onto the table first.

```python
d = res.draw(what="particles", color_by="segment_type")   # on res.with_structure()
d["D7-017"]
d.export()
```

- Each figure shows the analysed region in px over the image file (`full_image=True`: the whole image).
- One colour scale over all drawn images: numbers share one range, labels one colour each, so images compare at
  a glance; draw a series with `res.subset(images=...).draw(...)`.
- A figure is made when looked at (`d[name]`) or exported, from the rows copied at `draw()`; the image file is
  read then. A notebook shows the first 4.
- `d.export(folder=None)` writes one `<image>.png` per image (72 dpi; with `legend=False` pixel for pixel over the
  region), by default into `exports/drawings` in the collection. Every check comes before the first write. No
  record: export next to `res.export()` of the same results.

Options (selection, image opacity, colours, legend): `help(res.draw)`.

## Export

`res.export(folder=None)` writes the tables, their units and a record into a folder (created if missing) and
returns it. `None` gives `exports/results` in the collection; a relative folder is taken from the current
directory; a repeated export needs `overwrite=True`, so a folder named for each export is recommended. Only the files below are written or replaced; other files in the folder
are never touched.

```python
res.export("exports/2026-11 series a", tables={"summary": res.summary(by="series")})
```

| file | content |
|---|---|
| `results.xlsx` | sheets `images`, `agglomerates`, `particles`, `contacts`, the extra `tables`, then `units` and `info` |
| `csv/` | `<table>.csv` for every table and `units.csv`: UTF-8, comma separated, `.` as decimal mark, no index column |
| `record.yml` | what the numbers were made from (below) |

- **Units.** Column names stay as in the tables; the `units` sheet and `csv/units.csv` list `table | column |
  unit` (`m`, `m²`, `m³`; `px` for `width` / `height`). Extra tables get no units.
- **Table names.** Extra tables are named with letters, digits, `_` and `-`, at most 31 characters (Excel's
  sheet limit).
- **Large tables.** A table longer than Excel's 1,048,575 rows is written as CSV only; the record says so.
- **Index.** A table whose index is not `0 … n-1` (e.g. a `groupby` result) is written with its index as columns.
- **record.yml**: the agglpy version, export time, versions of the main dependencies and the tables written;
  when and on which collection and sheet `analyze()` ran; per image the file, size, pixel size and its source,
  region, circles left outside it, the particle set and file (count, sha256) and the detection or correction
  record it comes from, with its warnings; the rows with `include` false; the subset steps; the
  `with_structure()` settings. The `info` sheet holds the same as `key | value` rows.
- A file open in another program stops the export before anything is changed; a write that fails part-way
  removes the files it wrote.

Options and errors: `help(res.export)`.
