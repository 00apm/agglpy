# Methodology

agglpy implements a method for detecting spherical primary particles in Scanning Electron Microscopy (SEM) images and classifying them into agglomerates. The method was developed for analyzing aerosol particles (e.g. fly ash) collected onto microscope specimen stubs, but is applicable to any image where particles can reasonably be approximated as circles.


## Overview

The analysis proceeds through seven stages:

1. Input image preprocessing,
2. Edge detection and binarization,
3. Primary particle detection using the Hough Circle Transform (HCT),
4. (Optional) Manual correction of missed and incorrectly detected primary particles,
5. Primary particle connectivity determination,
6. Grouping of connected particles into agglomerates,
7. Estimating the primary particles hidden on the far side of larger particles.

## 1. Input image preprocessing

Raw SEM images are noisy, due to the electron beam signal acquisition process, particle surface roughness and resistivity, electron beam scattering on gas molecules in the SEM chamber, and local electric charge accumulation on high-resistivity samples (visible as locally overexposed regions or bright horizontal streaks along the scan direction).

Two corrections are applied:

- **Median filtering** — a 3×3 pixel mask is scanned across the image; the 9 pixel values under the mask are sorted and the median replaces the central pixel's value. This reduces noise while preserving edge sharpness.
- **Background normalization (Rolling-Ball algorithm)** — simulates a ball of a chosen radius rolled under the image intensity surface. The minimum intensity the ball can reach at each point forms a background estimate, which is subtracted from the original image. This removes slow-varying illumination/charging artifacts while preserving sharp particle features. The ball radius sets the scale of background structure that gets corrected and needs to be chosen relative to particle size.

Codebase: `img_process.preprocess_img` (crop, median blur, CLAHE), `img_process.rolling_ball_substraction` (rolling-ball).

## 2. Edge detection

The preprocessed grayscale image is converted to a binary edge image using the **Canny edge detector**:

1. Intensity gradient (magnitude and direction) is computed with the Sobel operator.
2. Local maxima of the gradient magnitude are identified as candidate edges.
3. Hysteresis thresholding tracks and connects edges within a threshold range, avoiding fragmentation in low-contrast regions.

The output is a binary matrix: 1 for edge pixels, 0 for background.

## 3. Shape detection — Hough Circle Transform (HCT)

Primary particles are detected by fitting circles to the edge image. A circle is parameterized as:

$$(x-a)^2+(y-b)^2=r^2$$

where $(a, b)$ is the center and $r$ the radius. The goal is to recover $(a, b, r)$ for each primary particle.

agglpy uses OpenCV's HCT implementation, based on a two-stage Hough Transform. Rather than the traditional approach of voting for every possible center $(a, b)$ around each edge pixel (expensive), it constrains the search using the gradient direction already computed during Canny edge detection: since a circle's center lies along the gradient direction at an edge pixel, only points along that gradient-defined line are evaluated as candidate centers. This substantially reduces the accumulator search space.

Each candidate center is accumulated in a voting matrix per radius; peaks in the accumulator indicate the most probable circle centers. The radius is then refined by fitting the circle equation to the surrounding edge pixels.

Codebase: `img_process.HCT` / `img_process.HCT_multi`.

**Limitation:** HCT approximates every detected feature as a circle. Near-spherical particles are represented well; particles that deviate significantly from spherical shape are poorly represented or missed entirely. This is addressed in the next stage.

## 4. Manual correction

The operator reviews the automatic detection result and:

- adds primary particles that were missed (e.g. due to low contrast or overlapping particles),
- removes/excludes detections that do not meaningfully approximate the true particle shape (irregular, non-spherical particles, or artifacts from substrate defects).

In the published study this was done in ImageJ; within agglpy this corresponds to the manual review/edit step between automatic HCT detection and the agglomerate classification stage.

## 5. Closest-neighbor finding

Determining which particles are connected requires comparing each particle against candidate neighbors. Checking every pair is expensive for images with many particles, so agglpy narrows the candidate set first.

- Primary particles are indexed in a 2D KD-tree (SciPy implementation, after Maneewongvatana & Mount).
- Particles are processed **from largest to smallest radius**. For each particle, only neighbors whose centers lie within a distance of **2× its radius** are considered — this guarantees that any true connection to a larger-or-equal particle is found without needing to compare against every particle in the image, since we always evaluate outward from the larger member of a pair first.

## 6. Connectivity criterion and agglomerate classification

Two primary particles $P_i$ and $P_j$ are considered connected when their circles intersect or one lies entirely within the other:

$$s \le r_{P_i} + r_{P_j}$$

where $s$ is the distance between the two particle centers and $r_{P_i}$, $r_{P_j}$ are their radii.

> This is a geometric simplification — a real connection may be over- or under-detected purely from circle overlap (see the article's discussion of this limitation). Treat it as an approximation, not a ground-truth adjacency test.

Once pairwise connections are known, agglomerates are formed by a **recursive/DFS-style grouping** over the connectivity graph:

1. Take the first unclassified primary particle and start a new agglomerate.
2. Add every particle directly connected to it.
3. For each newly added particle, recursively add its own unvisited connections, until no new particles are reachable.
4. Remove all particles now assigned to this agglomerate from the pool and repeat from step 1 for the remaining particles.

A particle with no connections becomes a single-particle "agglomerate." These are not true agglomerates — which is why agglpy's output is more accurately described as *aerosol particle data* rather than strictly *agglomerate data*.

Codebase: `aggl.py` (`Particle`/`Agglomerate` construction and classification into `"collector" | "attached2coll" | "separate" | "similar"`, driven by `Manager.collector_threshold`).

## 7. Hidden primary particles: enclosed particles and the with-hidden values

An SEM image shows only one side of each particle. Primary particles deposited on the far side of a larger particle (in particular of a collector) are not visible, so the number and volume of primary particles in an agglomerate are underestimated. agglpy gives a simple estimate of what is hidden.

**Enclosed particles.** A primary particle $P_j$ whose circle lies completely inside the circle of another particle $P_i$,

$$s \le r_{P_i} - r_{P_j},$$

is seen in front of $P_i$, i.e. on its visible side. Such particles are flagged as *enclosed*. (Partly overlapping particles, at the outline of the larger one, are not enclosed.)

**With-hidden values.** If deposition has no preferred direction (random, chaotic deposition), primary particles are spread uniformly over the whole surface of the larger particle. The hidden side then carries, statistically, the same particles as the visible side. The visible-side particles are the enclosed ones, so each enclosed particle is counted twice:

$$V_{hidden} = V + \sum_{j \in enclosed} V_{P_j}, \qquad N_{hidden} = N + N_{enclosed}, \qquad D_{hidden} = \left(\frac{6 V_{hidden}}{\pi}\right)^{1/3},$$

where $V$ is the summed volume of all member spheres (enclosed ones included once), $N$ the member count and $D_{hidden}$ the volume-equivalent diameter of the corrected agglomerate.

This is a deliberately simple estimate. It assumes isotropic deposition and ignores particles hidden behind the outline (only the enclosed particles are doubled). More precise estimates may exist and could replace it.

Codebase: `agglpy.core.agglomerates.find_enclosed` flags enclosed particles; `agglpy.core.properties.agglomerate_properties` gives `enclosed_count`, `volume_with_hidden`, `D_with_hidden` and `member_count_with_hidden`. agglpy 0.4 called them `idj` and `*_dsom`.

## 8. Agglomerate size and shape

All agglomerate properties describe the circle model: each primary particle is a sphere whose projection is its detected circle. They are computed in pixels and converted to physical units once, where images are pooled.

**Size.** $D$ is the volume-equivalent diameter, $D = (6V/\pi)^{1/3}$ with $V$ the summed member volume; it is the basis of the agglomerate size distribution. `D_mean`, `D_std`, `D_largest` and `size_ratio` (second largest / largest member diameter) describe the members.

**Surface.** `surface` is the summed sphere surface of the members, $\pi D^2$ each. Spheres touching in one point lose no surface (the model of the volume), so for sintered or fused particles it is an upper bound. `surface_with_hidden` counts enclosed members twice (section 7).

**Projected area.** `area` is the exact area of the union of the member circles: overlaps count once, an enclosed particle adds nothing. It is computed geometrically (the uncovered arcs of each circle, Green's theorem), not on a pixel grid and not on the image, so it describes the circle model, not the agglomerate's outline as the image shows it. $D_{pa} = \sqrt{4A/\pi}$ is the diameter of the circle with the same area.

**Feret diameters.** `D_feret_x` and `D_feret_y` are the widths along the image axes, exact for circles; over many randomly oriented agglomerates a fixed direction gives the classical statistical diameter (Walton, 1948). `D_feret_max` is the largest width in any direction, $\max (d_{ij} + r_i + r_j)$ over member pairs.

**Centre of mass and radius of gyration** (`x_com`, `y_com`, `rg`): members weighted by volume. The centre of mass is the exact projection of the 3D one; `rg` leaves out the unknown height differences, so it is a lower bound of the 3D radius of gyration (exact for a single sphere).

## 9. Population metrics and size distributions

Metrics describe a group of images: one image, a user group (a column of the images table, e.g. a condition) or all images. An image without particles is a result: it keeps its row with zero counts.

**Pooled.** `summary` pools the images of a group: counts are summed, ratios are computed from the summed counts and size descriptors from the pooled particles or agglomerates. Ratios are never means of per-image ratios: two images with $R_a$ = 1/2 (2 particles) and 2/20 (20 particles) pool to 3/22 = 0.136, while the mean of the ratios, 0.30, overstates it.

**Across images.** `summary_across_images` computes each metric per image, then the mean, the sample standard deviation $s$, the number of images $n$ with a value and the confidence interval $\bar{x} \pm t_{(1+c)/2,\,n-1}\, s/\sqrt{n}$ (level $c$, 0.95 by default). Each image weighs equally, so this describes the typical image; with $n = 1$ the spread is undefined (NaN). `values_across_images` does the same for any per-image value.

**Counts and ratios.** $N_{primary}$ primary particles, $N_{aerosol}$ aerosol particles (agglomerates and single particles), $N_{pp1}$ single particles, $N_{ppA} = N_{primary} - N_{pp1}$, $N_{aggl} = N_{aerosol} - N_{pp1}$. The agglomeration ratio $R_a = N_{aggl}/N_{primary}$ is that of Gotoh et al. (1996), when no particle is lost or added between deposition and counting. The agglomerated fraction $1 - N_{pp1}/N_{primary}$ is the number fraction of primary particles bound in agglomerates (called ER in agglpy ≤ 0.4). $n_{ppA} = N_{ppA}/N_{aggl}$ and $n_{ppP} = N_{primary}/N_{aerosol}$ are the mean numbers of primary particles per agglomerate and per aerosol particle.

**Per area.** With the analysed area of each image (field of view), `coverage` is the summed projected area of the agglomerates over the summed field of view (agglomerates never share area, so this is the exact union of all circles), and `N_*_per_area` are the counts per unit area. Circles crossing the image border count in full, so these values are biased high for large agglomerates on small images.

**Size descriptors.** Number-based: mean, sample standard deviation and the quantiles D10, D50, D90 (linear interpolation between sorted values) of the particle diameter and of the agglomerate's volume-equivalent diameter (over all aerosol particles), and the Sauter mean diameter $\sum D^3 / \sum D^2$ of the particles. All metrics use the visible values only; the with-hidden values (section 7) stay properties.

**Size distributions.** `distribution` counts any column over size classes $(a, b]$ (or $[a, b)$), the outer edge included. A value within $10^{-9}$ (relative) of an edge counts as on the edge, because a whole-pixel diameter times the pixel size misses round edges by one rounding step. Values outside the classes and missing values are not counted and are reported. Each class gives the amount (count, or the sum of a weight column such as the real particle volumes), the fraction, the cumulative fraction, the density (fraction / class width) and the log density, fraction / $\log_{10}(b/a)$ ($dN/d\log D$, ISO 9276-1).

**Units.** Properties are computed in pixels; each image is converted to physical units with its own pixel size before images are pooled, so metrics and distributions always see one unit.

## Output

Per-particle and per-agglomerate tables feed the population metrics and distributions of section 9 (`agglpy.core.metrics`, `agglpy.core.distributions`). agglpy 0.4 produced them in `Manager.generate_summary` and `generate_PSD` over `settings.analysis.PSD_space`.

## Implementation

agglpy is implemented in Python, built on OpenCV, NumPy, Pandas, and SciPy.


## References

- Gotoh, K., Karube, K., Masuda, H., Banba, Y., 1996. High-efficiency removal of fine particles deposited on a solid surface. Advanced Powder Technology 7, 219–232.
- ISO, 1998. ISO 9276-1:1998 Representation of results of particle size analysis — Part 1: Graphical representation. International Organization for Standardization, Geneva.
- Walton, W.H., 1948. Feret's statistical diameter as a measure of particle size. Nature 162, 329–330.
