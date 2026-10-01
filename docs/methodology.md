# Methodology

agglpy implements a method for detecting spherical primary particles in Scanning Electron Microscopy (SEM) images and classifying them into agglomerates. The method was developed for analyzing aerosol particles (e.g. fly ash) collected onto microscope specimen stubs, but is applicable to any image where particles can reasonably be approximated as circles.


## Overview

The analysis proceeds through six stages:

1. Input image preprocessing,
2. Edge detection and binarization,
3. Primary particle detection using the Hough Circle Transform (HCT),
4. (Optional) Manual correction of missed and incorrectly detected primary particles,
5. Primary particle connectivity determination,
6. Grouping of connected particles into agglomerates.

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

## Output

Per-agglomerate and per-particle results feed into batch-level aggregation (`Manager.generate_pTable` / `generate_aglTable` / `generate_PSD`), producing particle-size-distribution (PSD) statistics binned over `settings.analysis.PSD_space`.

## Implementation

agglpy is implemented in Python, built on OpenCV, NumPy, Pandas, and SciPy.


## References
