"""Golden (characterization) tests of the automatic analysis path.

They record what today's code produces for two real SEM images (D7-017,
D7-019), so the Phase 2 refactor can't change a result unnoticed. They
check that results stay the *same*, not that they are *correct*; the
synthetic tests (roadmap 1.4) cover correctness.

Workflow under test: the author's notebook without manual correction.
``Manager(...)`` runs HCT on each image (``HCT_file: null`` in settings),
then ``batch_detect_agglomerates``, ``calc_extended_agl_param`` and the
table generators. Two checkpoints:

1. HCT: the circles in ``<name>_HCT.csv``, in pixels.
2. Agglomerates: which particles are grouped together, the parameters of
   each agglomerate, and the summary tables.

Particle IDs and agglomerate names depend on detection order and on a
global counter, so they are never compared. A particle is identified by
its position and size, an agglomerate by its members (see
``_canonical_tables``).

Expected files live in ``tests/data/golden/expected/``. After an intended
change of results, regenerate them and review the diff before committing:

    pytest tests/test_golden.py --force-regen
"""

import shutil
from pathlib import Path

import pandas as pd
import pytest

from agglpy.img_ds import ImgDataSet
from agglpy.manager import Manager

GOLDEN_DIR = Path(__file__).parent / "data" / "golden"
IMAGES_DIR = Path(__file__).parent / "data" / "input" / "multiple_image"
IMAGE_NAMES = ["D7-017", "D7-019"]

# Repeated runs differ by ~1e-12 relative at most (last digits of the
# center of mass); a physically meaningful change is far larger.
FLOAT_TOL = {"rtol": 1e-9, "atol": 0.0}
EXACT = {"rtol": 0.0, "atol": 0.0}
# Rg of a single-particle agglomerate is float noise (~1e-20 m) instead of
# 0, and noise has no stable relative value. 1e-15 m is far below a pixel
# (~39 nm).
RG_TOL = {"rtol": 1e-9, "atol": 1e-15}

# Particle coordinates in pixels are rounded to this many decimals. HCT
# works on a 0.5 px grid, so this only removes the noise of the pixel ->
# metre -> pixel round trip.
PX_DECIMALS = 3

pytestmark = pytest.mark.slow


@pytest.fixture
def original_datadir() -> Path:
    """Where pytest-regressions reads and writes the expected files."""
    return GOLDEN_DIR / "expected"


@pytest.fixture(scope="module")
def golden_manager(tmp_path_factory: pytest.TempPathFactory) -> Manager:
    """Run the workflow once for the whole module, in a temporary copy.

    The Manager writes ``<name>_HCT.csv`` and HCT preview images next to
    each image, so it must never run on ``tests/data`` directly.
    """
    wdir = tmp_path_factory.mktemp("golden") / "D7"
    for name in IMAGE_NAMES:
        img_dir = wdir / "images" / name
        img_dir.mkdir(parents=True)
        shutil.copy2(IMAGES_DIR / "images" / name / f"{name}.tif", img_dir)
    shutil.copy2(GOLDEN_DIR / "D7" / "settings.yml", wdir)

    M = Manager(working_dir=wdir)  # runs HCT: HCT_file is null
    M.batch_detect_agglomerates(export_img=False)
    M.calc_extended_agl_param(include_dsom=True)
    M.generate_pTable()
    M.generate_aglTable()
    M.generate_DSsummary()
    M.generate_summary()
    return M


def _canonical_tables(ds: ImgDataSet) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Particle and agglomerate tables without IDs or names.

    Particles are sorted by position and size (X, Y, D in pixels), so the
    row order doesn't depend on detection order. Each agglomerate is
    labelled by the row number of its first member in that order: column
    ``agglomerate`` in the particle table says which particles are grouped
    together, and the agglomerate table is indexed by the same label.
    """
    p = ds.get_results_pTable()
    particles = pd.DataFrame(
        {
            "X": (p["X"] / ds.px_size).round(PX_DECIMALS),
            "Y": (p["Y"] / ds.px_size).round(PX_DECIMALS),
            "D": (p["D"] / ds.px_size).round(PX_DECIMALS),
            "type": p["type"],
            "idj": p["idj"],
            "affiliation": p["affiliation"],
        }
    )
    particles = particles.sort_values(["X", "Y", "D"], ignore_index=True)
    assert not particles.duplicated(["X", "Y", "D"]).any(), (
        "Two particles share position and size; the canonical order "
        "would be ambiguous."
    )
    label = particles.index.to_series().groupby(particles["affiliation"])
    particles["agglomerate"] = label.transform("min")
    label_of = particles.groupby("affiliation")["agglomerate"].first()

    agglomerates = ds.get_results_aglTable()
    agglomerates.insert(
        0, "agglomerate", agglomerates["name"].map(label_of).astype(int)
    )
    agglomerates = agglomerates.drop(columns=["ID", "name"])
    agglomerates = agglomerates.sort_values("agglomerate", ignore_index=True)

    particles = particles.drop(columns="affiliation")
    return particles, agglomerates


@pytest.mark.parametrize("name", IMAGE_NAMES)
def test_hct_circles(golden_manager, dataframe_regression, name):
    """Checkpoint 1: automatic HCT finds the same circles (pixels)."""
    csv = golden_manager.working_dir / "images" / name / f"{name}_HCT.csv"
    circles = pd.read_csv(csv)[["X", "Y", "R"]]
    circles = circles.sort_values(["X", "Y", "R"], ignore_index=True)
    dataframe_regression.check(
        circles, basename=f"{name}_hct_circles", default_tolerance=EXACT
    )


@pytest.mark.parametrize("name", IMAGE_NAMES)
def test_agglomerate_membership(golden_manager, dataframe_regression, name):
    """Checkpoint 2: the same particles are grouped together.

    Also pins each particle's type and its idj flag (lies inside another
    particle).
    """
    particles, _ = _canonical_tables(golden_manager[name])
    dataframe_regression.check(
        particles, basename=f"{name}_particles", default_tolerance=EXACT
    )


@pytest.mark.parametrize("name", IMAGE_NAMES)
def test_agglomerate_parameters(golden_manager, dataframe_regression, name):
    """Checkpoint 2: each agglomerate keeps its type and parameters."""
    _, agglomerates = _canonical_tables(golden_manager[name])
    dataframe_regression.check(
        agglomerates,
        basename=f"{name}_agglomerates",
        default_tolerance=FLOAT_TOL,
        tolerances={"Rg": RG_TOL, "Dg": RG_TOL},
    )


def test_dataset_summary(golden_manager, dataframe_regression):
    """Per-image summary: counts, ER, Ra, sep2agl, D10/D50/D90, SMD, ..."""
    dataframe_regression.check(
        golden_manager.batch_res_DSsummary,
        basename="DSsummary",
        default_tolerance=FLOAT_TOL,
    )


def test_batch_summary(golden_manager, dataframe_regression):
    """Summary over both images, as Manager.generate_summary builds it."""
    summary = golden_manager.batch_res_summary
    summary = summary.rename(columns={0: "value"}).rename_axis("metric")
    dataframe_regression.check(
        summary.reset_index(),
        basename="summary",
        default_tolerance=FLOAT_TOL,
    )
