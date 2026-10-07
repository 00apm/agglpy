"""Run synthetic cases through the legacy ImgDataSet code (v0.4 API).

This is the only place in the synthetic tests that knows the legacy
classes. It is deleted together with them (roadmap 2.9).

The case goes in the way real data does: as an agglpy particle CSV
next to an image file, loaded by ``ImgDataSet(auto_load=True)``. The
binning functions are called directly (``legacy_psd_bins``,
``legacy_distribution``).
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from agglpy._legacy.img_ds import ImgDataSet
from agglpy._legacy.manager import Manager, PSD_space

from .cases import Case
from .result import Result


def run_legacy(
    case: Case,
    tmp_path: Path,
    pixel_size: float = 1.0,
) -> Result:
    """Run one case through ImgDataSet and convert the outcome to a Result.

    Args:
        case: The synthetic case.
        tmp_path: A fresh directory; a dataset folder is created inside.
        pixel_size: Metres per pixel given to ImgDataSet. The legacy code
            scales coordinates by it before grouping.
    """
    ds_dir = tmp_path / case.name
    ds_dir.mkdir()
    # The constructor reads an image; its content is never used here.
    cv2.imwrite(str(ds_dir / "img.png"), np.zeros((8, 8), dtype=np.uint8))

    keys = [c.key for c in case.circles]
    pd.DataFrame(
        {
            "ID": range(1, len(keys) + 1),
            "X": [c.x for c in case.circles],
            "Y": [c.y for c in case.circles],
            "R": [c.r for c in case.circles],
        }
    ).to_csv(ds_dir / "particles.csv", index=False)

    settings = {
        "img_file": "img.png",
        "HCT_file": "particles.csv",
        "magnification": 1.0,
        "pixel_size": pixel_size,
    }
    ds = ImgDataSet(ds_dir, settings=settings, auto_load=True)
    _clear_shared_search_list()
    try:
        ds.detect_agglomerates()
    finally:
        _clear_shared_search_list()
    ds.classify_all_AGL(threshold=case.threshold)
    ds.calc_extended_agl_param(include_dsom=True)

    p = ds.get_results_pTable().set_index("ID")
    key_by_id = dict(zip(range(1, len(keys) + 1), keys, strict=True))
    particles = pd.DataFrame(
        {
            "x": p["X"] / pixel_size,
            "y": p["Y"] / pixel_size,
            "r": p["D"] / 2 / pixel_size,
            "agglomerate_id": p["affiliation"],
            "type": p["type"],
            "enclosed": p["idj"].astype(bool),
        }
    )
    particles.index = particles.index.map(key_by_id)
    particles = particles.loc[keys]

    agl = ds.get_results_aglTable().set_index("name")
    members = particles.groupby("agglomerate_id").groups
    px, px3 = pixel_size, pixel_size**3
    agglomerates = pd.DataFrame(
        {
            "members": [frozenset(members[name]) for name in agl.index],
            "type": agl["type"],
            "member_count": agl["members_count"],
            "volume": agl["volume"] / px3,
            "D": agl["D"] / px,
            "D_mean": agl["members_Dmean"] / px,
            "D_std": agl["members_Dstdev"] / px,
            "enclosed_count": agl["idj_members_count"],
            "volume_with_hidden": agl["volume_dsom"] / px3,
            "D_with_hidden": agl["D_dsom"] / px,
            "member_count_with_hidden": agl["members_count_dsom"],
        },
        index=agl.index.rename("agglomerate_id"),
    )

    # x / 0 -> inf and 0 / 0 -> NaN are intended (cases.SUMMARY); numpy
    # would warn about them in every case without agglomerates
    with np.errstate(divide="ignore", invalid="ignore"):
        summary = ds.get_summary().iloc[0].to_dict()
    for name in _SUMMARY_DIAMETERS:
        summary[name] /= px
    return Result(
        particles=particles, agglomerates=agglomerates, summary=summary
    )


_SUMMARY_DIAMETERS = (
    "particle_Dmean",
    "particle_Dstd",
    "particle_D10",
    "particle_D50",
    "particle_D90",
    "particle_SMD",
)


def legacy_psd_bins(
    start: float,
    end: float,
    periods: float,
    log: bool = False,
    step: bool = False,
) -> np.ndarray:
    """Bin edges from ``agglpy.manager.PSD_space``."""
    return PSD_space(start=start, end=end, periods=periods, log=log, step=step)


def legacy_distribution(values: list[float], bins: np.ndarray) -> pd.DataFrame:
    """Binned distribution from ``Manager.generate_PSD``.

    ``generate_PSD`` only reads ``batch_res_pDF.D`` and the bins, so a
    Manager is created without its constructor (which needs a working
    directory) and given just those.
    """
    mgr = Manager.__new__(Manager)
    mgr.name = "synthetic"
    mgr._shortID = "0"
    mgr.batch_res_pDF = pd.DataFrame({"D": values})
    mgr.generate_PSD(PSD_space=bins, plot=False)
    psd = mgr.batch_res_PSD
    return pd.DataFrame(
        {
            "left": psd["left"],
            "right": psd["right"],
            "mid": psd["mid"],
            "width": psd["width"],
            "count": psd["counts"],
            "cumulative": psd["cummulative"],
            "count_norm": psd["counts_norm"],
            "cumulative_norm": psd["cummulative_norm"],
            "volume": psd["volume"],
            "volume_norm": psd["volume_norm"],
            "volume_cumulative_norm": psd["volume_cummulative_norm"],
        }
    ).reset_index(drop=True)


def _clear_shared_search_list() -> None:
    """Empty the visited-ID list the legacy search keeps between calls.

    ``ImgDataSet._find_intersecting_family`` collects visited IDs in a
    mutable default argument (D-015). A search that stops halfway (e.g.
    on RecursionError) leaves its IDs there, and the next detection in
    the same process starts with them: it crashes or merges unrelated
    particles. All tests run in one process, so every run starts and
    ends with an empty list.
    """
    ImgDataSet._find_intersecting_family.__defaults__[0].clear()
