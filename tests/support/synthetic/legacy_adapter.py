"""Run synthetic cases through the legacy ImgDataSet code (v0.4 API).

This is the only place in the synthetic tests that knows the legacy
classes. It is deleted together with them (roadmap 2.9).

The case goes in the way real data does: as an agglpy particle CSV
next to an image file, loaded by ``ImgDataSet(auto_load=True)``.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from agglpy.img_ds import ImgDataSet

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
            "idj": p["idj"].astype(bool),
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
            "members_count": agl["members_count"],
            "volume": agl["volume"] / px3,
            "D": agl["D"] / px,
            "members_D_mean": agl["members_Dmean"] / px,
            "members_D_std": agl["members_Dstdev"] / px,
            "idj_count": agl["idj_members_count"],
            "volume_dsom": agl["volume_dsom"] / px3,
            "D_dsom": agl["D_dsom"] / px,
            "members_count_dsom": agl["members_count_dsom"],
        },
        index=agl.index.rename("agglomerate_id"),
    )
    return Result(particles=particles, agglomerates=agglomerates)


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
