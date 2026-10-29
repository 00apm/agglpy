"""Tuning previews: ``col[name].preview()`` and ``Preview``."""

import logging
import warnings
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

from agglpy.app import Collection, Preview
from agglpy.app import _preview as preview_module
from agglpy.errors import DuplicateParticlesWarning, ParamsError
from agglpy.io import read_image, read_particles
from agglpy.pipeline import detect_particles

from support.collection import work


def _file(col: Collection, name: str) -> Path:
    suffix = ".jpg" if name == "T3" else ".tif"
    return col.folder.parent / "images" / f"{name}{suffix}"


def _snapshot(folder: Path) -> list[tuple[Path, int]]:
    return sorted((p, p.stat().st_mtime_ns) for p in folder.rglob("*"))


def test_preview_shows_what_detect_stores(tuning_col: Collection) -> None:
    p = tuning_col["T1"].preview(progress=False)
    assert isinstance(p, Preview)
    assert p.image_name == "T1"
    params = tuning_col.params("T1")
    assert p.params == params
    assert params.hct is not None
    # range 2 finds the r 8.5 disc of range 1 again: ids 1 and 7
    with pytest.warns(DuplicateParticlesWarning, match=r"\(1, 7\)"):
        expected = detect_particles(
            read_image(_file(tuning_col, "T1")),
            region=params.region,
            preprocess=params.preprocess,
            hct=params.hct,
        )
    pd.testing.assert_frame_equal(p.particles, expected)
    assert len(p.particles) == 7
    with pytest.warns(DuplicateParticlesWarning):
        tuning_col.detect(["T1"], progress=False)
        stored = read_particles(work(tuning_col, "T1") / "particles_hct.csv")
    pd.testing.assert_frame_equal(stored, p.particles)
    detection = tuning_col["T1"].detection
    assert detection is not None
    assert p.warnings == tuple(detection["warnings"])
    assert [w.split(":")[0] for w in p.warnings] == [
        "DuplicateParticlesWarning"
    ]


def test_preview_writes_nothing_and_warns_nothing(
    tuning_col: Collection,
) -> None:
    before = _snapshot(tuning_col.folder)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p = tuning_col["T3"].preview(progress=False)
    assert caught == []
    assert _snapshot(tuning_col.folder) == before
    # JPEG: the lossy line first, as detect() records it
    assert p.warnings[0] == (
        "LossyImageWarning: ../images/T3.jpg is JPEG-compressed; edges may "
        "shift by the compression artefacts"
    )
    assert p.warnings[1].startswith("DuplicateParticlesWarning: ")


def test_preview_from_params_and_without_ranges(
    tuning_col: Collection,
) -> None:
    p1 = tuning_col["T1"].preview(progress=False)
    assert tuning_col.params("T2") != p1.params
    p2 = tuning_col["T2"].preview(params=p1.params, progress=False)
    assert p2.params == p1.params
    assert p2.image_name == "T2"
    assert len(p2.particles) == 4
    # T2's own empty list replaces the defaults' ranges
    tuning_col.set_params("T2", hct={"ranges": []})
    p3 = tuning_col["T2"].preview(progress=False)
    assert p3.params.hct is None
    assert p3.particles.empty
    assert list(p3.particles.columns) == ["id", "x", "y", "r"]
    assert p3.warnings == ()
    with pytest.raises(
        ParamsError,
        match=r"params must be ImageParams \(p\.params or "
        r"col\[name\]\.params\), got dict",
    ):
        tuning_col["T1"].preview(params={"hct": {}})  # type: ignore[arg-type]
    # an excluded image previews too
    sheet = tuning_col.read_sheet()
    sheet["include"] = [False, True, True]
    tuning_col.write_sheet(sheet)
    assert len(tuning_col["T1"].preview(progress=False).particles) == 7


class _Bar:
    made: ClassVar[list["_Bar"]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.descriptions: list[str] = []
        self.updates = 0
        self.closed = False
        _Bar.made.append(self)

    def set_description(self, text: str) -> None:
        self.descriptions.append(text)

    def update(self, n: int) -> None:
        self.updates += n

    def close(self) -> None:
        self.closed = True


def test_preview_progress_and_log(
    tuning_col: Collection,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    tuning_col["T1"].preview(progress=False)
    assert capsys.readouterr().err == ""
    _Bar.made = []
    monkeypatch.setattr(preview_module, "tqdm", _Bar)
    caplog.set_level(logging.INFO, logger="agglpy")
    p = tuning_col["T1"].preview()
    (bar,) = _Bar.made
    assert bar.kwargs == {"total": 5, "unit": "step", "disable": False}
    assert bar.descriptions == [
        "T1 · reading",
        "T1 · median",
        "T1 · rolling ball",
        "T1 · clahe",
        "T1 · hct",
    ]
    assert bar.updates == 5
    assert bar.closed
    (line,) = [r.getMessage() for r in caplog.records]
    assert line.startswith(f"T1: preview, {len(p.particles)} circles in 2 ")
    assert line.endswith(" s")


def test_preview_is_immutable(tuning_col: Collection) -> None:
    p = tuning_col["T1"].preview(progress=False)
    table = p.particles
    table["x"] += 1
    assert not p.particles.equals(table)
    with pytest.raises(AttributeError):
        p.params = tuning_col.params("T2")  # type: ignore[misc]
    held = p._state.prepared
    arrays = [held.raw, held.cut, *(s.image for s in held.stages)]
    arrays += [found.circles for found in p._state.passes]
    assert not any(a.flags.writeable for a in arrays)


def test_image_array(tuning_col: Collection) -> None:
    array = tuning_col["T1"].array
    assert array.dtype == np.uint8
    assert array.shape == (160, 200)  # the whole file, not the region
    np.testing.assert_array_equal(array, read_image(_file(tuning_col, "T1")))
    _file(tuning_col, "T1").unlink()
    with pytest.raises(FileNotFoundError):
        tuning_col["T1"].array
    with pytest.raises(FileNotFoundError):
        tuning_col["T1"].preview(progress=False)
