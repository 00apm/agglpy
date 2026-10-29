"""Tuning previews: ``col[name].preview()`` and ``Preview``."""

import logging
import warnings
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np
import pandas as pd
import pytest

import agglpy.core.preprocess as core_preprocess
from agglpy.app import Collection, Preview
from agglpy.app import _preview as preview_module
from agglpy.errors import DuplicateParticlesWarning, ParamsError
from agglpy.io import read_image, read_particles
from agglpy.params import (
    ClaheParams,
    HCTRange,
    PreprocessParams,
    RollingBallParams,
)
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


class _Spies:
    """Counts of the OpenCV and scikit-image calls a preview makes."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.counts: Counter[str] = Counter()
        for owner, attribute, key in (
            (cv2, "HoughCircles", "hough"),
            (cv2, "medianBlur", "median"),
            (cv2, "createCLAHE", "clahe"),
            (core_preprocess, "rolling_ball", "rolling ball"),
        ):
            monkeypatch.setattr(
                owner, attribute, self._wrap(getattr(owner, attribute), key)
            )

    def _wrap(self, real: Any, key: str) -> Any:
        def spy(*args: Any, **kwargs: Any) -> Any:
            self.counts[key] += 1
            return real(*args, **kwargs)

        return spy

    def reset(self) -> None:
        self.counts.clear()


@pytest.fixture
def spies(monkeypatch: pytest.MonkeyPatch) -> _Spies:
    return _Spies(monkeypatch)


def _ranges(p: Preview) -> tuple[HCTRange, ...]:
    assert p.params.hct is not None
    return p.params.hct.ranges


def test_changes_rerun_only_what_they_touch(
    tuning_col: Collection, spies: _Spies
) -> None:
    p = tuning_col["T1"].preview(progress=False)
    assert spies.counts == {
        "median": 1,
        "rolling ball": 1,
        "clahe": 1,
        "hough": 2,
    }
    first, second = _ranges(p)
    spies.reset()
    q = p.change_hct(range=1, canny_threshold=150)
    assert spies.counts == {"hough": 1}
    assert _ranges(q) == (replace(first, canny_threshold=150), second)
    assert [found.reused for found in q._state.passes] == [False, True]
    assert _ranges(p)[0].canny_threshold == 200  # the parent unchanged
    spies.reset()
    q = p.add_range(40, 60)
    assert spies.counts == {"hough": 1}
    assert _ranges(q) == (first, second, HCTRange(40, 60))
    spies.reset()
    q = p.remove_range(1)
    assert spies.counts == {}
    assert _ranges(q) == (second,)
    # the remaining pass's circles, numbered from 1 again
    assert q.particles["id"].tolist() == [1, 2, 3]
    assert q.warnings == ()  # no duplicate pair left
    spies.reset()
    q = p.change_preprocess(clahe={"clip_limit": 3.0})
    assert spies.counts == {"clahe": 1, "hough": 2}
    assert q.params.preprocess.clahe == ClaheParams(3.0, (4, 4))
    assert q.params.preprocess.rolling_ball == p.params.preprocess.rolling_ball
    spies.reset()
    q = p.change_region(bottom_fraction=0.2)
    assert spies.counts == {
        "median": 1,
        "rolling ball": 1,
        "clahe": 1,
        "hough": 2,
    }
    assert q.params.region.bottom_fraction == 0.2
    assert q._state.prepared.cut.shape == (128, 200)


def test_changed_preview_equals_a_fresh_one(tuning_col: Collection) -> None:
    p = tuning_col["T1"].preview(progress=False)
    q = p.change_preprocess(rolling_ball={"radius": 8}).change_hct(
        range=2, accumulator_threshold=12
    )
    assert q.params.preprocess.rolling_ball == RollingBallParams(8)
    fresh = tuning_col["T1"].preview(params=q.params, progress=False)
    pd.testing.assert_frame_equal(q.particles, fresh.particles)
    assert q.warnings == fresh.warnings
    # the JPEG line goes along with every change
    jpeg = tuning_col["T3"].preview(progress=False)
    assert jpeg.remove_range(2).warnings[0].startswith("LossyImageWarning")


def test_change_to_the_same_value_reuses_everything(
    tuning_col: Collection, spies: _Spies
) -> None:
    p = tuning_col["T1"].preview(progress=False)
    spies.reset()
    for q in (
        p.change_hct(range=1, canny_threshold=200),
        p.change_preprocess(median_blur=3, clahe={"tile_grid": (4, 4)}),
        p.change_region(bottom_fraction=0.1),
    ):
        assert q is not p
        assert q.params == p.params
        pd.testing.assert_frame_equal(q.particles, p.particles)
        assert q.warnings == p.warnings
        assert all(found.reused for found in q._state.passes)
        assert q._state.preprocess_reused
    assert spies.counts == {}


def test_set_ranges_and_shared_fields(tuning_col: Collection) -> None:
    p = tuning_col["T1"].preview(progress=False)
    q = p.set_ranges([(6, 18), (16, 60)], canny_threshold=180)
    assert [
        (r.d_min, r.d_max, r.canny_threshold, r.accumulator_threshold)
        for r in _ranges(q)
    ] == [(6, 18, 180.0, 15.0), (16, 60, 180.0, 15.0)]
    empty = q.remove_range(1).remove_range(1)
    assert empty.params.hct is None
    assert empty.particles.empty
    # from no ranges, a range can be added again
    assert _ranges(empty.add_range(8, 18)) == (HCTRange(8, 18),)
    # set_ranges reruns only the passes that changed
    same = p.set_ranges([(8, 18), (16, 44)])
    assert all(found.reused for found in same._state.passes)


def test_change_errors(tuning_col: Collection) -> None:
    p = tuning_col["T1"].preview(progress=False)
    cases: list[tuple[Any, str]] = [
        (
            lambda: p.change_hct(range=3, canny_threshold=1),
            "range 3 does not exist: T1 has 2 ranges",
        ),
        (
            lambda: p.change_hct(range=1, cany=1),
            r"range 1: unknown field 'cany' \(valid: d_min, d_max, "
            r"min_dist_ratio, canny_threshold, accumulator_threshold\)",
        ),
        (lambda: p.change_hct(range=1, d_min=50), "d_min < d_max"),
        (lambda: p.change_hct(range="1", d_min=5), "range must be an int"),
        (
            lambda: p.change_preprocess(clahe={"clip": 1}),
            "preprocess.clahe: unknown field 'clip'",
        ),
        (
            lambda: p.change_preprocess(blur=3),
            "preprocess: unknown field 'blur'",
        ),
        (
            lambda: p.change_preprocess(clahe=3),
            "preprocess.clahe: give a mapping, a ClaheParams or None, got int",
        ),
        (lambda: p.change_region(top=0.1), "region: unknown field 'top'"),
        (
            lambda: p.set_ranges([]),
            r"give at least one \(d_min, d_max\) pair; p\.remove_range\(k\) "
            "removes one",
        ),
        (lambda: p.set_ranges([(1, 5)], d_min=2), "d_min"),
        (lambda: p.set_ranges([(1, 5, 7)]), r"\(d_min, d_max\) pairs"),
        (
            lambda: p.remove_range(0),
            "range 0 does not exist: T1 has 2 ranges",
        ),
    ]
    for call, message in cases:
        with pytest.raises(ParamsError, match=message):
            call()
    off = (
        p.change_preprocess(median_blur=None)
        .change_preprocess(rolling_ball=None)
        .change_preprocess(clahe=None)
    )
    assert off.params.preprocess == PreprocessParams()
    with pytest.raises(
        ParamsError, match=r"preprocess.clahe: needs tile_grid"
    ):
        off.change_preprocess(clahe={"clip_limit": 2})
    assert off.change_preprocess(
        clahe=ClaheParams(2.0, (4, 4))
    ).params.preprocess == PreprocessParams(clahe=ClaheParams(2.0, (4, 4)))
    none = p.remove_range(1).remove_range(1)
    for call in (
        lambda: none.change_hct(range=1, canny_threshold=1),
        lambda: none.remove_range(1),
    ):
        with pytest.raises(
            ParamsError,
            match=r"T1 has no HCT ranges: p\.set_ranges\(\[\(d_min, d_max\)",
        ):
            call()
    # an object of another section is not taken
    with pytest.raises(ParamsError, match="RollingBallParams or None"):
        p.change_preprocess(rolling_ball=ClaheParams(2.0, (4, 4)))


def test_param_differences(tuning_col: Collection) -> None:
    p = tuning_col["T1"].preview(progress=False)
    differences = preview_module._differences
    assert differences(p.params, p.params) == []
    q = p.change_hct(range=2, canny_threshold=180).change_preprocess(
        rolling_ball={"radius": 40}, clahe=None
    )
    assert differences(p.params, q.params) == [
        "preprocess.rolling_ball.radius 20 → 40",
        "preprocess.clahe {…} → null",
        "ranges[2].canny_threshold 200 → 180",
    ]
    assert differences(p.params, p.add_range(60, 80).params) == [
        "ranges: 2 → 3"
    ]
    region = p.change_region(bottom_fraction=0.05).params
    assert differences(p.params, region) == [
        "region.bottom_fraction 0.1 → 0.05"
    ]
    empty = p.remove_range(1).remove_range(1)
    assert differences(empty.params, p.params) == ["ranges: 0 → 2"]
