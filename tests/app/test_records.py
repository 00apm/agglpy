"""Particle records: dataclasses, YAML in and out, image facts."""

import hashlib
import os
import warnings
from pathlib import Path

import pytest

from agglpy import _yaml
from agglpy.app import _records as records_module
from agglpy.app._config import ImageParams
from agglpy.app._records import (
    CorrectionRecord,
    DetectionRecord,
    ImageInfo,
    ParticlesInfo,
    SourceInfo,
    StartedFrom,
    image_info,
    missing_image,
    read_record,
    record_text,
    same_image,
    write_record,
)
from agglpy.errors import UnreadableRecordWarning
from agglpy.params import (
    ClaheParams,
    HCTParams,
    HCTRange,
    PreprocessParams,
    RegionParams,
    RollingBallParams,
)

from support.tiffs import write_zeiss

DEPENDENCIES = {"numpy": "2.1.3", "pandas": "2.2.3", "opencv": None}
IMAGE = ImageInfo(
    file="images/D7-017.tif",
    sha256="9c" * 32,
    size=10485760,
    modified=1727683872123456789,
    file_id=281474976710656123,
    width=1024,
    height=943,
    pixel_size_nm=2.31,
)


@pytest.fixture
def detection() -> DetectionRecord:
    params = ImageParams(
        RegionParams(),
        PreprocessParams(
            rolling_ball=RollingBallParams(radius=50),
            clahe=ClaheParams(2.0, (8, 8)),
        ),
        HCTParams(ranges=(HCTRange(12, 44), HCTRange(40, 90))),
    )
    return DetectionRecord(
        created="2026-10-26T14:02:11+01:00",
        agglpy="0.5.0",
        dependencies=DEPENDENCIES,
        image=IMAGE,
        params=params,
        particles=ParticlesInfo("particles_hct.csv", 405, "3f" * 32),
        warnings=("DuplicateParticlesWarning: 3 circles found twice",),
    )


def correction(started_from: StartedFrom | None) -> CorrectionRecord:
    return CorrectionRecord(
        created="2026-10-26T15:00:00+01:00",
        agglpy="0.5.0",
        dependencies=DEPENDENCIES,
        image=missing_image("images/D7-017.tif"),
        particles=ParticlesInfo("particles_corrected.csv", 390, "ab" * 32),
        warnings=(),
        source=SourceInfo("C:/data/fiji/D7-017_v2.csv", "imagej", "51" * 32),
        started_from=started_from,
    )


def test_detection_record_reads_back_equal(
    tmp_path: Path, detection: DetectionRecord
) -> None:
    write_record(tmp_path / "r.yml", detection)
    assert read_record(tmp_path / "r.yml", DetectionRecord) == detection


@pytest.mark.parametrize(
    "started_from",
    [
        StartedFrom("3f" * 32, "2026-10-19T14:02:00+01:00", 380, 405),
        StartedFrom("3f" * 32, None, 380, 405),
        None,
    ],
    ids=["with-detection", "detection-without-record", "no-detection"],
)
def test_correction_record_reads_back_equal(
    tmp_path: Path, started_from: StartedFrom | None
) -> None:
    record = correction(started_from)
    write_record(tmp_path / "r.yml", record)
    assert read_record(tmp_path / "r.yml", CorrectionRecord) == record


def test_record_text_starts_with_header_and_kind(
    detection: DetectionRecord,
) -> None:
    lines = record_text(detection).splitlines()
    assert lines[0].startswith("# Written by agglpy ")
    assert "Do not edit" in lines[0]
    assert "particles_hct.csv" in lines[1]
    assert lines[2:4] == ["kind: detection", "format: 1"]


def test_correction_text_names_its_file() -> None:
    lines = record_text(correction(None)).splitlines()
    assert "importing corrected particles" in lines[0]
    assert "particles_corrected.csv" in lines[1]
    assert lines[2] == "kind: correction"


def test_missing_record_is_none_without_warning(tmp_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert read_record(tmp_path / "x.yml", DetectionRecord) is None


def _drop_particles(data: dict) -> None:
    del data["particles"]


def _no_ranges(data: dict) -> None:
    data["params"]["hct"]["ranges"] = []


def _count_text(data: dict) -> None:
    data["particles"]["count"] = "many"


def _count_bool(data: dict) -> None:
    data["particles"]["count"] = True


DAMAGE = {
    "format: 2": lambda text: text.replace("format: 1", "format: 2"),
    "kind: correction": lambda text: text.replace(
        "kind: detection", "kind: correction"
    ),
    "not yaml": lambda text: "not: [yaml",
    "a list": lambda text: "- 1\n- 2\n",
}
DATA_DAMAGE = {
    "drop particles": _drop_particles,
    "hct ranges []": _no_ranges,
    "count: many": _count_text,
    "count: true": _count_bool,
}


@pytest.mark.parametrize("damage", [*DAMAGE, *DATA_DAMAGE])
def test_damaged_record_is_none_and_warns(
    tmp_path: Path, detection: DetectionRecord, damage: str
) -> None:
    path = tmp_path / "D7-017" / "particles_hct_record.yml"
    path.parent.mkdir()
    write_record(path, detection)
    if damage in DAMAGE:
        text = DAMAGE[damage](path.read_text(encoding="utf-8"))
    else:
        data = _yaml.load(path)
        DATA_DAMAGE[damage](data)
        text = _yaml.dumps(data)
    path.write_text(text, encoding="utf-8")
    with pytest.warns(
        UnreadableRecordWarning,
        match=r"particles_hct_record\.yml in D7-017/ .* treated as missing",
    ):
        assert read_record(path, DetectionRecord) is None


def test_image_info_reads_size_hash_and_pixel_size(tmp_path: Path) -> None:
    # Zeiss: the raw value is the pixel size at 1024 px, so 1 nm raw
    # at 64 px wide is 16 nm
    path = write_zeiss(tmp_path / "a.tif", 64, 48, raw=1e-9)
    info = image_info("images/a.tif", path)
    assert info.file == "images/a.tif"
    assert (info.width, info.height) == (64, 48)
    assert info.size == path.stat().st_size
    assert info.modified == path.stat().st_mtime_ns
    assert info.file_id == path.stat().st_ino
    assert info.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert info.pixel_size_nm == pytest.approx(16.0)


def test_missing_image_keeps_only_the_stored_path() -> None:
    info = missing_image("images/a.tif")
    assert info == ImageInfo(
        "images/a.tif", None, None, None, None, None, None, None
    )


def test_same_image_skips_hash_when_size_and_time_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_zeiss(tmp_path / "a.tif", 64, 48, raw=1e-9)
    info = image_info("images/a.tif", path)

    def no_hash(path: Path) -> str:
        raise AssertionError("hashed")

    monkeypatch.setattr(records_module, "sha256", no_hash)
    assert same_image(info, path)


def test_same_image_rehashes_after_new_mtime(tmp_path: Path) -> None:
    path = write_zeiss(tmp_path / "a.tif", 64, 48, raw=1e-9)
    info = image_info("images/a.tif", path)
    later = info.modified + 10**9
    os.utime(path, ns=(later, later))
    assert same_image(info, path)  # a copy: new time, same bytes
    path.write_bytes(b"x" * info.size)
    os.utime(path, ns=(later, later))
    assert not same_image(info, path)


def test_record_in_another_encoding_is_none_and_warns(
    tmp_path: Path, detection: DetectionRecord
) -> None:
    path = tmp_path / "D7-017" / "particles_hct_record.yml"
    path.parent.mkdir()
    # re-saved by an editor in a Windows code page
    path.write_bytes(
        record_text(detection).replace("D7-017", "próbka").encode("cp1250")
    )
    with pytest.warns(UnreadableRecordWarning, match="treated as missing"):
        assert read_record(path, DetectionRecord) is None


def test_folder_in_place_of_record_is_none_and_warns(tmp_path: Path) -> None:
    path = tmp_path / "D7-017" / "particles_hct_record.yml"
    path.mkdir(parents=True)
    with pytest.warns(UnreadableRecordWarning, match="treated as missing"):
        assert read_record(path, DetectionRecord) is None
