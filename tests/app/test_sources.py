"""Image sources: files from folders, stored paths, the filename rule."""

import os
import unicodedata
from pathlib import Path

import pytest

from agglpy.app._sources import (
    Found,
    apply_rule,
    find_images,
    resolve_stored,
    stored_path,
)
from agglpy.errors import ParamsError, UnmatchedPathsWarning


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    (raw / "sub").mkdir(parents=True)
    for name in ("D7-017.tif", "D7-019.TIF", "notes.txt"):
        (raw / name).touch()
    (raw / "sub" / "D7-017_HCT.jpg").touch()
    return raw


def keys(found: list[Found]) -> list[str]:
    return [f.key for f in found]


def found(*keys: str) -> list[Found]:
    return [Found(Path(k).resolve(), k) for k in keys]


def test_folder_gives_image_files_sorted_not_recursive(raw: Path) -> None:
    result = find_images(raw)
    assert keys(result) == ["D7-017.tif", "D7-019.TIF"]
    assert result[0].path == (raw / "D7-017.tif").resolve()


def test_pattern_globs_and_recursion(raw: Path) -> None:
    assert keys(find_images(raw, "**/*.jpg")) == ["sub/D7-017_HCT.jpg"]
    # On Windows both globs match both files: each is still found once.
    assert keys(find_images(raw, ["*.tif", "*.TIF"])) == [
        "D7-017.tif",
        "D7-019.TIF",
    ]


def test_file_given_directly_taken_as_is(raw: Path) -> None:
    assert find_images(raw / "notes.txt") == [
        Found((raw / "notes.txt").resolve(), "notes.txt")
    ]


def test_list_keeps_order_and_dedupes(raw: Path) -> None:
    result = find_images([raw / "D7-019.TIF", raw])
    assert keys(result) == ["D7-019.TIF", "D7-017.tif"]


def test_folder_without_images_gives_nothing(raw: Path) -> None:
    assert find_images(raw / "sub", "*.tif") == []


def test_missing_path(raw: Path) -> None:
    with pytest.raises(FileNotFoundError, match="nothing"):
        find_images(raw / "nothing")


def test_stored_path_relative_with_slash(raw: Path, tmp_path: Path) -> None:
    path = raw / "sub" / "D7-017_HCT.jpg"
    assert stored_path(path, tmp_path) == "raw/sub/D7-017_HCT.jpg"
    assert resolve_stored("raw/sub/D7-017_HCT.jpg", tmp_path) == (
        path.resolve()
    )


@pytest.mark.skipif(os.name != "nt", reason="drive letters")
def test_stored_path_other_drive_absolute(tmp_path: Path) -> None:
    drive = "Z:" if tmp_path.drive.upper() != "Z:" else "Y:"
    path = Path(f"{drive}/SEM/D7-017.tif")
    assert stored_path(path, tmp_path) == f"{drive}/SEM/D7-017.tif"
    assert resolve_stored(f"{drive}/SEM/D7-017.tif", tmp_path) == path


def test_no_rule_names_are_stems() -> None:
    names, columns, values = apply_rule(found("sub/D7-017.tif"), None)
    assert (names, columns, values) == (["D7-017"], [], [{}])


def test_rule_fills_columns() -> None:
    names, columns, values = apply_rule(
        found("D7-017.tif"), r"(?P<series>D\d+)-(?P<n>\d+)"
    )
    assert names == ["D7-017"]
    assert columns == ["series", "n"]
    assert values[0] == {"series": "D7", "n": "017"}


def test_rule_image_group_sets_name() -> None:
    names, columns, _ = apply_rule(
        found("x_D7-017_y.tif"), r"(?P<image>D\d+-\d+)"
    )
    assert names == ["D7-017"]
    assert columns == []


def test_rule_matches_relative_path_with_slash() -> None:
    _, _, values = apply_rule(found("day1/a.tif"), r"^(?P<day>[^/]+)/")
    assert values == [{"day": "day1"}]


def test_rule_nfc(tmp_path: Path) -> None:
    nfd = unicodedata.normalize("NFD", "zło-ó.tif")
    (tmp_path / nfd).touch()
    result = find_images(tmp_path)
    assert keys(result) == [unicodedata.normalize("NFC", "zło-ó.tif")]
    names, _, values = apply_rule(result, "(?P<w>zło-ó)")
    assert names == [unicodedata.normalize("NFC", "zło-ó")]
    assert values == [{"w": unicodedata.normalize("NFC", "zło-ó")}]


def test_unmatched_paths_warn_once() -> None:
    with pytest.warns(UnmatchedPathsWarning) as record:
        names, _, values = apply_rule(
            found("D7-017.tif", "a.tif", "b.tif"), r"(?P<series>D\d+)"
        )
    assert len(record) == 1
    message = str(record[0].message)
    assert "a.tif" in message and "b.tif" in message
    assert "D7-017.tif" not in message
    assert names == ["D7-017", "a", "b"]
    assert values == [{"series": "D7"}, {}, {}]


def test_unmatched_paths_listed_up_to_ten() -> None:
    many = [f"{i:02d}.tif" for i in range(12)]
    with pytest.warns(UnmatchedPathsWarning, match=r"09\.tif and 2 more"):
        apply_rule(found(*many), r"(?P<series>D\d+)")


def test_rule_reserved_group_name() -> None:
    with pytest.raises(ParamsError, match="'file'"):
        apply_rule(found("a.tif"), "(?P<file>.*)")


def test_bad_regex() -> None:
    with pytest.raises(ParamsError, match="columns_from_path"):
        apply_rule(found("a.tif"), "(?P<x")


def test_hidden_files_left_out_of_folder_listing(raw: Path) -> None:
    # macOS writes ._<name> companions on FAT / exFAT / network drives.
    (raw / "._D7-017.tif").touch()
    (raw / ".hidden.png").touch()
    assert keys(find_images(raw)) == ["D7-017.tif", "D7-019.TIF"]
