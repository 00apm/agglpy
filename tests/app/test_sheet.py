"""The image sheet: finding, reading, validating, typing user columns."""

from datetime import datetime
from pathlib import Path

import openpyxl
import pytest
from openpyxl.worksheet.formula import ArrayFormula

from agglpy.app._sheet import (
    FIXED_COLUMNS,
    RESERVED_COLUMNS,
    SHEET_FILES,
    Entry,
    find_sheet,
    read_grid,
    typed_user_columns,
    validate,
)
from agglpy.errors import FileFormatError, SheetError

from support.sheets import write_csv, write_xlsx

H = ["image", "file", "include", "pixel_size_nm", "note"]


@pytest.fixture
def xlsx(tmp_path: Path):
    def make(rows: list[list[object]]):
        return validate(read_grid(write_xlsx(tmp_path / "images.xlsx", rows)))

    return make


def problems(xlsx, rows: list[list[object]]) -> str:
    with pytest.raises(SheetError) as e:
        xlsx(rows)
    return str(e.value)


def row(**cells: object) -> list[object]:
    """One row under ``H``: image a, file raw/a.tif, the rest empty."""
    values = {"image": "a", "file": "raw/a.tif"} | cells
    return [values.get(name) for name in H]


def test_constants() -> None:
    assert SHEET_FILES == ("images.xlsx", "images.csv")
    assert FIXED_COLUMNS == (
        "image",
        "file",
        "include",
        "pixel_size_nm",
        "note",
    )
    assert {
        "pixel_size",
        "pixel_size_source",
        "fov_area",
        "width",
        "height",
        "particle_set",
        "outside_region",
    } < RESERVED_COLUMNS
    assert set(FIXED_COLUMNS) < RESERVED_COLUMNS


def test_valid_xlsx_sheet(xlsx) -> None:
    s = xlsx(
        [
            [*H, "T"],
            ["a", "raw/a.tif", True, None, None, 20],
            ["b", "raw/b.tif", False, 3.9, "blurry", 100],
        ]
    )
    assert s.names == ("a", "b")
    assert s.user_columns == ("T",)
    assert s.columns == (*H, "T")
    assert s.entries[0] == Entry(
        2, "a", "raw/a.tif", True, None, "", {"T": 20}
    )
    assert s.entries[1] == Entry(
        3, "b", "raw/b.tif", False, 3.9, "blurry", {"T": 100}
    )


def test_optional_fixed_columns_may_be_missing(xlsx) -> None:
    s = xlsx([["image", "file"], ["a", "raw/a.tif"]])
    assert s.entries == (Entry(2, "a", "raw/a.tif", True, None, "", {}),)


def test_csv_reads_text_and_tolerates_bom(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "images.csv",
        [[*H, "T"], ["a", "raw/a.tif", "", "3.9", "", "007"]],
        bom=True,
        newline="\r\n",
    )
    s = validate(read_grid(path))
    assert s.entries == (
        Entry(2, "a", "raw/a.tif", True, 3.9, "", {"T": "007"}),
    )


def test_csv_must_be_utf8(tmp_path: Path) -> None:
    path = tmp_path / "images.csv"
    path.write_bytes("image,file\nż,a.tif\n".encode("cp1250"))
    with pytest.raises(FileFormatError, match="UTF-8"):
        read_grid(path)


@pytest.mark.parametrize(
    "value", [True, False, "TRUE", "yes", "No", "1", "0", 1, 0, None, ""]
)
def test_include_accepted(xlsx, value: object) -> None:
    s = xlsx([H, row(include=value)])
    assert s.entries[0].include == (
        value in (True, "TRUE", "yes", "1", 1, None, "")
    )


def test_include_rejected_mentions_polish_excel(xlsx) -> None:
    message = problems(xlsx, [H, row(include="PRAWDA")])
    assert "row 2, include" in message
    assert "PRAWDA/FAŁSZ" in message
    assert "images.xlsx" in message


def test_image_names_unique_ignoring_case(xlsx) -> None:
    message = problems(xlsx, [H, row(image="a"), row(image="A")])
    assert message == (
        "images.xlsx has 1 problem:\n"
        "  rows 2, 3, image: 'a' and 'A' differ only in case"
    )


def test_repeated_image_name(xlsx) -> None:
    message = problems(xlsx, [H, row(image="a"), row(), row(image="b")])
    assert "  rows 2, 3, image: 'a' is repeated" in message


@pytest.mark.parametrize(
    "name",
    [
        "a/b",
        "a\\b",
        "a:b",
        "x?",
        "dot.",
        "space ",
        "CON",
        "con.tif",
        "LPT9",
        "COM1",
        "",
        "tab\there",
    ],
)
def test_bad_image_names(xlsx, name: str) -> None:
    message = problems(xlsx, [H, row(image=name)])
    assert "row 2, image" in message


@pytest.mark.parametrize("name", ["D7-019", "próbka ł", "CONSOLE", "a.b"])
def test_good_image_names(xlsx, name: str) -> None:
    assert xlsx([H, row(image=name)]).names == (name,)


def test_image_and_file_read_as_text(xlsx) -> None:
    s = xlsx([H, row(image=7, file=12.0)])
    assert (s.entries[0].image, s.entries[0].file) == ("7", "12")


def test_file_is_required(xlsx) -> None:
    message = problems(xlsx, [H, row(file=None)])
    assert "  row 2, file: required" in message


@pytest.mark.parametrize(
    ("value", "nm"),
    [
        (0.001, 0.001),
        (39.0625, 39.0625),
        (10000, 10000.0),
        ("3.9", 3.9),
        (None, None),
    ],
)
def test_pixel_size_accepted(xlsx, value: object, nm: float | None) -> None:
    assert xlsx([H, row(pixel_size_nm=value)]).entries[0].pixel_size_nm == nm


def test_pixel_size_in_metres_suggests_nm(xlsx) -> None:
    message = problems(xlsx, [H, row(pixel_size_nm=3.9e-08)])
    assert (
        "row 2, pixel_size_nm: 3.9e-08 is outside 0.001 to 10000 nm; the "
        "column is in nm (3.9e-08 m = 39 nm)"
    ) in message


@pytest.mark.parametrize("value", [0, -1, 20000, "3,9", True, "nan", "x"])
def test_pixel_size_rejected(xlsx, value: object) -> None:
    message = problems(xlsx, [H, row(pixel_size_nm=value)])
    assert "row 2, pixel_size_nm" in message


def test_note_is_text(xlsx) -> None:
    assert xlsx([H, row(note=3)]).entries[0].note == "3"


def test_fully_empty_rows_are_skipped(xlsx) -> None:
    s = xlsx([H, row(), [None] * 5, row(image="b")])
    assert [e.row for e in s.entries] == [2, 4]


def test_missing_required_column_hints_at_case(xlsx) -> None:
    message = problems(xlsx, [["Image", "file"], ["a", "raw/a.tif"]])
    assert message == (
        "images.xlsx has 1 problem:\n"
        "  column 'image' is missing (column names are case-sensitive; "
        "found 'Image')"
    )


def test_missing_both_required_columns(xlsx) -> None:
    message = problems(xlsx, [["note"], ["x"]])
    assert "column 'image' is missing" in message
    assert "column 'file' is missing" in message


@pytest.mark.parametrize(
    "col", ["pixel_size", "fov_area", "particle_set", "width"]
)
def test_reserved_user_column(xlsx, col: str) -> None:
    message = problems(xlsx, [[*H, col], [*row(), 1]])
    assert (
        f"column '{col}': agglpy adds a column of this name to the images "
        "table; rename it"
    ) in message


def test_reserved_names_are_case_sensitive(xlsx) -> None:
    assert xlsx([[*H, "Pixel_Size"], [*row(), 1]]).user_columns == (
        "Pixel_Size",
    )


def test_formulas_are_errors(tmp_path: Path) -> None:
    rows = [[*H, "F"], row(), row(image="b"), row(image="c")]
    path = write_xlsx(tmp_path / "images.xlsx", rows)
    book = openpyxl.load_workbook(path)
    sheet = book.active
    sheet["D2"] = "=F2*2"
    sheet["D3"] = "=F3*2"
    sheet["F4"] = ArrayFormula("F4", "=SUM(A1:A2)")
    book.save(path)
    with pytest.raises(SheetError) as e:
        read_grid(path)
    assert str(e.value) == (
        "images.xlsx: D2:D3, F4 contain formulas; replace them with values "
        "(Paste Special → Values)"
    )


def test_two_worksheets(tmp_path: Path) -> None:
    path = write_xlsx(tmp_path / "images.xlsx", [H, row()], sheets=2)
    with pytest.raises(SheetError) as e:
        read_grid(path)
    assert str(e.value) == (
        "images.xlsx has 2 worksheets (Sheet, Sheet1); keep one"
    )


def test_header_problems(tmp_path: Path) -> None:
    rows = [["image", "file", None, 2024, "file"], ["a", "b", "x", 1, "c"]]
    path = write_xlsx(tmp_path / "images.xlsx", rows)
    with pytest.raises(SheetError) as e:
        read_grid(path)
    assert str(e.value) == (
        "images.xlsx has 3 problems:\n"
        "  column C: no name in row 1, but the column has values\n"
        "  column D: the name 2024 is not text\n"
        "  column 'file' is repeated (columns B, E)"
    )


def test_trailing_empty_header_without_data_is_dropped(tmp_path) -> None:
    path = write_xlsx(tmp_path / "images.xlsx", [[*H, None], [*row(), None]])
    assert read_grid(path).header == H


def test_all_problems_at_once(xlsx) -> None:
    message = problems(
        xlsx,
        [
            H,
            row(image="a", include="maybe"),
            row(image="b"),
            row(image="b"),
            row(image="c", file=None),
        ],
    )
    lines = message.split("\n")
    assert lines[0] == "images.xlsx has 3 problems:"
    assert lines[1].startswith("  row 2, include")
    assert lines[2].startswith("  rows 3, 4, image")
    assert lines[3] == "  row 5, file: required"


def test_both_sheet_files(tmp_path: Path) -> None:
    (tmp_path / "images.xlsx").touch()
    (tmp_path / "images.csv").touch()
    with pytest.raises(SheetError, match=r"images\.xlsx.*images\.csv"):
        find_sheet(tmp_path)


def test_find_sheet(tmp_path: Path) -> None:
    (tmp_path / "images.csv").touch()
    assert find_sheet(tmp_path) == tmp_path / "images.csv"


def test_no_sheet(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"Collection\.create"):
        find_sheet(tmp_path)


def test_unreadable_xlsx(tmp_path: Path) -> None:
    path = tmp_path / "images.xlsx"
    path.write_bytes(b"not a zip")
    with pytest.raises(FileFormatError, match=r"images\.xlsx"):
        read_grid(path)


# ---------------------------------------------------------------------
# Typing of user columns
# ---------------------------------------------------------------------


def test_user_columns_typing(xlsx) -> None:
    header = [*H, "T", "ID", "R", "Mixed", "Sel", "Gap", "When"]
    s = xlsx(
        [
            header,
            [*row(), 20, "007", "0.5", "x", True, 1.5, datetime(2026, 1, 2)],
            [
                *row(image="b"),
                100,
                "008",
                "2",
                3,
                False,
                None,
                datetime(2026, 1, 3),
            ],
        ]
    )
    t = typed_user_columns(s)
    assert list(t.columns) == [
        "image",
        "T",
        "ID",
        "R",
        "Mixed",
        "Sel",
        "Gap",
        "When",
    ]
    assert t["image"].tolist() == ["a", "b"]
    assert t["T"].dtype == "int64"
    assert t["ID"].tolist() == ["007", "008"]
    assert t["R"].dtype == "float64"
    assert t["R"].tolist() == [0.5, 2.0]
    assert t["Mixed"].tolist() == ["x", "3"]
    assert t["Sel"].dtype == "boolean"
    assert t["Gap"].dtype == "float64"
    assert t["Gap"].isna().tolist() == [False, True]
    assert str(t["When"].dtype).startswith("datetime64")


def test_whole_numbers_with_a_gap_are_float(xlsx) -> None:
    s = xlsx([[*H, "N"], [*row(), 1], [*row(image="b"), None]])
    assert typed_user_columns(s)["N"].dtype == "float64"


def test_text_column_writes_whole_floats_and_bools_plainly(xlsx) -> None:
    s = xlsx(
        [
            [*H, "M"],
            [*row(), 2.0],
            [*row(image="b"), True],
            [*row(image="c"), "x"],
            [*row(image="d"), None],
        ]
    )
    assert typed_user_columns(s)["M"].tolist() == ["2", "TRUE", "x", None]


@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        (["20", "100"], "int64"),
        (["0", "-3"], "int64"),
        (["0.5", "-3"], "float64"),
        (["-07", "1"], "object"),
        (["1e3x", "1"], "object"),
        (["nan", "1"], "object"),
        (["inf", "1"], "object"),
    ],
)
def test_user_columns_csv_text_typing(
    tmp_path: Path, values: list[str], dtype: str
) -> None:
    path = write_csv(
        tmp_path / "images.csv",
        [[*H, "T"], [*row(), values[0]], [*row(image="b"), values[1]]],
    )
    t = typed_user_columns(validate(read_grid(path)))
    assert t["T"].dtype == dtype
    if dtype == "object":
        assert t["T"].tolist() == values
