"""agglpy.io.read_particles and write_particles.

Replaces the CSV importer tests of tests/legacy/test_particle_csv.py
for the new API (that file stays until roadmap 2.9).
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agglpy.errors import (
    DuplicateParticlesWarning,
    FileFormatError,
    NonCircularParticlesWarning,
    ParamsError,
    ParticleTableError,
)
from agglpy.io import read_particles, write_particles
from agglpy.tables import PARTICLE_COLUMNS, make_particles

from support.paths import DATA_DIR

INPUT = DATA_DIR / "input"
D7_017_FITTING = (
    INPUT / "multiple_image" / "images" / "D7-017" / "D7-017_fitting.csv"
)
D7_017_V00 = INPUT / "multiple_image" / "images" / "D7-017" / "D7-017_HCT.csv"
V04_FILE = INPUT / "PPcsv_small" / "set1_HCT.csv"

# Header and one row per format
AGGLPY = ("id,x,y,r", "1,100.5,100.5,30.0")
IMAGEJ_HEADER = " ,Index,Name,Type,Group,X,Y,Width,Height"
IMAGEJ = (IMAGEJ_HEADER, "1,0,0001-0100,Oval,none,70,70,60,60")
V04 = ("ID,X,Y,R", "0,100.5,100.5,30.0")
V00 = (
    "ID,X (pixels),Y (pixels),Radius (pixels)",
    "0,100.5,100.5,30.0",
)


@pytest.fixture(autouse=True)
def _no_unexpected_warnings():
    # every expected warning is caught with pytest.warns
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


def write_csv(path: Path, lines: list[str], bom: bool = False) -> Path:
    """Write lines as a UTF-8 file, optionally with a byte order mark."""
    path.write_text(
        "\n".join(lines) + "\n", encoding="utf-8-sig" if bom else "utf-8"
    )
    return path


def _imagej(path: Path, rows: list[str]) -> Path:
    return write_csv(path, [IMAGEJ_HEADER, *rows])


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        (AGGLPY, (1, 100.5, 100.5, 30.0)),
        # ImageJ: edge -> centre, id = Index + 1
        (IMAGEJ, (1, 99.5, 99.5, 30.0)),
        (V04, (0, 100.5, 100.5, 30.0)),
        (V00, (0, 100.5, 100.5, 30.0)),
    ],
    ids=["agglpy", "imagej", "agglpy_v04", "agglpy_v00"],
)
def test_each_format_is_recognised_and_converted(
    tmp_path: Path, lines: tuple[str, str], expected: tuple
) -> None:
    table = read_particles(write_csv(tmp_path / "p.csv", list(lines)))
    assert list(table.columns) == list(PARTICLE_COLUMNS)
    assert tuple(table.iloc[0]) == expected
    assert table["id"].dtype == np.int64


@pytest.mark.parametrize(
    "lines", [AGGLPY, IMAGEJ, V04, V00], ids=["agglpy", "imagej", "v04", "v00"]
)
def test_byte_order_mark_is_tolerated(
    tmp_path: Path, lines: tuple[str, str]
) -> None:
    # Excel writes one; it must not stick to the first column name
    path = write_csv(tmp_path / "p.csv", list(lines), bom=True)
    assert len(read_particles(path)) == 1


def test_file_is_read_as_utf8(tmp_path: Path) -> None:
    # an extra text column with non-ASCII text is ignored, not garbled
    path = write_csv(
        tmp_path / "p.csv", ["id,x,y,r,note", "1,10,10,5,µm żółw"]
    )
    assert read_particles(path)["x"].tolist() == [10.0]


def test_format_argument_skips_recognition(tmp_path: Path) -> None:
    # both column sets: recognition is ambiguous, format= decides
    path = write_csv(
        tmp_path / "p.csv", ["id,x,y,r,ID,X,Y,R", "1,10,10,5,0,99,99,9"]
    )
    with pytest.raises(FileFormatError, match=r"p\.csv.*format="):
        read_particles(path)
    assert read_particles(path, format="agglpy")["x"].tolist() == [10.0]
    assert read_particles(path, format="agglpy_v04")["x"].tolist() == [99.0]


def test_format_argument_still_requires_its_columns(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "p.csv", list(AGGLPY))
    with pytest.raises(FileFormatError, match=r"p\.csv.*Index"):
        read_particles(path, format="imagej")


def test_unknown_format_name_is_a_params_error(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "p.csv", list(AGGLPY))
    with pytest.raises(ParamsError, match="'agglpy', 'imagej'"):
        read_particles(path, format="fiji")


def test_unknown_header_names_the_public_formats(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "p.csv", ["a,b,c", "1,2,3"])
    with pytest.raises(FileFormatError, match=r"p\.csv.*'agglpy', 'imagej'"):
        read_particles(path)


def test_recognition_is_case_sensitive(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "p.csv", ["Id,X,Y,r", "1,10,10,5"])
    with pytest.raises(FileFormatError):
        read_particles(path)


@pytest.mark.parametrize(
    "lines", [AGGLPY, IMAGEJ, V04, V00], ids=["agglpy", "imagej", "v04", "v00"]
)
def test_missing_column_is_named(
    tmp_path: Path, lines: tuple[str, str]
) -> None:
    header, row = lines
    path = write_csv(
        tmp_path / "p.csv", [header.rsplit(",", 1)[0], row.rsplit(",", 1)[0]]
    )
    with pytest.raises(FileFormatError, match=r"p\.csv"):
        read_particles(path)


def test_extra_columns_are_ignored(tmp_path: Path) -> None:
    # files from later versions with more columns still read
    path = write_csv(
        tmp_path / "p.csv",
        ["note,id,x,y,r,agglomerate_id", "a,1,10,10,5,7", "b,2,90,90,5,8"],
    )
    table = read_particles(path)
    assert list(table.columns) == list(PARTICLE_COLUMNS)
    assert table["id"].tolist() == [1, 2]


def test_agglpy_ids_are_kept_exactly(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "p.csv", ["id,x,y,r", "17,10,10,5", "3,90,90,5"]
    )
    assert read_particles(path)["id"].tolist() == [17, 3]


def test_real_imagej_file() -> None:
    # Fiji ROI Manager export: 1003 circles, the first ROI is
    # Index 0, X 2674, Width 54 -> id 1, x 2700.5 (the HCT centre)
    table = read_particles(D7_017_FITTING)
    assert len(table) == 1003
    assert tuple(table.iloc[0]) == (1, 2700.5, 2005.5, 27.0)
    assert table["id"].tolist() == list(range(1, 1004))


def test_real_v04_file() -> None:
    table = read_particles(V04_FILE)
    assert len(table) == 20 and table["id"].tolist() == list(range(20))


def test_real_v00_file_is_numbered_by_row() -> None:
    # its ids restart at 0 for the second HCT range (rows 483-521); the
    # two ranges found 3 circles twice: a real detection mistake
    with pytest.warns(DuplicateParticlesWarning, match="3 pairs"):
        table = read_particles(D7_017_V00)
    assert table["id"].tolist() == list(range(522))
    assert tuple(table.iloc[0]) == (0, 2700.5, 2005.5, 26.6)


def test_v00_ids_are_row_numbers(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "p.csv",
        [V00[0], "5,10,10,5", "0,90,90,5", "0,190,190,5"],
    )
    assert read_particles(path)["id"].tolist() == [0, 1, 2]


def test_imagej_shift_matches_hct_centre(tmp_path: Path) -> None:
    # HCT x = 2700.5 and the ImageJ oval X = 2674, Width = 54 are the
    # same centre (D-051)
    path = _imagej(tmp_path / "p.csv", ["1,0,a,Oval,none,2674,1978,54,54"])
    assert tuple(read_particles(path).iloc[0, 1:]) == (2700.5, 2004.5, 27.0)


def test_imagej_negative_position_is_kept(tmp_path: Path) -> None:
    # a partly visible particle drawn over the left edge
    path = _imagej(tmp_path / "p.csv", ["1,0,a,Oval,none,-20,100,50,50"])
    assert read_particles(path)["x"].tolist() == [4.5]


def test_near_circles_are_read_as_circles_and_warned(tmp_path: Path) -> None:
    path = _imagej(
        tmp_path / "p.csv",
        [
            "1,0,a,Oval,none,0,0,60,60",
            "2,1,b,Oval,none,100,0,30,31",  # |W - H| = 1: a circle
            "3,2,c,Oval,none,200,0,31,30",
        ],
    )
    with pytest.warns(NonCircularParticlesWarning) as record:
        table = read_particles(path)
    assert len(record) == 1  # one warning per file
    message = str(record[0].message)
    assert "p.csv: 3 ROIs" in message
    assert "2 near-circular ovals read as circles" in message
    assert "ids 2, 3" in message
    assert "dropped" not in message
    # r = (W + H) / 4; centre from each side's own half
    assert table["r"].tolist() == [30.0, 15.25, 15.25]
    assert table["y"].tolist() == [29.5, 15.0, 14.5]


def test_non_circular_rois_are_dropped_and_warned(tmp_path: Path) -> None:
    path = _imagej(
        tmp_path / "p.csv",
        [
            "1,0,a,Oval,none,0,0,60,60",
            "2,1,b,Oval,none,100,0,60,40",  # ellipse
            "3,2,c,Rectangle,none,200,0,60,60",
            "4,3,d,Oval,none,300,0,20,20",
            "5,4,e,Oval,none,400,0,30,31",  # near-circle
        ],
    )
    with pytest.warns(NonCircularParticlesWarning) as record:
        table = read_particles(path)
    assert len(record) == 1
    message = str(record[0].message)
    assert "5 ROIs" in message
    assert "1 near-circular ovals read as circles" in message
    assert "2 non-circular ROIs dropped" in message
    assert "2: Oval 60 x 40" in message and "3: Rectangle 60 x 60" in message
    assert table["id"].tolist() == [1, 4, 5]


def test_long_lists_in_the_warning_are_cut(tmp_path: Path) -> None:
    rows = [f"{i},{i},n,Line,none,{10 * i},0,5,1" for i in range(13)]
    path = _imagej(tmp_path / "p.csv", rows)
    with pytest.warns(NonCircularParticlesWarning, match="and 3 more"):
        table = read_particles(path)
    assert len(table) == 0  # all dropped: an empty table


def test_imagej_circles_only_give_no_warning() -> None:
    read_particles(D7_017_FITTING)  # the fixture makes a warning fail


@pytest.mark.parametrize("header", [AGGLPY[0], IMAGEJ_HEADER, V04[0], V00[0]])
def test_header_only_file_gives_an_empty_table(
    tmp_path: Path, header: str
) -> None:
    table = read_particles(write_csv(tmp_path / "p.csv", [header]))
    assert len(table) == 0
    assert list(table.columns) == list(PARTICLE_COLUMNS)
    assert table["id"].dtype == np.int64 and table["x"].dtype == np.float64


def test_empty_file_is_a_file_format_error(tmp_path: Path) -> None:
    path = tmp_path / "p.csv"
    path.write_bytes(b"")
    with pytest.raises(FileFormatError, match=r"p\.csv"):
        read_particles(path)


@pytest.mark.parametrize(
    "row",
    [
        "1,10,10,0",  # r = 0
        "1,10,,5",  # missing y
        "1,ten,10,5",  # text
        "1.5,10,10,5",  # fractional id
    ],
)
def test_schema_breaking_values_name_the_file(
    tmp_path: Path, row: str
) -> None:
    path = write_csv(tmp_path / "p.csv", ["id,x,y,r", row])
    with pytest.raises(FileFormatError, match=r"p\.csv") as info:
        read_particles(path)
    assert isinstance(info.value.__cause__, ParticleTableError)


def test_repeated_ids_name_the_file(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "p.csv", ["id,x,y,r", "1,10,10,5", "1,90,90,5"]
    )
    with pytest.raises(FileFormatError, match=r"p\.csv.*unique"):
        read_particles(path)


def test_non_utf8_file_is_a_file_format_error(tmp_path: Path) -> None:
    path = tmp_path / "p.csv"
    path.write_bytes("id,x,y,r,note\n1,10,10,5,µm\n".encode("cp1252"))
    with pytest.raises(FileFormatError, match=r"p\.csv.*UTF-8"):
        read_particles(path)


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_particles(tmp_path / "missing.csv")


def test_semicolon_file_names_its_header(tmp_path: Path) -> None:
    # Excel in a comma-decimal locale saves ';'-separated files
    path = write_csv(tmp_path / "p.csv", ["id;x;y;r", "1;10,5;10;5"])
    with pytest.raises(FileFormatError, match=r"id;x;y;r"):
        read_particles(path)


def test_imagej_row_without_a_size_is_dropped_and_named(
    tmp_path: Path,
) -> None:
    path = _imagej(
        tmp_path / "p.csv",
        ["1,0,a,Oval,none,0,0,60,60", "2,1,b,Oval,none,100,0,,"],
    )
    with pytest.warns(NonCircularParticlesWarning, match="2: Oval nan x nan"):
        table = read_particles(path)
    assert table["id"].tolist() == [1]


def test_shape_warning_points_at_the_caller(tmp_path: Path) -> None:
    path = _imagej(tmp_path / "p.csv", ["1,0,a,Rectangle,none,0,0,6,6"])
    with pytest.warns(NonCircularParticlesWarning) as record:
        read_particles(path)
    assert record[0].filename == __file__


def test_exact_round_trip_of_random_floats(tmp_path: Path) -> None:
    # pandas writes the shortest decimal that reads back to the same
    # float; the default parser would miss the last digit for 17 % of
    # these values, round_trip for none
    rng = np.random.default_rng(2026)
    n = 100_000
    # random digits on an 8-px grid: no near-identical circles
    cell = np.arange(n)
    table = make_particles(
        (cell % 400) * 8 + rng.uniform(-1, 1, n) - 50,
        (cell // 400) * 8 + rng.uniform(-1, 1, n) - 50,
        rng.uniform(0.5, 120, n),
        ids=rng.permutation(n) * 3 + 7,
    )
    write_particles(table, tmp_path / "p.csv")
    pd.testing.assert_frame_equal(
        read_particles(tmp_path / "p.csv"), table, check_exact=True
    )


def test_written_bytes_are_fixed(tmp_path: Path) -> None:
    # one header, comma, '.', no index, integer ids, LF on every OS,
    # no byte order mark
    table = make_particles([2700.5, 0.1], [2005.5, -3.0], [26.6, 1e-3])
    write_particles(table, tmp_path / "p.csv")
    assert (tmp_path / "p.csv").read_bytes() == (
        b"id,x,y,r\n1,2700.5,2005.5,26.6\n2,0.1,-3.0,0.001\n"
    )


def test_only_the_schema_columns_are_written(tmp_path: Path) -> None:
    table = make_particles([10.0], [10.0], [5.0])
    table.insert(0, "note", ["x"])
    table["agglomerate_id"] = [4]
    write_particles(table, tmp_path / "p.csv")
    assert (tmp_path / "p.csv").read_text().splitlines()[0] == "id,x,y,r"


def test_empty_table_writes_a_header(tmp_path: Path) -> None:
    write_particles(make_particles([], [], []), tmp_path / "p.csv")
    assert (tmp_path / "p.csv").read_bytes() == b"id,x,y,r\n"
    assert len(read_particles(tmp_path / "p.csv")) == 0


def test_existing_file_is_never_overwritten_by_default(
    tmp_path: Path,
) -> None:
    path = write_csv(tmp_path / "p.csv", ["keep me"])
    with pytest.raises(FileExistsError):
        write_particles(make_particles([1.0], [1.0], [1.0]), path)
    assert path.read_text() == "keep me\n"
    write_particles(make_particles([1.0], [1.0], [1.0]), path, overwrite=True)
    assert path.read_text().startswith("id,x,y,r")


def test_bad_table_writes_nothing(tmp_path: Path) -> None:
    bad = pd.DataFrame({"id": [1], "x": [1.0], "y": [1.0], "r": [-1.0]})
    with pytest.raises(ParticleTableError):
        write_particles(bad, tmp_path / "p.csv")
    assert not (tmp_path / "p.csv").exists()


def test_missing_folder_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        write_particles(
            make_particles([1.0], [1.0], [1.0]), tmp_path / "no" / "p.csv"
        )


def test_writing_does_not_warn_about_duplicates(tmp_path: Path) -> None:
    # the duplicate warning belongs to creating or reading a table
    with pytest.warns(Warning):
        table = make_particles([10.0, 10.5], [10.0, 10.0], [5.0, 5.0])
    write_particles(table, tmp_path / "p.csv")  # fixture: no warning


def test_duplicate_warning_names_the_file_and_points_at_the_caller(
    tmp_path: Path,
) -> None:
    # a notebook reading many files must see which one has duplicates
    path = write_csv(
        tmp_path / "p.csv", ["id,x,y,r", "1,10,10,5", "2,10.5,10,5"]
    )
    with pytest.warns(DuplicateParticlesWarning, match=r"p\.csv: 1 pairs"):
        read_particles(path)
    with pytest.warns(DuplicateParticlesWarning) as record:
        read_particles(path)
    assert record[0].filename == __file__


@pytest.mark.parametrize(
    "row",
    ["1,1,5,2,3", "1,10,20,3,"],
    ids=["decimal comma", "trailing comma"],
)
def test_rows_longer_than_the_header_are_rejected(
    tmp_path: Path, row: str
) -> None:
    # pandas would take the first field as the index and shift every
    # value one column left: wrong geometry without an error
    path = write_csv(tmp_path / "p.csv", ["id,x,y,r", row])
    with pytest.raises(FileFormatError, match=r"p\.csv.*more fields"):
        read_particles(path)


@pytest.mark.parametrize("column", ["Index", "X", "Y"])
def test_imagej_empty_position_names_the_file_column_and_line(
    tmp_path: Path, column: str
) -> None:
    # the error names the column as the file has it, and the line
    rows = ["1,0,a,Oval,none,0,0,60,60", "2,1,b,Oval,none,100,0,60,60"]
    position = {"Index": 1, "X": 5, "Y": 6}[column]
    fields = rows[1].split(",")
    fields[position] = ""
    path = _imagej(tmp_path / "p.csv", [rows[0], ",".join(fields)])
    with pytest.raises(FileFormatError, match=rf"p\.csv.*'{column}'.*line 3"):
        read_particles(path)


def test_large_imagej_ids_are_printed_as_integers(tmp_path: Path) -> None:
    path = _imagej(tmp_path / "p.csv", ["1,1234566,a,Rectangle,none,0,0,6,6"])
    with pytest.warns(NonCircularParticlesWarning, match="ids 1234567: "):
        read_particles(path)


def test_format_argument_error_shows_the_header(tmp_path: Path) -> None:
    # a ';' file read with format= shows why the columns are missing,
    # as auto-detection does (no comma in it: one field per row)
    path = write_csv(tmp_path / "p.csv", ["id;x;y;r", "1;10;10;5"])
    with pytest.raises(FileFormatError, match=r"header.*id;x;y;r"):
        read_particles(path, format="agglpy")
