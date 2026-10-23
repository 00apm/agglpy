"""Image sheets for tests: xlsx and CSV files from rows of cells."""

import csv
from pathlib import Path

import openpyxl


def write_xlsx(
    path: Path, rows: list[list[object]], *, sheets: int = 1
) -> Path:
    """Write ``rows`` (header first) into a new workbook at ``path``.

    ``sheets`` > 1 adds empty worksheets after the first.
    """
    book = openpyxl.Workbook()
    sheet = book.active
    for row in rows:
        sheet.append(row)
    for _ in range(sheets - 1):
        book.create_sheet()
    book.save(path)
    return path


def write_csv(
    path: Path,
    rows: list[list[object]],
    *,
    bom: bool = False,
    newline: str = "\n",
) -> Path:
    """Write ``rows`` (header first) as UTF-8 CSV; None is empty."""
    encoding = "utf-8-sig" if bom else "utf-8"
    with path.open("w", encoding=encoding, newline="") as fh:
        writer = csv.writer(fh, lineterminator=newline)
        for row in rows:
            writer.writerow("" if cell is None else cell for cell in row)
    return path
