"""``Collection``: a folder of what to analyse and how."""

import contextlib
import os
import shutil
import stat
import unicodedata
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import pandas as pd

from agglpy.app._analyze import run as run_analyze
from agglpy.app._config import ImageParams, template
from agglpy.app._config_file import CONFIG_NAME, ConfigFile
from agglpy.app._correction import run_import
from agglpy.app._detect import run as run_detect
from agglpy.app._files import replace_file
from agglpy.app._image import Image
from agglpy.app._report import reemit
from agglpy.app._results import Results
from agglpy.app._sheet import (
    _NUMERIC_TEXT,
    FIXED_COLUMNS,
    SHEET_FILES,
    Entry,
    Grid,
    Sheet,
    find_sheet,
    read_grid,
    validate,
)
from agglpy.app._sheet_writer import _plain, new_sheet, write_sheet
from agglpy.app._sources import (
    PathLike,
    apply_rule,
    find_images,
    resolve_stored,
    stored_path,
)
from agglpy.app._status import table as status_table
from agglpy.app._work import (
    CORRECTED_CSV,
    EXPORTS_FOLDER,
    WORK_FOLDER,
    image_folder,
)
from agglpy.errors import (
    OutputExistsError,
    ParamsError,
    SheetError,
    TableError,
)
from agglpy.export import _check_replaceable
from agglpy.params import HCTParams, PreprocessParams, RegionParams

__all__ = ["Collection"]

_FORMATS = ("xlsx", "csv")
# a name, names, or a True / False Series on the sheet's rows
Images = str | Sequence[str] | pd.Series


class Collection:
    """A folder of images to analyse: what (the sheet) and how (params).

    The folder holds the image sheet ``images.xlsx`` or ``images.csv``
    (one row per image), ``agglpy.yml`` (the params) and ``agglpy/``
    (agglpy's work). The source images stay where they are; the sheet
    stores their paths relative to the folder.

    Parameters
    ----------
    folder : str or Path
        The collection folder; nothing is read or checked here.
        `open` checks the files, `create` makes a new collection.

    Notes
    -----
    The files are the only state: every call reads them again, so
    edits by hand and calls from a notebook never conflict. The column
    ``pixel_size_nm`` is in nm. Charts and pictures embedded in
    ``images.xlsx`` are not kept when agglpy writes the sheet. See
    *Collection* (``docs/collection.md``), section *The folder*.
    """

    def __init__(self, folder: PathLike) -> None:
        self._folder = Path(folder).resolve()

    # -----------------------------------------------------------------
    # Creating and opening
    # -----------------------------------------------------------------

    @classmethod
    def create(
        cls,
        folder: PathLike,
        images: PathLike | Sequence[PathLike] | None = None,
        *,
        pattern: str | Sequence[str] | None = None,
        columns_from_path: str | None = None,
        sheet_format: Literal["xlsx", "csv"] = "xlsx",
    ) -> "Collection":
        """Create a collection in `folder` from image folders or files.

        Writes the sheet (one row per image), ``agglpy.yml`` (the
        template) and the empty folder ``agglpy/``.

        Parameters
        ----------
        folder : str or Path
            The collection folder; created when missing.
        images : str or Path, or a list of them, optional
            Image folders and files, as in `add_images`; None gives a
            sheet without rows.
        pattern : str or list of str, optional
            As in `add_images`.
        columns_from_path : str, optional
            As in `add_images`.
        sheet_format : {"xlsx", "csv"}, default "xlsx"
            The sheet file: ``images.xlsx`` or ``images.csv``.

        Returns
        -------
        Collection
            The new collection.

        Raises
        ------
        OutputExistsError
            If `folder` already holds a sheet or ``agglpy.yml``;
            nothing is written.
        SheetError
            If image names clash (ignoring case) or are not valid;
            nothing is written.
        ParamsError
            If `sheet_format` is not ``"xlsx"`` or ``"csv"``, or
            `columns_from_path` is not a valid regular expression or
            names a reserved column; nothing is written.
        FileNotFoundError
            If a path in `images` does not exist; nothing is written.

        Warns
        -----
        UnmatchedPathsWarning
            If image paths do not match `columns_from_path`; their
            columns stay empty.

        Notes
        -----
        A write that fails removes what this call wrote and re-raises
        the error.
        """
        if sheet_format not in _FORMATS:
            raise ParamsError(
                f"sheet_format must be 'xlsx' or 'csv', not {sheet_format!r}"
            )
        col = cls(folder)
        existing = [
            name
            for name in (*SHEET_FILES, CONFIG_NAME)
            if (col.folder / name).exists()
        ]
        if existing:
            raise OutputExistsError(
                f"{col.folder} already holds {', '.join(existing)}; open it "
                "with Collection.open"
            )
        sheet_path = col.folder / f"images.{sheet_format}"
        header, rows = col._new_rows(
            sheet_path,
            (),
            list(FIXED_COLUMNS),
            images,
            pattern,
            columns_from_path,
        )
        validate(Grid(sheet_path, header, _numbered(rows)))
        config_path = col.folder / CONFIG_NAME
        made_folder = not col.folder.exists()
        try:
            col.folder.mkdir(parents=True, exist_ok=True)
            new_sheet(sheet_path, header, rows)
            replace_file(
                config_path,
                lambda tmp: tmp.write_bytes(template().encode()),
            )
            (col.folder / WORK_FOLDER).mkdir(exist_ok=True)
        except BaseException:
            # both files were checked absent above, so they are ours
            sheet_path.unlink(missing_ok=True)
            config_path.unlink(missing_ok=True)
            if made_folder and col.folder.is_dir():
                with contextlib.suppress(OSError):
                    col.folder.rmdir()  # only if empty
            raise
        return col

    @classmethod
    def open(cls, folder: PathLike) -> "Collection":
        """Open the collection in `folder`, checking both files.

        Parameters
        ----------
        folder : str or Path
            The collection folder.

        Returns
        -------
        Collection
            The collection.

        Raises
        ------
        FileNotFoundError
            If the sheet or ``agglpy.yml`` is missing.
        FileFormatError
            If the sheet or ``agglpy.yml`` does not read as xlsx,
            UTF-8 CSV or YAML.
        SheetError
            If the sheet has problems; the message lists every one.
        ParamsError
            If ``agglpy.yml`` has problems; the message lists every
            one.
        """
        col = cls(folder)
        col._read(check=True)
        return col

    @property
    def folder(self) -> Path:
        """The collection folder, absolute."""
        return self._folder

    @property
    def sheet_path(self) -> Path:
        """The image sheet's path: ``images.xlsx`` or ``images.csv``."""
        return find_sheet(self._folder)

    def __getitem__(self, name: str) -> Image:
        """Return a handle on image `name`, reading the files on access.

        Raises
        ------
        KeyError
            If `name` is not in the sheet.
        """
        sheet = validate(read_grid(self.sheet_path))
        if name not in sheet.names:
            raise KeyError(name)
        return Image(self, name)

    def __repr__(self) -> str:
        shown = self._folder.as_posix()
        try:
            count = len(read_grid(self.sheet_path).rows)
        except (OSError, ValueError):
            return f"Collection('{shown}')"
        noun = "image" if count == 1 else "images"
        return f"Collection('{shown}', {count} {noun})"

    # -----------------------------------------------------------------
    # The sheet
    # -----------------------------------------------------------------

    def read_sheet(self) -> pd.DataFrame:
        """Return the sheet's cells as stored, one row per image.

        Change the table and give it to `write_sheet`.

        Returns
        -------
        DataFrame
            The sheet's named columns, dtype ``object``: None for
            empty cells, xlsx cells as Excel stores them, CSV cells as
            text; index 0 … n-1.
        """
        grid = read_grid(self.sheet_path)
        named = [i for i, name in enumerate(grid.header) if name]
        return pd.DataFrame(
            [[cells[i] for i in named] for _, cells in grid.rows],
            columns=[grid.header[i] for i in named],
            dtype=object,
        )

    def write_sheet(self, table: pd.DataFrame) -> None:
        """Make the sheet hold `table`.

        Rows are matched by ``image`` (ignoring case): changed cells
        are set, rows left out removed, new rows appended. Everything
        else in the file is kept. The new content is checked first.

        Parameters
        ----------
        table : DataFrame
            The new content, as from `read_sheet`: one column per
            sheet column, ``image`` and ``file`` required.

        Raises
        ------
        SheetError
            If the new content has problems; the message lists every
            one, nothing is written.
        ParamsError
            If ``agglpy.yml`` has settings for an image the table
            leaves out, or has other problems; nothing is written.
        """
        sheet, config = self._read()
        header = [str(c) for c in table.columns]
        rows = [
            [_plain(v) for v in row]
            for row in table.itertuples(index=False, name=None)
        ]
        new = validate(Grid(sheet.path, header, _numbered(rows)))
        dropped = [n for n in config.images() if n not in new.names]
        dropped = [n for n in dropped if n in sheet.names]
        if dropped:
            raise ParamsError(
                "\n".join(
                    f"agglpy.yml has settings for {n}, which the new sheet "
                    f"does not have; remove them with reset_params({n!r}) "
                    "first"
                    for n in dropped
                )
            )
        config.check(new.names)
        write_sheet(sheet.path, header, rows)

    def include(self, images: Images) -> None:
        """Set ``include`` true for `images`.

        Their ``note`` stays. Cells already holding 1/0 get 1/0.

        Parameters
        ----------
        images : str, list of str or Series
            A name, a list of names, or a True / False Series on the
            rows of `status` or `read_sheet`; an empty selection
            writes nothing.

        Raises
        ------
        SheetError
            If the sheet has problems; nothing is written.
        ParamsError
            If a name is not in the sheet or ``agglpy.yml`` has
            problems; nothing is written.
        TableError
            If a mask is not on the sheet's rows; nothing is written.
        """
        self._set_include(images, True, None)

    def exclude(self, images: Images, *, note: str | None = None) -> None:
        """Set ``include`` false for `images`.

        Cells already holding 1/0 get 1/0.

        Parameters
        ----------
        images : str, list of str or Series
            As in `include`.
        note : str, optional
            The reason, written into the images' ``note``; None keeps
            their ``note``.

        Raises
        ------
        SheetError
            If the sheet has problems; nothing is written.
        ParamsError
            If a name is not in the sheet or ``agglpy.yml`` has
            problems; nothing is written.
        TableError
            If a mask is not on the sheet's rows; nothing is written.
        """
        self._set_include(images, False, note)

    def _set_include(
        self, images: Images, value: bool, note: str | None
    ) -> None:
        sheet = validate(read_grid(self.sheet_path))
        names = _names(sheet, images)
        if not names:
            return
        table = self.read_sheet()
        # Rows from the checked names: a cell may hold 17 for "17"
        rows = pd.Series(
            [e.image in names for e in sheet.entries], index=table.index
        )
        cells = [
            _include_cell(old, value) if row else old
            for old, row in zip(table["include"], rows)
        ]
        table["include"] = pd.Series(cells, index=table.index, dtype=object)
        if note is not None:
            table["note"] = table["note"].mask(rows, note)
        self.write_sheet(table)

    def add_images(
        self,
        images: PathLike | Sequence[PathLike],
        *,
        pattern: str | Sequence[str] | None = None,
        columns_from_path: str | None = None,
    ) -> list[str]:
        """Append rows for image files not yet in the sheet.

        Files already in the sheet are skipped; nothing typed into the
        sheet changes. The collection's own ``agglpy/`` and
        ``exports/`` folders are left out.

        Parameters
        ----------
        images : str or Path, or a list of them
            A folder (its image files, directly inside it), a file
            (taken as is) or a list of them.
        pattern : str or list of str, optional
            Glob(s) replacing the folders' default, e.g.
            ``"**/*.tif"`` for subfolders; None takes the files with
            an image suffix directly inside each folder.
        columns_from_path : str, optional
            A regular expression with named groups, searched in each
            file's path relative to the folder given
            (``/``-separated): the groups fill new columns, the group
            ``image`` sets the image name (else the file stem).

        Returns
        -------
        list of str
            The new image names, in order.

        Raises
        ------
        SheetError
            If image names clash (ignoring case) or are not valid;
            nothing is written.
        ParamsError
            If `columns_from_path` is not a valid regular expression
            or names a reserved column; nothing is written.
        FileNotFoundError
            If a path in `images` does not exist; nothing is written.

        Warns
        -----
        UnmatchedPathsWarning
            If image paths do not match `columns_from_path`; their
            columns stay empty.
        """
        sheet, _ = self._read()
        grid = read_grid(sheet.path)
        named = [i for i, name in enumerate(grid.header) if name]
        header = [grid.header[i] for i in named]
        old = [[cells[i] for i in named] for _, cells in grid.rows]
        new_header, rows = self._new_rows(
            sheet.path,
            sheet.entries,
            header,
            images,
            pattern,
            columns_from_path,
        )
        if not rows:
            return []
        width = len(new_header)
        old = [row + [None] * (width - len(row)) for row in old]
        validate(Grid(sheet.path, new_header, _numbered(old + rows)))
        write_sheet(sheet.path, new_header, old + rows)
        column = new_header.index("image")
        return [str(row[column]) for row in rows]

    def remove_images(
        self, images: Images, *, force: bool = False
    ) -> list[str]:
        """Take `images` out of the collection.

        Removes each image's sheet row, its entry in ``agglpy.yml`` and
        its work folder ``agglpy/<name>/``. The image file is never
        touched: `add_images` on its folder adds it again. `exclude`
        keeps an image in the collection without using it.

        Parameters
        ----------
        images : str, list of str or Series
            As in `include`.
        force : bool, default False
            Remove images with a manual correction too, deleting
            their ``particles_corrected.csv``.

        Returns
        -------
        list of str
            The removed names, in sheet order.

        Raises
        ------
        ParamsError
            If a name is not in the sheet or ``agglpy.yml`` has
            problems; nothing is changed.
        TableError
            If a mask is not on the sheet's rows; nothing is changed.
        OutputExistsError
            If an image has a manual correction
            (``particles_corrected.csv``) and `force` is False;
            nothing is changed.
        OutputLockedError
            If the sheet, ``agglpy.yml`` or a file in a work folder is
            held open; nothing is changed.
        """
        sheet, config = self._read()
        names = _names(sheet, images)
        if not names:
            return []
        folders = [image_folder(self._folder, n) for n in names]
        if not force:
            corrected = [
                f"{n} has a manual correction "
                f"({WORK_FOLDER}/{n}/{CORRECTED_CSV}); "
                "remove_images(..., force=True) deletes it too"
                for n, f in zip(names, folders)
                if (f / CORRECTED_CSV).is_file()
            ]
            if corrected:
                raise OutputExistsError("\n".join(corrected))
        grid = read_grid(sheet.path)
        gone = {e.row for e in sheet.entries if e.image in names}
        named = [i for i, name in enumerate(grid.header) if name]
        header = [grid.header[i] for i in named]
        rows = [
            [cells[i] for i in named]
            for number, cells in grid.rows
            if number not in gone
        ]
        new = validate(Grid(sheet.path, header, _numbered(rows)))
        for name in names:
            config.reset(name)
        config.check(new.names)
        folders = [f for f in folders if f.is_dir()]
        files = [p for f in folders for p in f.rglob("*") if p.is_file()]
        # a folder can't be renamed while a program works inside it
        _check_replaceable([sheet.path, config.path, *files, *folders])
        # the sheet first: a stop after it leaves a yml entry that every
        # call reports, never params dropped silently
        write_sheet(sheet.path, header, rows)
        self._write_config(new, config)
        for path in files:  # Windows refuses to delete read-only files
            path.chmod(stat.S_IWRITE | stat.S_IREAD)
        for folder in folders:
            shutil.rmtree(folder)
        return names

    def _new_rows(
        self,
        sheet_path: Path,
        entries: Sequence[Entry],
        header: list[str],
        images: PathLike | Sequence[PathLike] | None,
        pattern: str | Sequence[str] | None,
        rule: str | None,
    ) -> tuple[list[str], list[list[object]]]:
        """Return the header and the rows of the files not in the sheet.

        Raises SheetError when names clash; the caller validates the
        rows.
        """
        known = {
            _same_file(resolve_stored(e.file, self._folder)) for e in entries
        }
        # the collection's own folders hold no new images
        skip = (self._folder / WORK_FOLDER, self._folder / EXPORTS_FOLDER)
        found = (
            [] if images is None else find_images(images, pattern, skip=skip)
        )
        found = [f for f in found if _same_file(f.path) not in known]
        names, columns, values = apply_rule(found, rule)
        files = [stored_path(f.path, self._folder) for f in found]
        _check_clashes(
            [(e.image, e.file) for e in entries], list(zip(names, files))
        )
        header = header + [c for c in columns if c not in header]
        typed: Sequence[Mapping[str, object]] = (
            _typed_values(values, columns)
            if sheet_path.suffix == ".xlsx"
            else values
        )
        rows: list[list[object]] = []
        for name, file, row_values in zip(names, files, typed):
            cells: dict[str, object] = {"image": name, "file": file}
            cells.update(row_values)
            rows.append([cells.get(c) for c in header])
        return header, rows

    # -----------------------------------------------------------------
    # Params
    # -----------------------------------------------------------------

    def params(self, image: str | None = None) -> ImageParams:
        """Return the params of `image`.

        Parameters
        ----------
        image : str, optional
            The image name; None gives the defaults.

        Returns
        -------
        ImageParams
            The defaults with the image's entry in ``agglpy.yml`` laid
            over them.

        Raises
        ------
        ParamsError
            If `image` is not in the sheet or ``agglpy.yml`` has
            problems.
        """
        sheet, config = self._read(check=True)
        if image is not None:
            _known(sheet, image)
        return config.params(image)

    def set_params(
        self,
        image: str | None,
        *,
        region: Mapping[str, object] | RegionParams | None = None,
        preprocess: Mapping[str, object] | PreprocessParams | None = None,
        hct: Mapping[str, object] | HCTParams | None = None,
    ) -> None:
        """Write params for `image` into ``agglpy.yml``.

        An image's entry keeps only what differs from the defaults.
        Every image is checked before writing.

        Parameters
        ----------
        image : str or None
            The image name; None writes the defaults.
        region : mapping or RegionParams, optional
            The ``region`` section: a mapping is merged into what is
            there, a params object replaces the section; None leaves
            it as it is.
        preprocess : mapping or PreprocessParams, optional
            The ``preprocess`` section, as `region`.
        hct : mapping or HCTParams, optional
            The ``hct`` section, as `region`.

        Raises
        ------
        ParamsError
            If `image` is not in the sheet or the result has problems;
            nothing is written.
        """
        sheet, config = self._read()
        if image is not None:
            _known(sheet, image)
        given = {"region": region, "preprocess": preprocess, "hct": hct}
        config.set(image, {k: v for k, v in given.items() if v is not None})
        self._write_config(sheet, config)

    def reset_params(self, image: str, key: str | None = None) -> None:
        """Remove `key`, or the whole entry, from the params of `image`.

        The image then takes that value from the defaults. An entry for
        an image the sheet no longer has can be removed too.

        Parameters
        ----------
        image : str
            The image name.
        key : str, optional
            A dotted params key, e.g. ``"preprocess.rolling_ball"``;
            None removes the image's whole entry.

        Raises
        ------
        ParamsError
            If `image` has neither a row nor an entry, `key` is not a
            params key, or the result has problems; nothing is
            written.
        """
        sheet, config = self._read()
        if image not in config.images():
            _known(sheet, image)
        config.reset(image, key)
        self._write_config(sheet, config)

    # -----------------------------------------------------------------
    # Particles
    # -----------------------------------------------------------------

    def detect(
        self,
        images: Images | None = None,
        *,
        force: bool = False,
        progress: bool = True,
    ) -> pd.DataFrame:
        """Detect particles (HCT) on the images, one after another.

        Writes ``agglpy/<name>/particles_hct.csv`` and its record per
        image. An image is skipped while its record matches its params
        and image file. ``particles_corrected.csv`` is never touched.

        Parameters
        ----------
        images : str, list of str or Series, optional
            As in `include`, images with ``include`` false too; None
            takes every included image.
        force : bool, default False
            Detect again images that are up to date or have a
            correction, and replace a ``particles_hct.csv`` changed by
            a person.
        progress : bool, default True
            Show a progress bar over the images.

        Returns
        -------
        DataFrame
            One row per image in sheet order: ``image``, ``outcome``
            (``detected``, ``replaced``, ``skipped``, ``kept`` or
            ``error``), ``particle_count`` (when written),
            ``message``, ``warnings``.

        Raises
        ------
        ParamsError
            If a name is not in the sheet or ``agglpy.yml`` has
            problems; nothing runs.
        TableError
            If a mask is not on the sheet's rows; nothing runs.

        Warns
        -----
        IncompleteRunWarning
            If an image ended ``kept``, ``error`` or ``skipped`` for
            any reason but ``up to date``.
        LossyImageWarning
            If an image file is JPEG-compressed.

        Notes
        -----
        An image that fails is left as it was; the other images go on.
        Warnings of each image go into its record and the ``warnings``
        column; after the batch, each warning class is raised once,
        naming the images. See *Collection* (``docs/collection.md``),
        section *Particle sets: detection and correction*.
        """
        sheet, config = self._read(check=True)
        if images is None:
            names = [e.image for e in sheet.entries if e.include]
        else:
            names = _names(sheet, images)
        table, summaries = run_detect(
            self._folder, sheet, config, names, force=force, progress=progress
        )
        # after every file is written, so a filter set to "error" raises
        # only once the batch is done
        for cls, message in summaries:
            warnings.warn(message, cls, stacklevel=2)
        return table

    def import_particles(
        self,
        name: str,
        path: PathLike,
        *,
        format: str = "auto",
        force: bool = False,
    ) -> pd.DataFrame:
        """Import a corrected particle file for image `name`.

        Writes it as ``agglpy/<name>/particles_corrected.csv`` with
        its record; from then on the image uses it. The source file is
        only read.

        Parameters
        ----------
        name : str
            The image name.
        path : str or Path
            The corrected particle file: an ImageJ / Fiji ROI CSV, an
            agglpy file or an older agglpy file.
        format : {"auto", "agglpy", "imagej"}, default "auto"
            The file format, as in `agglpy.io.read_particles`.
        force : bool, default False
            Replace a ``particles_corrected.csv`` changed by a person.

        Returns
        -------
        DataFrame
            A one-row table like `detect`'s; ``outcome`` is
            ``imported`` or ``replaced``.

        Raises
        ------
        ParamsError
            If `name` is not in the sheet or `format` is not a known
            format name.
        FileNotFoundError
            If `path` doesn't exist.
        FileFormatError
            If `path` is not a particle file.
        OutputExistsError
            If ``particles_corrected.csv`` was changed by a person and
            `force` is False; nothing is written.

        Warns
        -----
        UserWarning
            If the correction keeps less than half of the detected
            circles: it is then shown as not based on the detection.
        NonCircularParticlesWarning
            If ImageJ shapes were read as circles or dropped.
        DuplicateParticlesWarning
            If the file holds near-identical circles.

        Notes
        -----
        Works without the image file and without a detection. The
        record names the source and the detection the correction
        started from, with how many detected circles it kept. See
        *Collection* (``docs/collection.md``), section *Particle sets:
        detection and correction*.
        """
        sheet, _ = self._read()
        _known(sheet, name)
        entry = next(e for e in sheet.entries if e.image == name)
        table, caught = run_import(
            self._folder, entry, Path(path), format=format, force=force
        )
        reemit(caught, stacklevel=3)
        return table

    def status(self) -> pd.DataFrame:
        """Report where each image's particles come from.

        Writes nothing. Images are hashed again only when their size or
        time differs from the record; particle files always.

        Returns
        -------
        DataFrame
            One row per sheet image, in sheet order (index 0 … n-1):
            ``image``, ``include``, ``file``, ``pixel_size_nm``,
            ``pixel_size_source``, ``particle_set``,
            ``particle_count``, ``outside_image`` (circles entirely
            outside the image), ``detection``, ``correction``,
            ``kept`` and ``warnings`` (of the set in use).

        Raises
        ------
        ParamsError
            If ``agglpy.yml`` has problems.

        Warns
        -----
        LeftoverFoldersWarning
            If ``agglpy/`` holds folders of images the sheet does not
            list.
        UnreadableRecordWarning
            If a record does not read; it counts as missing.
        UserWarning
            If a particle CSV does not read.

        Notes
        -----
        The values of ``file``, ``pixel_size_source``,
        ``particle_set``, ``detection`` and ``correction``: see
        *Collection* (``docs/collection.md``), section *Status*.
        """
        sheet, config = self._read(check=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            table = status_table(self._folder, sheet, config)
        reemit(caught, stacklevel=3)
        return table

    def analyze(self, *, progress: bool = True) -> Results:
        """Group and measure the particles of every included image.

        Reads the sheet, ``agglpy.yml``, each image's particle set in
        use (``corrected`` if there is one, else ``hct``) and the image
        metadata (no pixels). Each image is converted from px with its
        own pixel size. Writes nothing; run `analyze` again after a
        change.

        Parameters
        ----------
        progress : bool, default True
            Show a progress bar over the images.

        Returns
        -------
        Results
            Every table in SI units (m, m², m³); a snapshot that never
            reads the collection again.

        Raises
        ------
        ParamsError
            If ``agglpy.yml`` has problems.
        NotReadyError
            If no image is included, or an included image has no
            pixel size, no particles, or a file that is missing or
            does not read; nothing is computed, the message lists
            every such image.

        Warns
        -----
        OutdatedParticlesWarning
            First, if a detection in use was made with other params or
            another image file: the results don't reflect the current
            params.
        ParticlesOutsideRegionWarning
            If circles lying entirely outside the region were left out
            (counted in ``outside_region``).
        DuplicateParticlesWarning
            If an enclosed particle lies inside a near-identical
            circle; it counts as enclosed.
        PixelSizeWarning
            If an image file's pixel size, used for the results, is
            doubtful.
        UnreadableRecordWarning
            If a record does not read; it counts as missing.

        Notes
        -----
        Each image's warnings are caught and kept for the export
        record; after the loop each class is raised once, naming the
        images with their messages. One log line per image. See
        *Results* (``docs/results.md``), section *Units and keys*.
        """
        sheet, config = self._read(check=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results, summaries = run_analyze(
                self._folder, sheet, config, progress=progress
            )
        for cls, message in summaries:
            warnings.warn(message, cls, stacklevel=2)
        reemit(caught, stacklevel=3)
        return results

    # -----------------------------------------------------------------
    # Files
    # -----------------------------------------------------------------

    def _read(self, check: bool = False) -> tuple[Sheet, ConfigFile]:
        sheet = validate(read_grid(self.sheet_path))
        path = self._folder / CONFIG_NAME
        if not path.exists():
            raise FileNotFoundError(
                f"{self._folder} has no {CONFIG_NAME}; Collection.create "
                "writes it"
            )
        config = ConfigFile.read(path)
        if check:
            config.check(sheet.names)
        return sheet, config

    def _write_config(self, sheet: Sheet, config: ConfigFile) -> None:
        config.check(sheet.names)
        text = config.text()
        replace_file(config.path, lambda tmp: tmp.write_bytes(text.encode()))


def _numbered(rows: list[list[object]]) -> list[tuple[int, list[object]]]:
    """Return the rows with the numbers Excel shows (header = row 1)."""
    return [(i + 2, row) for i, row in enumerate(rows)]


def _same_file(path: Path) -> str:
    return os.path.normcase(str(path))


def _include_cell(old: object, value: bool) -> object:
    """Return ``value`` for an ``include`` cell; 1/0 if it holds 1/0."""
    if isinstance(old, int | float) and not isinstance(old, bool):
        return int(value)
    if isinstance(old, str) and old.strip() in ("0", "1"):
        return "1" if value else "0"
    return value


_BAD_MASK = (
    "images must be a True / False Series on the rows of status() or "
    "read_sheet(), e.g. col.status().particle_set == 'hct'"
)


def _names(sheet: Sheet, images: Images) -> list[str]:
    """Return the images ``images`` selects, in sheet order, each once.

    A bool Series is a mask on the sheet's rows (the index of
    ``status()`` and ``read_sheet()``); any other value lists names.

    Raises
    ------
    TableError
        If a mask is not on the sheet's rows, has missing values, or
        a Series holds neither bools nor names.
    ParamsError
        If a name is not in the sheet.
    """
    if isinstance(images, pd.Series):
        if pd.api.types.is_bool_dtype(images):
            rows = pd.RangeIndex(len(sheet.entries))
            if not images.index.equals(rows) or images.isna().any():
                raise TableError(_BAD_MASK)
            return [e.image for e, keep in zip(sheet.entries, images) if keep]
        if not all(isinstance(v, str) for v in images):
            raise TableError(_BAD_MASK)
    wanted = [images] if isinstance(images, str) else list(images)
    unknown = [n for n in dict.fromkeys(wanted) if n not in sheet.names]
    if unknown:
        listed = ", ".join(repr(n) for n in unknown)
        raise ParamsError(f"no image {listed} in {sheet.path.name}")
    return [n for n in sheet.names if n in set(wanted)]


def _known(sheet: Sheet, image: str) -> None:
    if image not in sheet.names:
        raise ParamsError(f"no image {image!r} in {sheet.path.name}")


def _check_clashes(
    existing: list[tuple[str, str]], new: list[tuple[str, str]]
) -> None:
    """Raise SheetError when names clash ignoring case.

    ``existing`` and ``new`` hold ``(name, file)``; a new name may
    clash with the sheet or with an earlier new one.
    """
    first: dict[str, tuple[str, str]] = {}
    for name, file in existing:
        first.setdefault(_fold(name), (name, file))
    clashes = []
    for name, file in new:
        key = _fold(name)
        if key in first:
            other_name, other_file = first[key]
            clashes.append(
                f"{other_file} and {file} both give the image name "
                f"{other_name!r}"
            )
        else:
            first[key] = (name, file)
    if clashes:
        raise SheetError(
            "\n".join(clashes)
            + "; set names with columns_from_path=r'…(?P<image>…)…' or "
            "choose files with pattern="
        )


def _fold(name: str) -> str:
    return unicodedata.normalize("NFC", name).casefold()


def _typed_values(
    values: list[dict[str, str]], columns: list[str]
) -> list[dict[str, object]]:
    """Return the rule values typed for xlsx.

    A column becomes numbers when every value is numeric text without
    a leading zero (``"17"``, not ``"017"``).
    """
    numeric = {
        c
        for c in columns
        if all(_NUMERIC_TEXT.fullmatch(v[c]) for v in values if c in v)
    }
    return [
        {c: _number(text) if c in numeric else text for c, text in v.items()}
        for v in values
    ]


def _number(text: str) -> int | float:
    return float(text) if "." in text else int(text)
