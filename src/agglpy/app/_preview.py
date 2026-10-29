"""``Preview``: one image's detection in memory, for tuning params."""

import dataclasses
import html
import numbers
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import perf_counter
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.typing import ColorType
from numpy.typing import NDArray
from tqdm.auto import tqdm

from agglpy import _yaml, draw
from agglpy.app._config import ImageParams, plain
from agglpy.app._detect import warn_if_lossy
from agglpy.app._files import replace_file
from agglpy.app._records import now, versions
from agglpy.app._report import warning_text
from agglpy.app._sources import PathLike
from agglpy.app._work import export_folder
from agglpy.core.detect import hct_edges
from agglpy.core.region import Bounds
from agglpy.errors import OutputExistsError, ParamsError
from agglpy.export import _check_replaceable
from agglpy.io import read_image
from agglpy.logger import logger
from agglpy.params import (
    ClaheParams,
    HCTParams,
    HCTRange,
    PreprocessParams,
    RegionParams,
    RollingBallParams,
)
from agglpy.pipeline._detection import (
    Stage,
    cut_region,
    hct_pass,
    particles_from_passes,
    preprocess_stages,
)
from agglpy.tables import find_duplicates, validate_particles

if TYPE_CHECKING:
    from agglpy.app._collection import Collection

__all__ = ["Preview", "make"]

# Which image a view draws underneath
Underlay = Literal["raw", "preprocessed"] | None
# Colours of the passes, in pass order
_PASS_COLOURS = "tab10"
# Comparison colours: only in this preview, only in the other, in both
_NEW, _GONE, _SAME = "new", "gone", "same"
_CIRCLE_COLOURS = {_NEW: "lime", _GONE: "red", _SAME: "grey"}
_EDGE_COLOURS = {
    _NEW: (0.0, 1.0, 0.0),
    _GONE: (1.0, 0.0, 1.0),
    _SAME: (1.0, 1.0, 1.0),
}
# Panel labels as large as a legend's text: this share of the width
_LABEL_SIZE = 0.012
# Column heads of the card's table of passes
_CARD_HEADS = (
    "range",
    "d (px)",
    "canny",
    "accumulator",
    "min dist",
    "circles",
    "reused",
)

# The preprocessing steps that take a params object, by field
_NESTED: dict[str, type] = {
    "rolling_ball": RollingBallParams,
    "clahe": ClaheParams,
}


@dataclass(frozen=True)
class _Prepared:
    """The image read, cut and preprocessed (arrays read-only).

    The warnings are record texts: ``read_warnings`` of reading the
    file, ``stage_warnings`` of the preprocessing steps; ``seconds``
    the time the steps that ran took.
    """

    raw: NDArray[np.uint8]
    bounds: Bounds
    cut: NDArray[np.uint8]
    stages: tuple[Stage, ...]
    read_warnings: tuple[str, ...]
    stage_warnings: tuple[str, ...]
    seconds: float

    @property
    def final(self) -> NDArray[np.uint8]:
        """The image HCT runs on: the last stage, else the cut."""
        return self.stages[-1].image if self.stages else self.cut


@dataclass(frozen=True)
class _Pass:
    """One HCT pass: its range, its circles (region frame, read-only)."""

    hct: HCTRange
    circles: NDArray[np.float32]
    seconds: float
    reused: bool


@dataclass(frozen=True)
class _Card:
    """What the card shows, as plain texts.

    ``table`` is one row per pass then the total, empty with no
    ranges; ``colours`` the pass colour of each pass row.
    """

    header: list[str]
    table: list[list[str]]
    colours: list[str]
    lines: list[str]
    views: list[tuple[str, str]]
    calls: list[str]


@dataclass(frozen=True)
class _State:
    """Everything a preview holds; never changed after it is made."""

    collection: "Collection"
    name: str
    params: ImageParams
    # the JPEG line, checked once on the file
    source_warnings: tuple[str, ...]
    prepared: _Prepared
    preprocess_reused: bool
    passes: tuple[_Pass, ...]
    particles: pd.DataFrame
    warnings: tuple[str, ...]
    # the image's params in agglpy.yml when the first preview was made
    stored: ImageParams
    stored_at: str
    progress: bool
    seconds: float


class Preview:
    """One image's detection with given params, held in memory.

    Made by ``col[name].preview()`` and the change methods; not called
    directly. Nothing on a preview changes after it is made: a change
    returns a new preview and leaves this one as it was, and reruns
    only what it touches. A preview writes no file. HCT ranges are
    numbered from 1, as in ``p.change_hct(range=1, ...)``.
    """

    def __init__(self, state: _State) -> None:
        self._state = state

    @property
    def image_name(self) -> str:
        """The name of the previewed image, as in the sheet."""
        return self._state.name

    @property
    def params(self) -> ImageParams:
        """The params of this preview; ``hct`` is None with no ranges."""
        return self._state.params

    @property
    def particles(self) -> pd.DataFrame:
        """The circles found: a particle table ``id, x, y, r`` in px.

        In the raw image's frame, numbered range by range: the table
        ``col.detect()`` would store with these params. A new copy on
        each access; empty with no ranges.
        """
        return self._state.particles.copy()

    @property
    def warnings(self) -> tuple[str, ...]:
        """The warnings ``col.detect()`` would record, as their texts.

        ``"<Class>: <message>"``. A preview emits no Python warning.
        """
        return self._state.warnings

    def change_region(self, **fields: object) -> "Preview":
        """A new preview with changed ``RegionParams`` fields.

        E.g. ``bottom_fraction=0.1``. Reruns preprocessing and every
        pass (the file is not read again).

        Raises:
            ParamsError: For an unknown field or an invalid value.
        """
        region = _with_fields(
            RegionParams, self.params.region, fields, "region"
        )
        return self._changed(region=region)

    def change_preprocess(self, **fields: object) -> "Preview":
        """A new preview with changed ``PreprocessParams`` fields.

        ``rolling_ball`` and ``clahe`` take a mapping merged into the
        current value (``rolling_ball={"radius": 40}``), a params object
        that replaces it, or None to switch the step off; so does
        ``median_blur=None``. Reruns from the first changed step on,
        then every pass.

        Raises:
            ParamsError: For an unknown field, a missing required field
                of a step switched on, or an invalid value.
        """
        current = self.params.preprocess
        _check_names(PreprocessParams, fields, "preprocess")
        values: dict[str, Any] = {}
        for name, value in fields.items():
            kind = _NESTED.get(name)
            if kind is None or value is None or isinstance(value, kind):
                values[name] = value
            elif isinstance(value, Mapping):
                values[name] = _with_fields(
                    kind,
                    getattr(current, name),
                    {str(k): v for k, v in value.items()},
                    f"preprocess.{name}",
                )
            else:
                raise ParamsError(
                    f"preprocess.{name}: give a mapping, a {kind.__name__} "
                    f"or None, got {type(value).__name__}"
                )
        preprocess = dataclasses.replace(current, **values)
        return self._changed(preprocess=preprocess)

    def change_hct(self, *, range: int, **fields: object) -> "Preview":
        """A new preview with changed ``HCTRange`` fields of one range.

        E.g. ``p.change_hct(range=1, canny_threshold=180)``. Reruns that
        range's pass only.

        Raises:
            ParamsError: If range ``range`` does not exist, for an
                unknown field or an invalid value.
        """
        ranges = list(self._ranges())
        i = self._index(range)
        ranges[i] = _with_fields(HCTRange, ranges[i], fields, f"range {range}")
        return self._changed(hct=HCTParams(tuple(ranges)))

    def set_ranges(
        self, bounds: Sequence[tuple[int, int]], **shared: object
    ) -> "Preview":
        """A new preview with new ranges, one per ``(d_min, d_max)``.

        ``shared`` fields (``canny_threshold=180``, ...) apply to every
        range; the others take the ``HCTRange`` defaults. Reruns only
        the passes that differ from this preview's.

        Raises:
            ParamsError: For no pair, a pair that is not two values,
                ``d_min`` / ``d_max`` among ``shared``, an unknown field
                or an invalid value.
        """
        for key in ("d_min", "d_max"):
            if key in shared:
                raise ParamsError(
                    f"set_ranges: {key} comes from the (d_min, d_max) "
                    "pairs, not from the shared fields"
                )
        _check_names(HCTRange, shared, "set_ranges")
        pairs = list(bounds)
        if not pairs:
            raise ParamsError(
                "give at least one (d_min, d_max) pair; p.remove_range(k) "
                "removes one"
            )
        ranges = []
        for k, pair in enumerate(pairs, start=1):
            try:
                d_min, d_max = pair
            except (TypeError, ValueError):
                raise ParamsError(
                    f"set_ranges takes (d_min, d_max) pairs, got {pair!r}"
                ) from None
            ranges.append(
                _with_fields(
                    HCTRange,
                    None,
                    {"d_min": d_min, "d_max": d_max, **shared},
                    f"range {k}",
                )
            )
        return self._changed(hct=HCTParams(tuple(ranges)))

    def add_range(self, d_min: int, d_max: int, **fields: object) -> "Preview":
        """A new preview with one more range, after the others.

        Fields not given take the ``HCTRange`` defaults. Runs the new
        range's pass only.

        Raises:
            ParamsError: For an unknown field or an invalid value.
        """
        ranges = self._ranges()
        new = _with_fields(
            HCTRange,
            None,
            {"d_min": d_min, "d_max": d_max, **fields},
            f"range {len(ranges) + 1}",
        )
        return self._changed(hct=HCTParams((*ranges, new)))

    def remove_range(self, k: int) -> "Preview":
        """A new preview without range ``k``; the rest renumbered.

        Runs no pass. Removing the last range gives a preview with no
        ranges.

        Raises:
            ParamsError: If range ``k`` does not exist.
        """
        ranges = list(self._ranges())
        del ranges[self._index(k)]
        return self._changed(hct=HCTParams(tuple(ranges)) if ranges else None)

    def keep(self) -> None:
        """Write this preview's params into ``agglpy.yml``, for its image.

        The region, preprocessing and HCT ranges replace the image's
        entry (only what differs from the defaults is stored), whatever
        the file holds now. ``col.detect()`` then runs with them;
        ``col.status()`` shows ``params changed`` until it does.

        Raises:
            ParamsError: If the preview has no HCT ranges (keeping it
                would switch detection off), the image is no longer in
                the sheet, or the file has problems; nothing is written.
        """
        state = self._state
        name = state.name
        params = self.params
        if params.hct is None:
            raise ParamsError(
                "this preview has no HCT ranges; keep() would turn "
                f"detection off for {name}. Add ranges "
                "(p.set_ranges(...)), or turn it off on purpose: "
                f'col.set_params("{name}", hct={{"ranges": []}})'
            )
        collection = state.collection
        changes = _differences(collection.params(name), params)
        collection.set_params(
            name,
            region=params.region,
            preprocess=params.preprocess,
            hct=params.hct,
        )
        logger.info(
            "%s: params kept in agglpy.yml (%s)",
            name,
            "; ".join(changes) or "unchanged",
        )

    def export(
        self, folder: PathLike | None = None, *, overwrite: bool = False
    ) -> list[Path]:
        """Write every view as a PNG, and ``<image> preview.yml``.

        The views in their default form: ``<image> image.png``,
        ``<image> stage <step>.png`` per step, ``<image> edges <k>.png``
        and ``<image> circles <k>.png`` per range, ``<image>
        circles.png``. ``preview.yml`` holds the params in the
        ``agglpy.yml`` layout, the numbers per range, the warnings, the
        date and the versions.

        Args:
            folder: The folder to write into; created with its parents.
                None gives ``exports/tuning`` in the collection, where a
                repeated export needs ``overwrite=True``. A relative
                path is taken from the current working directory.
            overwrite: Replace files of this image already there.

        Returns:
            The files written, absolute, in the order above.

        Raises:
            ParamsError: If ``folder`` lies in the collection's work
                folder ``agglpy/``.
            OutputExistsError: If one of the files is there and
                ``overwrite`` is False; nothing is written.
            OutputLockedError: If a file to replace is open in another
                program; nothing is changed.
        """
        state = self._state
        folder = export_folder(state.collection.folder, folder, "tuning")
        views = self._export_views()
        names = [*views, "preview.yml"]
        paths = [folder / f"{state.name} {name}" for name in names]
        existing = [p for p in paths if p.exists()]
        if existing and not overwrite:
            listed = ", ".join(p.name for p in existing)
            raise OutputExistsError(
                f"{folder} already holds {listed}; pass overwrite=True to "
                "replace them"
            )
        _check_replaceable(existing)
        folder.mkdir(parents=True, exist_ok=True)
        text = _yaml.dumps(self._export_record())
        made: list[Path] = []
        try:
            for view, path in zip(views.values(), paths, strict=False):
                new = not path.exists()
                # one figure at a time: a large image's figures are big
                fig = view()
                replace_file(
                    path, partial(fig.savefig, format="png", dpi="figure")
                )
                if new:
                    made.append(path)
            new = not paths[-1].exists()
            replace_file(
                paths[-1], lambda tmp: tmp.write_bytes(text.encode("utf-8"))
            )
            if new:
                made.append(paths[-1])
        except BaseException:
            # the new files only: a replaced one can't be put back
            for path in made:
                path.unlink(missing_ok=True)
            raise
        logger.info("%s: preview exported to %s", state.name, folder)
        return paths

    def _export_views(self) -> dict[str, Callable[[], Figure]]:
        """File name ending -> the view that draws it, in export order."""
        views: dict[str, Callable[[], Figure]] = {"image.png": self.image}
        for stage in self._state.prepared.stages:
            views[f"stage {stage.name}.png"] = partial(self.stage, stage.name)
        ks = range(1, len(self._ranges()) + 1)
        for k in ks:
            views[f"edges {k}.png"] = partial(self.edges, k)
        for k in ks:
            views[f"circles {k}.png"] = partial(self.circles, k)
        if ks:
            views["circles.png"] = self.circles
        return views

    def _export_record(self) -> dict[str, Any]:
        """The content of ``preview.yml``."""
        state = self._state
        agglpy_version, dependencies = versions()
        params = plain(self.params, "params")
        # as agglpy.yml writes no ranges
        params["hct"] = params["hct"] or {"ranges": []}
        ranges = [
            {
                "range": k,
                **plain(found.hct, f"ranges[{k}]"),
                "circles": len(found.circles),
                "seconds": round(found.seconds, 3),
            }
            for k, found in enumerate(state.passes, start=1)
        ]
        return {
            "image": state.name,
            "created": now(),
            "agglpy": agglpy_version,
            "dependencies": dependencies,
            "params": params,
            "ranges": ranges,
            "circles": len(state.particles),
            "info": self._info_lines(),
            "warnings": list(state.warnings),
        }

    # -----------------------------------------------------------------
    # Views: one new full-size figure per call
    # -----------------------------------------------------------------

    def image(self) -> Figure:
        """The raw image, the analysed region outlined in yellow.

        One figure pixel per image pixel; the outline lies on the
        region's outermost pixels and is drawn only when the region is
        smaller than the image.
        """
        held = self._state.prepared
        fig, ax = draw.figure(held.raw.shape, legend=False)
        draw.image(ax, held.raw)
        height, width = held.raw.shape
        b = held.bounds
        if (b.x_min, b.y_min, b.x_max, b.y_max) != (
            -0.5,
            -0.5,
            width - 0.5,
            height - 0.5,
        ):
            ax.add_patch(
                Rectangle(
                    (b.x_min + 0.5, b.y_min + 0.5),
                    b.x_max - b.x_min - 1,
                    b.y_max - b.y_min - 1,
                    fill=False,
                    edgecolor="yellow",
                    linewidth=1,
                    # on the pixels themselves, not snapped 1 px inward
                    snap=False,
                )
            )
        return fig

    def stage(self, name: str) -> Figure:
        """The region after a preprocessing step, at its place.

        ``name`` is a step that ran: ``"median"``, ``"rolling ball"``,
        ``"clahe"``. The rest of the image is black.

        Raises:
            ParamsError: If no step of that name ran.
        """
        stages = self._state.prepared.stages
        for stage in stages:
            if stage.name == name:
                fig, ax = self._figure(legend=False, background="black")
                draw.image(ax, stage.image, origin=self._origin())
                return fig
        ran = ", ".join(s.name for s in stages) or "no preprocessing"
        raise ParamsError(f"no stage {name!r}: {self.image_name} ran {ran}")

    def edges(
        self,
        k: int,
        *,
        compare: "Preview | None" = None,
        side_by_side: bool = False,
        image: Underlay = None,
        image_alpha: float = 1.0,
    ) -> Figure:
        """The edges HoughCircles sees in range ``k``.

        ``core.detect.hct_edges`` on the preprocessed region: white on
        black, or yellow over ``image``. With ``compare``, the edges of
        both previews' range ``k``: green only in this one (new),
        magenta only in ``compare`` (gone), white in both.

        Args:
            k: The range, numbered from 1.
            compare: Another preview of this image.
            side_by_side: Two panels zoomed together: ``compare``'s
                edges left, these right; without ``compare`` the image
                left (raw when ``image`` is None), the edges on black
                right.
            image: The image underneath: ``"raw"``, ``"preprocessed"``
                (the region) or None.
            image_alpha: Its opacity, 0-1.

        Raises:
            ParamsError: If range ``k`` does not exist (in either
                preview), ``compare`` is not a preview of this image or
                an option is invalid.
        """
        i = self._index(k)
        mine = self._edge_mask(i)
        if compare is None:
            if side_by_side:
                fig, left, right = self._pair(False, "black")
                self._underlay(left, image or "raw", image_alpha)
                _paint(right, {"": (mine, _EDGE_COLOURS[_SAME])})
                _label(left, image or "raw")
                _label(right, "edges")
                return fig
            fig, ax = self._figure(legend=False, background="black")
            self._underlay(ax, image, image_alpha)
            colour = (1.0, 1.0, 1.0) if image is None else (1.0, 1.0, 0.0)
            _paint(ax, {"": (mine, colour)})
            return fig
        other = self._compared(compare, k)
        theirs = other._edge_mask(i)
        masks = {
            _NEW: mine & ~theirs,
            _GONE: theirs & ~mine,
            _SAME: mine & theirs,
        }
        labels = {
            key: f"{key} ({int(mask.sum())} px)" for key, mask in masks.items()
        }
        legend = {labels[key]: _EDGE_COLOURS[key] for key in masks}
        layers = {
            key: (mask, _EDGE_COLOURS[key]) for key, mask in masks.items()
        }
        if side_by_side:
            fig, left, right = self._pair(True, "black")
            other._underlay(left, image, image_alpha)
            _paint(left, {key: layers[key] for key in (_GONE, _SAME)})
            self._underlay(right, image, image_alpha)
            _paint(right, {key: layers[key] for key in (_NEW, _SAME)})
            self._pair_labels(left, right, other)
            draw.legend(right, legend)
            return fig
        fig, ax = self._figure(legend=True, background="black")
        self._underlay(ax, image, image_alpha)
        _paint(ax, layers)
        draw.legend(ax, legend)
        return fig

    def circles(
        self,
        k: int | None = None,
        *,
        compare: "Preview | None" = None,
        side_by_side: bool = False,
        image: Underlay = "raw",
        image_alpha: float = 1.0,
        background: ColorType | None = "black",
        alpha: float = 0.0,
        linewidth: float = 1.0,
    ) -> Figure:
        """The circles of range ``k``, or of every range by range.

        Outlines over the raw image by default, so each circle is
        judged against what the eye sees; one colour per range, the
        legend with the counts. With ``compare``: green the circles
        only in this preview (new), red those only in ``compare``
        (gone), grey those in both, matched with the tolerances of
        ``tables.find_duplicates`` (a circle that moved is gone + new).

        Args:
            k: The range, numbered from 1; None for every range.
            compare: Another preview of this image.
            side_by_side: Two panels zoomed together: ``compare``'s
                circles left, these right; without ``compare`` the
                image left (raw when ``image`` is None), the circles on
                ``background`` right.
            image: The image underneath: ``"raw"``, ``"preprocessed"``
                (the region) or None (the background).
            image_alpha: Its opacity, 0-1.
            background: The figure's colour; None for transparent.
            alpha: Opacity of the fill, 0 (outlines only) to 1.
            linewidth: Width of the outlines in px.

        Raises:
            ParamsError: If range ``k`` does not exist (in either
                preview), ``compare`` is not a preview of this image or
                an option is invalid.
        """
        if k is None:
            self._index(1)  # the message of no ranges
            chosen = list(range(len(self._state.passes)))
        else:
            chosen = [self._index(k)]
        style = {"alpha": alpha, "linewidth": linewidth}
        if compare is None:
            table, colours = self._by_range(chosen)
            if side_by_side:
                fig, left, right = self._pair(True, background)
                self._underlay(left, image or "raw", image_alpha)
                _circles(right, table, colours, style)
                _label(left, image or "raw")
                _label(right, "circles")
                draw.legend(right, colours)
                return fig
            fig, ax = self._figure(legend=True, background=background)
            self._underlay(ax, image, image_alpha)
            _circles(ax, table, colours, style)
            draw.legend(ax, colours)
            return fig
        other = self._compared(compare, k)
        there = list(range(len(other._state.passes))) if k is None else chosen
        mine = self._by_range(chosen)[0]
        theirs = other._by_range(there)[0]
        same_here, same_there = _matched(mine, theirs)
        labels = {
            _NEW: f"{_NEW} ({int((~same_here).sum())})",
            _GONE: f"{_GONE} ({int((~same_there).sum())})",
            _SAME: f"{_SAME} ({int(same_here.sum())})",
        }
        colours = {labels[key]: _CIRCLE_COLOURS[key] for key in labels}
        mine = mine.assign(
            range=np.where(same_here, labels[_SAME], labels[_NEW])
        )
        theirs = theirs.assign(
            range=np.where(same_there, labels[_SAME], labels[_GONE])
        )
        if side_by_side:
            fig, left, right = self._pair(True, background)
            other._underlay(left, image, image_alpha)
            _circles(left, theirs, colours, style)
            self._underlay(right, image, image_alpha)
            _circles(right, mine, colours, style)
            self._pair_labels(left, right, other)
            draw.legend(right, colours)
            return fig
        fig, ax = self._figure(legend=True, background=background)
        self._underlay(ax, image, image_alpha)
        # a circle in both is drawn once, from this preview
        overlay = pd.concat([mine, theirs[~same_there]], ignore_index=True)
        _circles(ax, overlay, colours, style)
        draw.legend(ax, colours)
        return fig

    def _by_range(
        self, chosen: Sequence[int]
    ) -> tuple[pd.DataFrame, dict[str, ColorType]]:
        """The circles of passes ``chosen`` (0-based), labelled in
        column ``range``, and the colour of each label."""
        passes = self._state.passes
        labels = {
            i: f"range {i + 1} ({len(passes[i].circles)})" for i in chosen
        }
        colours: dict[str, ColorType] = {
            labels[i]: _pass_colour(i) for i in chosen
        }
        counts = [len(found.circles) for found in passes]
        table = self._state.particles.assign(
            range=np.repeat(np.arange(len(passes)), counts)
        )
        table = table[table["range"].isin(chosen)]
        return table.assign(range=table["range"].map(labels)), colours

    def _edge_mask(self, i: int) -> NDArray[np.bool_]:
        """Pass ``i``'s (0-based) edge pixels in the raw image's frame."""
        held = self._state.prepared
        x, y = (round(v) for v in self._origin())
        edges = hct_edges(held.final, self._ranges()[i]) > 0
        mask = np.zeros(held.raw.shape, bool)
        mask[y : y + edges.shape[0], x : x + edges.shape[1]] = edges
        return mask

    def _compared(self, compare: object, k: int | None) -> "Preview":
        """``compare`` checked: a preview of this image with range ``k``."""
        if not isinstance(compare, Preview):
            raise ParamsError(
                "compare must be another Preview, got "
                f"{type(compare).__name__}"
            )
        name = self._state.name
        other = compare._state
        if (
            other.name != name
            or other.prepared.raw.shape != self._state.prepared.raw.shape
        ):
            raise ParamsError(
                f"compare must be a preview of {name}, got one of "
                f"{other.name}: the pictures would not line up"
            )
        here, there = len(self._ranges()), len(compare._ranges())
        if k is not None and k > there:
            noun = "range" if here == 1 else "ranges"
            raise ParamsError(
                f"range {k} does not exist in compare: {name} has "
                f"{here} {noun}, {there} in compare"
            )
        return compare

    def _pair(
        self, legend: bool, background: ColorType | None
    ) -> tuple[Figure, Axes, Axes]:
        shape = self._state.prepared.raw.shape
        return draw.figure_pair(shape, legend=legend, background=background)

    def _pair_labels(self, left: Axes, right: Axes, other: "Preview") -> None:
        _label(left, "before")
        changes = _differences(other.params, self.params)
        _label(right, "; ".join(changes) or "same params")

    def _figure(
        self, *, legend: bool, background: ColorType | None
    ) -> tuple[Figure, Axes]:
        shape = self._state.prepared.raw.shape
        return draw.figure(shape, legend=legend, background=background)

    def _origin(self) -> tuple[float, float]:
        """Where the region's top-left pixel centre lies in the image."""
        b = self._state.prepared.bounds
        return b.x_min + 0.5, b.y_min + 0.5

    def _underlay(self, ax: Axes, image: object, alpha: float) -> None:
        held = self._state.prepared
        if image == "raw":
            draw.image(ax, held.raw, alpha=alpha)
        elif image == "preprocessed":
            draw.image(ax, held.final, alpha=alpha, origin=self._origin())
        elif image is not None:
            raise ParamsError(
                f"image must be 'raw', 'preprocessed' or None, got {image!r}"
            )

    def __repr__(self) -> str:
        """The card as text: the same lines as in a notebook."""
        card = self._card()
        parts = ["\n".join(card.header)]
        if card.table:
            parts.append(_text_table(card.table))
        if card.lines:
            parts.append("\n".join(card.lines))
        width = max(len(call) for call, _ in card.views)
        views = [f"  {call:<{width}}  {what}" for call, what in card.views]
        parts.append("\n".join(["Views", *views]))
        calls = [f"  {call}" for call in card.calls]
        parts.append("\n".join(["Change and keep (current values)", *calls]))
        return "\n\n".join(parts)

    def _repr_html_(self) -> str:
        """The card for a notebook: the lines and tables, no figures."""
        card = self._card()
        parts = ["<div>" + "<br>".join(map(_escape, card.header)) + "</div>"]
        if card.table:
            heads, *rows = card.table
            cells = ["<tr>" + "".join(f"<th>{h}</th>" for h in heads)]
            for i, row in enumerate(rows):
                first = _escape(row[0])
                if i < len(card.colours):
                    swatch = f'<span style="color:{card.colours[i]}">■</span>'
                    first = f"{swatch} {first}"
                rest = "".join(f"<td>{_escape(v)}</td>" for v in row[1:])
                cells.append(f"<tr><td>{first}</td>{rest}</tr>")
            parts.append("<table>" + "".join(cells) + "</table>")
        parts += [f"<div>{_escape(line)}</div>" for line in card.lines]
        views = "".join(
            f"<tr><td><code>{_escape(call)}</code></td>"
            f"<td>{_escape(what)}</td></tr>"
            for call, what in card.views
        )
        parts.append(f"<p><b>Views</b></p><table>{views}</table>")
        calls = "".join(
            f"<tr><td><code>{_escape(call)}</code></td></tr>"
            for call in card.calls
        )
        parts.append(
            f"<p><b>Change and keep (current values)</b></p><table>{calls}"
            "</table>"
        )
        return "".join(parts)

    def _card(self) -> _Card:
        state = self._state
        held = state.prepared
        height, width = held.raw.shape
        first = [f"Preview · {state.name}", f"{width} x {height} px"]
        if state.source_warnings:
            first.append("JPEG-compressed")
        first.append(f"region: {_region_text(self.params.region)}")
        steps = _preprocess_text(self.params.preprocess)
        timing = (
            "reused from the previous preview, 0 s"
            if state.preprocess_reused
            else f"{held.seconds:.1f} s"
        )
        second = f"preprocessing: {steps} · {timing}"
        if state.passes:
            ran = sum(p.seconds for p in state.passes if not p.reused)
            second += f" · HCT {ran:.1f} s"
        changes = _differences(state.stored, self.params)
        # the time of day the stored params were read
        at = state.stored_at[11:19]
        third = (
            f"vs agglpy.yml at {at}: " + "; ".join(changes)
            if changes
            else f"same as agglpy.yml at {at}"
        )
        header = [" · ".join(first), second, third]
        ranges = self._ranges()
        table: list[list[str]] = []
        if ranges:
            table.append(list(_CARD_HEADS))
            for k, (hct, found) in enumerate(
                zip(ranges, state.passes, strict=True), start=1
            ):
                table.append(
                    [
                        str(k),
                        f"{hct.d_min}-{hct.d_max}",
                        f"{hct.canny_threshold:g}",
                        f"{hct.accumulator_threshold:g}",
                        f"{hct.min_dist_ratio:g}",
                        str(len(found.circles)),
                        "yes" if found.reused else "no",
                    ]
                )
            table.append(
                ["total", "", "", "", "", str(len(state.particles)), ""]
            )
        else:
            header.append(
                "no HCT ranges set: p.set_ranges([(d_min, d_max), …])"
            )
        return _Card(
            header=header,
            table=table,
            colours=[to_hex(_pass_colour(i)) for i in range(len(ranges))],
            lines=self._info_lines() + list(state.warnings),
            views=self._view_calls(),
            calls=self._change_calls(),
        )

    def _info_lines(self) -> list[str]:
        """Per pass and pair of passes, what a person tuning would ask:
        a pass with no circles, circles two passes both found."""
        state = self._state
        final = state.prepared.final
        lines = [
            f"range {k}: no circles (pixel values {final.min()}-{final.max()})"
            for k, found in enumerate(state.passes, start=1)
            if len(found.circles) == 0
        ]
        counts = [len(found.circles) for found in state.passes]
        pass_of = dict(
            zip(
                state.particles["id"],
                np.repeat(np.arange(len(counts)), counts),
                strict=True,
            )
        )
        pairs: dict[tuple[int, int], int] = {}
        for a, b in find_duplicates(state.particles):
            i, j = sorted((int(pass_of[a]), int(pass_of[b])))
            if i != j:
                pairs[i, j] = pairs.get((i, j), 0) + 1
        for (i, j), n in sorted(pairs.items()):
            noun = "pair" if n == 1 else "pairs"
            lines.append(
                f"{n} {noun} of near-identical circles between ranges "
                f"{i + 1} and {j + 1}"
            )
        return lines

    def _view_calls(self) -> list[tuple[str, str]]:
        stages = self._state.prepared.stages
        count = len(self._ranges())
        views = [("p.image()", "the raw image, the region outlined")]
        if stages:
            views.append(
                (
                    " · ".join(f'p.stage("{s.name}")' for s in stages),
                    "the region after a preprocessing step",
                )
            )
        if count:
            ks = range(1, count + 1)
            views += [
                (
                    " · ".join(f"p.edges({k})" for k in ks),
                    "the edges HoughCircles sees in that range",
                ),
                (
                    " · ".join(
                        [*(f"p.circles({k})" for k in ks), "p.circles()"]
                    ),
                    "the circles of one range / of every range",
                ),
                (
                    "… compare=other, side_by_side=True",
                    "on edges and circles: against another preview",
                ),
            ]
        views.append(
            ("p.particles · p.warnings", "the circle table (px) · warnings")
        )
        return views

    def _change_calls(self) -> list[str]:
        ranges = self._ranges()
        if not ranges:
            return ["p.export()"]
        calls = [
            f"p.change_hct(range={k}, "
            f"canny_threshold={hct.canny_threshold:g}, "
            f"accumulator_threshold={hct.accumulator_threshold:g})"
            for k, hct in enumerate(ranges, start=1)
        ]
        pairs = ", ".join(f"({r.d_min}, {r.d_max})" for r in ranges)
        calls += [f"p.set_ranges([{pairs}])", "p.keep()  ·  p.export()"]
        return calls

    def _ranges(self) -> tuple[HCTRange, ...]:
        hct = self.params.hct
        return hct.ranges if hct is not None else ()

    def _index(self, k: object) -> int:
        """The list index of range ``k`` (numbered from 1)."""
        if isinstance(k, bool) or not isinstance(k, numbers.Integral):
            raise ParamsError(
                f"range must be an int (numbered from 1), got {k!r}"
            )
        count = len(self._ranges())
        name = self._state.name
        if count == 0:
            raise ParamsError(
                f"{name} has no HCT ranges: p.set_ranges([(d_min, d_max), …])"
            )
        if not 1 <= int(k) <= count:
            noun = "range" if count == 1 else "ranges"
            raise ParamsError(
                f"range {k} does not exist: {name} has {count} {noun}"
            )
        return int(k) - 1

    def _changed(self, **sections: Any) -> "Preview":
        params = dataclasses.replace(self.params, **sections)
        return _derive(self._state, params)


def make(
    collection: "Collection",
    name: str,
    params: ImageParams | None,
    progress: bool,
) -> Preview:
    """The preview of ``img.preview()``: read, preprocess, detect."""
    stored = collection.params(name)
    if params is None:
        params = stored
    elif not isinstance(params, ImageParams):
        raise ParamsError(
            "params must be ImageParams (p.params or col[name].params), "
            f"got {type(params).__name__}"
        )
    image = collection[name]
    entry = image._entry()
    path = image._path()
    stored_at = now()
    ranges = _ranges_of(params)
    steps = 1 + len(_configured(params.preprocess)) + (1 if ranges else 0)
    start = perf_counter()
    with _Steps(name, steps, progress) as report:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_if_lossy(entry.file, path)
        source = tuple(warning_text(w) for w in caught)
        prepared = _prepare(path, params.region, params.preprocess, report)
        passes = _run_passes(prepared.final, ranges, (), report)
    return _new(
        collection=collection,
        name=name,
        params=params,
        source_warnings=source,
        prepared=prepared,
        preprocess_reused=False,
        passes=passes,
        stored=stored,
        stored_at=stored_at,
        progress=progress,
        start=start,
    )


def _derive(parent: _State, params: ImageParams) -> Preview:
    """A preview of the parent's image with new params.

    Stages are taken from the parent while the region is unchanged and
    the steps match from the first one; a pass is taken when the image
    it runs on is the parent's and the parent has an equal range.
    """
    held = parent.prepared
    configured = _configured(params.preprocess)
    same_region = params.region == parent.params.region
    kept = _kept_stages(held.stages, configured) if same_region else 0
    every_stage = same_region and kept == len(configured) == len(held.stages)
    reuse = parent.passes if every_stage else ()
    ranges = _ranges_of(params)
    runs = sum(all(p.hct != r for p in reuse) for r in ranges)
    steps = len(configured) - kept + (1 if runs else 0)
    start = perf_counter()
    with _Steps(parent.name, steps, parent.progress) as report:
        prepared = (
            held
            if every_stage
            else _restage(held, params, kept, same_region, report)
        )
        passes = _run_passes(prepared.final, ranges, reuse, report)
    return _new(
        collection=parent.collection,
        name=parent.name,
        params=params,
        source_warnings=parent.source_warnings,
        prepared=prepared,
        preprocess_reused=every_stage,
        passes=passes,
        stored=parent.stored,
        stored_at=parent.stored_at,
        progress=parent.progress,
        start=start,
    )


def _new(
    *,
    collection: "Collection",
    name: str,
    params: ImageParams,
    source_warnings: tuple[str, ...],
    prepared: _Prepared,
    preprocess_reused: bool,
    passes: tuple[_Pass, ...],
    stored: ImageParams,
    stored_at: str,
    progress: bool,
    start: float,
) -> Preview:
    """Number the circles, gather the warnings, log the new preview."""
    particles, assembly = _assemble(prepared, passes)
    preview = Preview(
        _State(
            collection=collection,
            name=name,
            params=params,
            source_warnings=source_warnings,
            prepared=prepared,
            preprocess_reused=preprocess_reused,
            passes=passes,
            particles=particles,
            # detect()'s order: the file, reading, preprocessing, HCT
            warnings=(
                source_warnings
                + prepared.read_warnings
                + prepared.stage_warnings
                + assembly
            ),
            stored=stored,
            stored_at=stored_at,
            progress=progress,
            seconds=perf_counter() - start,
        )
    )
    _log(preview)
    return preview


def _prepare(
    path: Path,
    region: RegionParams,
    preprocess: PreprocessParams,
    on_step: Callable[[str], None],
) -> _Prepared:
    """Read, cut and preprocess the image: the only way a fresh preview
    does, so a cache can wrap this one function later."""
    start = perf_counter()
    with warnings.catch_warnings(record=True) as read:
        warnings.simplefilter("always")
        on_step("reading")
        raw = read_image(path)
        raw.setflags(write=False)
        bounds, cut = cut_region(raw, region)
    with warnings.catch_warnings(record=True) as preprocessed:
        warnings.simplefilter("always")
        stages = preprocess_stages(cut, preprocess, on_step=on_step)
    return _Prepared(
        raw=raw,
        bounds=bounds,
        cut=cut,
        stages=stages,
        read_warnings=tuple(warning_text(w) for w in read),
        stage_warnings=tuple(warning_text(w) for w in preprocessed),
        seconds=perf_counter() - start,
    )


def _restage(
    held: _Prepared,
    params: ImageParams,
    kept: int,
    same_region: bool,
    on_step: Callable[[str], None],
) -> _Prepared:
    """Preprocess again from stage ``kept`` on; the raw image is kept."""
    start = perf_counter()
    if same_region:
        bounds, cut = held.bounds, held.cut
    else:
        bounds, cut = cut_region(held.raw, params.region)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stages = preprocess_stages(
            cut,
            params.preprocess,
            reuse=held.stages[:kept],
            on_step=on_step,
        )
    return _Prepared(
        raw=held.raw,
        bounds=bounds,
        cut=cut,
        stages=stages,
        read_warnings=held.read_warnings,
        # a reused stage's warnings are not carried over
        stage_warnings=tuple(warning_text(w) for w in caught),
        seconds=perf_counter() - start,
    )


def _run_passes(
    image: NDArray[np.uint8],
    ranges: Sequence[HCTRange],
    reuse: Sequence[_Pass],
    on_step: Callable[[str], None],
) -> tuple[_Pass, ...]:
    """One pass per range; a range equal to one in ``reuse`` is taken
    from there (``reuse`` must have run on the same ``image``)."""
    passes: list[_Pass] = []
    reported = False
    for hct in ranges:
        earlier = next((p for p in reuse if p.hct == hct), None)
        if earlier is not None:
            passes.append(_Pass(hct, earlier.circles, earlier.seconds, True))
            continue
        if not reported:
            on_step("hct")
            reported = True
        start = perf_counter()
        circles = hct_pass(image, hct)
        circles.setflags(write=False)
        passes.append(_Pass(hct, circles, perf_counter() - start, False))
    return tuple(passes)


def _assemble(
    prepared: _Prepared, passes: Sequence[_Pass]
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """The particle table of the passes and the warnings it raised."""
    if not passes:
        return _no_particles(), ()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        table = particles_from_passes(
            prepared.final, [p.circles for p in passes], prepared.bounds
        )
    return table, tuple(warning_text(w) for w in caught)


def _no_particles() -> pd.DataFrame:
    return validate_particles(
        pd.DataFrame(
            {
                "id": np.empty(0, np.int64),
                "x": np.empty(0),
                "y": np.empty(0),
                "r": np.empty(0),
            }
        )
    )


def _pass_colour(i: int) -> tuple[float, float, float, float]:
    """The colour of pass ``i`` (0-based) in the views."""
    colour: tuple[float, float, float, float] = mpl.colormaps[_PASS_COLOURS](
        i % 10
    )
    return colour


def _paint(
    ax: Axes,
    layers: Mapping[str, tuple[NDArray[np.bool_], tuple[float, ...]]],
) -> None:
    """Pixel masks (raw image frame) in their colours, as one image;
    transparent elsewhere."""
    first = next(iter(layers.values()))[0]
    rgba = np.zeros((*first.shape, 4), np.uint8)
    for mask, colour in layers.values():
        rgba[mask] = [round(255 * c) for c in colour] + [255]
    draw.image(ax, rgba)


def _circles(
    ax: Axes,
    table: pd.DataFrame,
    colours: Mapping[str, ColorType],
    style: Mapping[str, float],
) -> None:
    """Outlines coloured by the label in column ``range``, no legend."""
    draw.particles(
        ax,
        table,
        color_by="range",
        colors=colours,
        others=None,
        alpha=style["alpha"],
        linewidth=style["linewidth"],
        legend=False,
    )


def _matched(
    mine: pd.DataFrame, theirs: pd.DataFrame
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Which circles of each table have a near-identical one in the
    other, by ``find_duplicates``' tolerances."""
    # one table for find_duplicates: the second's ids after the first's
    offset = len(mine)
    both = pd.concat(
        [
            mine[["x", "y", "r"]].assign(id=np.arange(offset)),
            theirs[["x", "y", "r"]].assign(id=offset + np.arange(len(theirs))),
        ],
        ignore_index=True,
    )
    here = np.zeros(len(mine), bool)
    there = np.zeros(len(theirs), bool)
    for a, b in find_duplicates(both):
        if a < offset <= b:
            here[a] = True
            there[b - offset] = True
    return here, there


def _label(ax: Axes, text: str) -> None:
    """A panel's label in its top-left corner, white on dark grey."""
    fig = ax.get_figure()
    assert isinstance(fig, Figure)
    width = ax.get_position().width * fig.get_figwidth() * fig.dpi
    ax.text(
        0.01,
        0.99,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=_LABEL_SIZE * width * 72 / fig.dpi,
        bbox={"facecolor": "black", "alpha": 0.5, "edgecolor": "none"},
    )


def _escape(text: str) -> str:
    return html.escape(text, quote=False)


def _text_table(rows: Sequence[Sequence[str]]) -> str:
    """Rows as aligned text: the first column left, the rest right."""
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = []
    for row in rows:
        cells = [row[0].ljust(widths[0])]
        cells += [v.rjust(w) for v, w in zip(row[1:], widths[1:], strict=True)]
        lines.append("  ".join(cells).rstrip())
    return "\n".join(lines)


def _region_text(region: RegionParams) -> str:
    if region.bottom_fraction == 0:
        return "the whole image"
    return f"bottom {region.bottom_fraction * 100:g} % cut"


def _preprocess_text(preprocess: PreprocessParams) -> str:
    """The steps switched on, in run order: ``median 3 → CLAHE``."""
    steps = []
    if preprocess.median_blur is not None:
        steps.append(f"median {preprocess.median_blur}")
    if preprocess.rolling_ball is not None:
        steps.append(f"rolling ball r {preprocess.rolling_ball.radius:g}")
    if preprocess.clahe is not None:
        steps.append("CLAHE")
    return " → ".join(steps) or "none"


def _differences(old: ImageParams, new: ImageParams) -> list[str]:
    """What changed from ``old`` to ``new``, one text per value.

    ``preprocess.rolling_ball.radius 50 → 40``; ranges numbered from 1
    (``ranges[2].canny_threshold 220 → 180``), a different number of
    ranges as one item ``ranges: 2 → 3``.
    """
    before, after = plain(old, "params"), plain(new, "params")
    ranges = [
        (before.pop("hct") or {}).get("ranges", []),
        (after.pop("hct") or {}).get("ranges", []),
    ]
    changes = _changes(before, after, "")
    if len(ranges[0]) != len(ranges[1]):
        changes.append(f"ranges: {len(ranges[0])} → {len(ranges[1])}")
        return changes
    for k, (a, b) in enumerate(zip(*ranges, strict=True), start=1):
        changes += _changes(a, b, f"ranges[{k}].")
    return changes


def _changes(old: Any, new: Any, path: str) -> list[str]:
    """The leaves that differ between two plain values, as texts."""
    if isinstance(old, dict) and isinstance(new, dict):
        changes = []
        for key in old:
            changes += _changes(old[key], new[key], f"{path}{key}.")
        return changes
    if old == new:
        return []
    return [f"{path.rstrip('.')} {_shown(old)} → {_shown(new)}"]


def _shown(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, dict):
        return "{…}"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _ranges_of(params: ImageParams) -> tuple[HCTRange, ...]:
    return params.hct.ranges if params.hct is not None else ()


def _configured(preprocess: PreprocessParams) -> list[tuple[str, object]]:
    """The steps switched on, in run order: ``(field, value)``."""
    return [
        (field.name, value)
        for field in dataclasses.fields(preprocess)
        if (value := getattr(preprocess, field.name)) is not None
    ]


def _kept_stages(
    stages: Sequence[Stage], configured: Sequence[tuple[str, object]]
) -> int:
    """How many leading stages match the configured steps."""
    kept = 0
    for stage, (field, value) in zip(stages, configured, strict=False):
        if stage.field != field or stage.value != value:
            break
        kept += 1
    return kept


def _check_names(cls: type, fields: Mapping[str, object], where: str) -> None:
    valid = [f.name for f in dataclasses.fields(cls)]
    if where == "set_ranges":
        valid = [name for name in valid if name not in ("d_min", "d_max")]
    unknown = [name for name in fields if name not in valid]
    if unknown:
        noun = "field" if len(unknown) == 1 else "fields"
        listed = ", ".join(repr(name) for name in unknown)
        raise ParamsError(
            f"{where}: unknown {noun} {listed} (valid: {', '.join(valid)})"
        )


def _with_fields(
    cls: type, current: Any, fields: Mapping[str, object], where: str
) -> Any:
    """``current`` with ``fields`` replaced, or a new ``cls`` from
    ``fields`` alone when ``current`` is None; the class checks the
    values."""
    _check_names(cls, fields, where)
    if current is not None:
        return dataclasses.replace(current, **fields)
    missing = [
        f.name
        for f in dataclasses.fields(cls)
        if f.default is dataclasses.MISSING
        and f.default_factory is dataclasses.MISSING
        and f.name not in fields
    ]
    if missing:
        raise ParamsError(f"{where}: needs {', '.join(missing)}")
    return cls(**fields)


def _log(preview: Preview) -> None:
    state = preview._state
    logger.info(
        "%s: preview, %d circles in %d ranges, %.1f s",
        state.name,
        len(state.particles),
        len(state.passes),
        state.seconds,
    )


class _Steps:
    """A progress bar of steps, ``"<image> · <step>"``; one per call."""

    def __init__(self, name: str, total: int, progress: bool) -> None:
        self._name = name
        # a change that reuses everything shows no bar
        self._bar = tqdm(
            total=total, unit="step", disable=not progress or total == 0
        )
        self._started = False

    def __call__(self, step: str) -> None:
        # a step is done when the next one starts
        if self._started:
            self._bar.update(1)
        self._started = True
        self._bar.set_description(f"{self._name} · {step}")

    def __enter__(self) -> "_Steps":
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        if self._started and kind is None:
            self._bar.update(1)
        self._bar.close()
