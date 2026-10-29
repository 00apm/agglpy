"""``Preview``: one image's detection in memory, for tuning params."""

import dataclasses
import numbers
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import TracebackType
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm.auto import tqdm

from agglpy.app._config import ImageParams
from agglpy.app._detect import warn_if_lossy
from agglpy.app._records import now
from agglpy.app._report import warning_text
from agglpy.core.region import Bounds
from agglpy.errors import ParamsError
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
from agglpy.tables import validate_particles

if TYPE_CHECKING:
    from agglpy.app._collection import Collection

__all__ = ["Preview", "make"]

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

    def __repr__(self) -> str:
        state = self._state
        return (
            f"Preview({state.name!r}, {len(state.particles)} circles in "
            f"{len(state.passes)} ranges)"
        )

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
