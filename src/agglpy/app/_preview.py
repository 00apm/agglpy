"""``Preview``: one image's detection in memory, for tuning params."""

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import TracebackType
from typing import TYPE_CHECKING

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
from agglpy.params import HCTRange, PreprocessParams, RegionParams
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


@dataclass(frozen=True)
class _Prepared:
    """The image read, cut and preprocessed (arrays read-only).

    ``warnings`` are those of reading and preprocessing, as record
    texts; ``seconds`` the time it all took.
    """

    raw: NDArray[np.uint8]
    bounds: Bounds
    cut: NDArray[np.uint8]
    stages: tuple[Stage, ...]
    warnings: tuple[str, ...]
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
    returns a new preview and leaves this one as it was. A preview
    writes no file.
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

    def __repr__(self) -> str:
        state = self._state
        return (
            f"Preview({state.name!r}, {len(state.particles)} circles in "
            f"{len(state.passes)} ranges)"
        )


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
    ranges = params.hct.ranges if params.hct is not None else ()
    steps = 1 + _step_count(params.preprocess) + (1 if ranges else 0)
    start = perf_counter()
    with _Steps(name, steps, progress) as report:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_if_lossy(entry.file, path)
        source = tuple(warning_text(w) for w in caught)
        prepared = _prepare(path, params.region, params.preprocess, report)
        passes = _run_passes(prepared.final, ranges, (), report)
    particles, assembly = _assemble(prepared, passes)
    preview = Preview(
        _State(
            collection=collection,
            name=name,
            params=params,
            prepared=prepared,
            preprocess_reused=False,
            passes=passes,
            particles=particles,
            warnings=source + prepared.warnings + assembly,
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
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        on_step("reading")
        raw = read_image(path)
        raw.setflags(write=False)
        bounds, cut = cut_region(raw, region)
        stages = preprocess_stages(cut, preprocess, on_step=on_step)
    return _Prepared(
        raw=raw,
        bounds=bounds,
        cut=cut,
        stages=stages,
        warnings=tuple(warning_text(w) for w in caught),
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


def _step_count(preprocess: PreprocessParams) -> int:
    return sum(
        value is not None
        for value in (
            preprocess.median_blur,
            preprocess.rolling_ball,
            preprocess.clahe,
        )
    )


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
        self._bar = tqdm(total=total, unit="step", disable=not progress)
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
