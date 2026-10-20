"""Parameters of the computations, as frozen dataclasses.

Each class holds only values that control a computation (D-021). They
are validated when created (including by ``dataclasses.replace``),
immutable, comparable with ``==`` and hashable, so they are safe to
share between threads and to compare with a recorded run. Defaults are
neutral; a user's usual settings come from the Phase 3 config.
"""

import math
import numbers
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from agglpy.errors import ParamsError


def _number(name: str, value: object) -> float:
    """Return ``value`` as a float, rejecting bools, text and NaN/inf.

    ``numbers.Real`` also accepts NumPy scalars (values taken from a
    DataFrame). ``bool`` is excluded explicitly: it is a subclass of
    ``int``, and YAML turns ``yes`` into ``True``.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ParamsError(f"{name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ParamsError(f"{name} must be finite, got {value!r}")
    return number


def _integer(name: str, value: object) -> int:
    """Return ``value`` as an int, rejecting bools and non-integers."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ParamsError(f"{name} must be an integer, got {value!r}")
    return int(value)


def _positive(name: str, value: object) -> float:
    number = _number(name, value)
    if number <= 0:
        raise ParamsError(f"{name} must be > 0, got {value!r}")
    return number


def _flag(name: str, value: object) -> bool:
    """Return ``value`` as a bool; only True/False are accepted.

    Text such as ``"no"`` is rejected instead of being read as true.
    """
    if not isinstance(value, bool | np.bool_):
        raise ParamsError(f"{name} must be True or False, got {value!r}")
    return bool(value)


def _sequence(name: str, value: object) -> tuple[object, ...]:
    """Return ``value`` as a tuple; text and non-iterables are rejected."""
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise ParamsError(f"{name} must be a sequence, got {value!r}")
    return tuple(value)


def _set(obj: object, name: str, value: object) -> None:
    """Store a normalised value on a frozen dataclass.

    Validated values are stored as plain Python types (a NumPy scalar
    becomes int or float), so ``asdict`` gives a plain record.
    """
    object.__setattr__(obj, name, value)


@dataclass(frozen=True)
class HCTRange:
    """One Hough Circle Transform pass over a range of diameters (px).

    The values go to ``cv2.HoughCircles`` (``HOUGH_GRADIENT``, ``dp=1``)
    as ``minRadius = floor(d_min / 2)`` and ``maxRadius = ceil(d_max /
    2)`` (rounded outward, so the search covers every diameter from
    ``d_min`` to ``d_max``), ``minDist = min_dist_ratio * d_max /
    2``, ``param1 = canny_threshold``, ``param2 =
    accumulator_threshold``.

    Attributes:
        d_min: Smallest circle diameter searched, in px.
        d_max: Largest circle diameter searched, in px.
        min_dist_ratio: Minimum distance between detected centres, as a
            fraction of the range's largest radius (``d_max / 2``).
            OpenCV keeps the stronger of two centres closer than that,
            so a wide range can lose real overlapping small particles
            (r 30-120 and 0.4: two r 30 particles 40 px apart, 40 < 48);
            narrow the range then.
        canny_threshold: Upper threshold of the Canny edge detector
            inside OpenCV's HoughCircles (``param1``; the lower one is
            half of it).
        accumulator_threshold: Votes a centre needs (``param2``); lower
            values find more circles, and more false ones.
    """

    d_min: int
    d_max: int
    min_dist_ratio: float = 0.4
    canny_threshold: float = 200.0
    accumulator_threshold: float = 15.0

    def __post_init__(self) -> None:
        d_min = _integer("d_min", self.d_min)
        d_max = _integer("d_max", self.d_max)
        if not 1 <= d_min < d_max:
            raise ParamsError(
                f"need 1 <= d_min < d_max, got d_min={d_min}, d_max={d_max}"
            )
        _set(self, "d_min", d_min)
        _set(self, "d_max", d_max)
        for name in (
            "min_dist_ratio",
            "canny_threshold",
            "accumulator_threshold",
        ):
            _set(self, name, _positive(name, getattr(self, name)))


@dataclass(frozen=True)
class HCTParams:
    """All HCT ranges for one image, run one after another.

    Attributes:
        ranges: The passes, at least one. A list is stored as a tuple.
    """

    ranges: tuple[HCTRange, ...]

    def __post_init__(self) -> None:
        ranges = _sequence("ranges", self.ranges)
        if not ranges:
            raise ParamsError("HCTParams needs at least one HCTRange")
        for item in ranges:
            if not isinstance(item, HCTRange):
                raise ParamsError(
                    f"ranges must contain HCTRange objects, got {item!r}"
                )
        _set(self, "ranges", ranges)


@dataclass(frozen=True)
class ClaheParams:
    """Contrast Limited Adaptive Histogram Equalization (OpenCV).

    Attributes:
        clip_limit: Contrast limit, > 0.
        tile_grid: Number of tiles (columns, rows), each >= 1.
    """

    clip_limit: float
    tile_grid: tuple[int, int]

    def __post_init__(self) -> None:
        _set(self, "clip_limit", _positive("clip_limit", self.clip_limit))
        values = _sequence("tile_grid", self.tile_grid)
        if len(values) != 2:
            raise ParamsError(f"tile_grid needs 2 values, got {values!r}")
        grid = tuple(_integer("tile_grid", size) for size in values)
        if min(grid) < 1:
            raise ParamsError(f"tile_grid values must be >= 1: {grid}")
        _set(self, "tile_grid", grid)


@dataclass(frozen=True)
class RollingBallParams:
    """Rolling-ball background subtraction with ImageJ's semantics.

    The ball is scikit-image's exact one, so values may differ slightly
    from Fiji's.

    Attributes:
        radius: Ball radius in px, > 0. Larger than the largest particle
            for a pure background estimate; a smaller ball also goes
            inside large particles and raises the contrast within them.
            The time grows with radius² (about 50 s at 50 px on a
            3072 x 2073 image).
        light_background: True if the particles are darker than the
            background: the image is inverted, its background
            subtracted and the result inverted back, so the polarity
            stays.
        presmooth: Estimate the background on a 3x3 mean-blurred copy,
            less sensitive to noise; the background is subtracted from
            the unblurred image.
    """

    radius: float
    light_background: bool = False
    presmooth: bool = True

    def __post_init__(self) -> None:
        _set(self, "radius", _positive("radius", self.radius))
        for name in ("light_background", "presmooth"):
            _set(self, name, _flag(name, getattr(self, name)))


@dataclass(frozen=True)
class RegionParams:
    """The part of an image that is analysed.

    Attributes:
        bottom_fraction: Fraction of the image height cut off at the
            bottom (the SEM data bar), 0 <= bottom_fraction < 1. The
            rows kept are ``int((1 - bottom_fraction) * height)``.
    """

    bottom_fraction: float = 0.0

    def __post_init__(self) -> None:
        fraction = _number("bottom_fraction", self.bottom_fraction)
        if not 0.0 <= fraction < 1.0:
            raise ParamsError(f"need 0 <= bottom_fraction < 1, got {fraction}")
        _set(self, "bottom_fraction", fraction)


@dataclass(frozen=True)
class PreprocessParams:
    """Image preprocessing before detection. Defaults change nothing.

    The steps run in the order of the fields; ``None`` skips a step.

    Attributes:
        median_blur: Odd kernel size >= 3, or None for no blur.
        rolling_ball: Background subtraction settings, or None.
        clahe: CLAHE settings, or None.
    """

    median_blur: int | None = None
    rolling_ball: RollingBallParams | None = None
    clahe: ClaheParams | None = None

    def __post_init__(self) -> None:
        if self.median_blur is not None:
            kernel = _integer("median_blur", self.median_blur)
            if kernel < 3 or kernel % 2 == 0:
                raise ParamsError(
                    f"median_blur must be odd and >= 3, got {kernel}"
                )
            _set(self, "median_blur", kernel)
        for name, cls in (
            ("rolling_ball", RollingBallParams),
            ("clahe", ClaheParams),
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, cls):
                raise ParamsError(
                    f"{name} must be a {cls.__name__} or None, got {value!r}"
                )
