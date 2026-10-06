"""Parameters of the computations, as frozen dataclasses.

Each class holds only values that control a computation (D-021). They
are validated when created (including by ``dataclasses.replace``),
immutable, comparable with ``==`` and hashable, so they are safe to
share between threads and to compare with a recorded run. Defaults are
neutral; a user's usual settings come from the Phase 3 config.
"""

import math
import numbers
from dataclasses import dataclass

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


@dataclass(frozen=True)
class HCTRange:
    """One Hough Circle Transform pass over a range of diameters (px).

    Attributes:
        d_min: Smallest circle diameter searched, in px.
        d_max: Largest circle diameter searched, in px.
        dist2R: Minimum distance between detected centres, relative
            to ``d_max``.
        param1: Upper threshold of the Canny edge detector inside
            OpenCV's HoughCircles (the lower one is half of it).
        param2: Accumulator threshold; lower values find more circles
            and more false ones.
    """

    d_min: int
    d_max: int
    # established HCT name, renamed with the detector in 2.6
    dist2R: float = 0.4  # noqa: N815
    param1: float = 200.0
    param2: float = 15.0

    def __post_init__(self) -> None:
        d_min = _integer("d_min", self.d_min)
        d_max = _integer("d_max", self.d_max)
        if not 1 <= d_min < d_max:
            raise ParamsError(
                f"need 1 <= d_min < d_max, got d_min={d_min}, d_max={d_max}"
            )
        _positive("dist2R", self.dist2R)
        _positive("param1", self.param1)
        _positive("param2", self.param2)


@dataclass(frozen=True)
class HCTParams:
    """All HCT ranges for one image, run one after another.

    Attributes:
        ranges: The passes, at least one. A list is stored as a tuple.
    """

    ranges: tuple[HCTRange, ...]

    def __post_init__(self) -> None:
        ranges = tuple(self.ranges)
        if not ranges:
            raise ParamsError("HCTParams needs at least one HCTRange")
        for item in ranges:
            if not isinstance(item, HCTRange):
                raise ParamsError(
                    f"ranges must contain HCTRange objects, got {item!r}"
                )
        # frozen: the normalised value is set through object.__setattr__
        object.__setattr__(self, "ranges", ranges)


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
        _positive("clip_limit", self.clip_limit)
        grid = tuple(self.tile_grid)
        if len(grid) != 2:
            raise ParamsError(f"tile_grid needs 2 values, got {grid!r}")
        for size in grid:
            if _integer("tile_grid", size) < 1:
                raise ParamsError(f"tile_grid values must be >= 1: {grid}")
        object.__setattr__(self, "tile_grid", grid)


@dataclass(frozen=True)
class RollingBallParams:
    """Rolling-ball background subtraction.

    Attributes:
        radius: Ball radius in px, > 0.
        light_background: True if particles are darker than the
            background.
        presmooth: Apply a 3x3 mean blur before estimating the
            background.
    """

    radius: float
    light_background: bool = True
    presmooth: bool = True

    def __post_init__(self) -> None:
        _positive("radius", self.radius)


@dataclass(frozen=True)
class PreprocessParams:
    """Image preprocessing before detection. Defaults change nothing.

    Attributes:
        crop_ratio: Fraction of the image height cut off at the bottom
            (the SEM info bar), 0 <= crop_ratio < 1.
        median_blur: Odd kernel size >= 3, or None for no blur.
        clahe: CLAHE settings, or None.
        rolling_ball: Background subtraction settings, or None.
    """

    crop_ratio: float = 0.0
    median_blur: int | None = None
    clahe: ClaheParams | None = None
    rolling_ball: RollingBallParams | None = None

    def __post_init__(self) -> None:
        crop = _number("crop_ratio", self.crop_ratio)
        if not 0.0 <= crop < 1.0:
            raise ParamsError(f"need 0 <= crop_ratio < 1, got {crop}")
        if self.median_blur is not None:
            kernel = _integer("median_blur", self.median_blur)
            if kernel < 3 or kernel % 2 == 0:
                raise ParamsError(
                    f"median_blur must be odd and >= 3, got {kernel}"
                )


@dataclass(frozen=True)
class AnalysisParams:
    """Parameters of agglomerate classification.

    Attributes:
        collector_threshold: If the second-largest member's diameter
            divided by the largest member's diameter is at or below
            this value, the agglomerate is a collector (its largest
            particle collected the others); otherwise all members are
            similar. 0 <= threshold <= 1.
    """

    collector_threshold: float = 0.5

    def __post_init__(self) -> None:
        value = _number("collector_threshold", self.collector_threshold)
        if not 0.0 <= value <= 1.0:
            raise ParamsError(
                f"need 0 <= collector_threshold <= 1, got {value}"
            )
