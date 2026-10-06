"""Registry of implementations the synthetic cases run against.

Phase 2 adds ``"core": run_core``. Known bugs of an implementation are
listed in ``KNOWN_FAILURES`` and become strict xfails for that
implementation only: the test reports them as expected failures, and
fails as soon as one of them starts passing, so a fixed bug can't stay
marked.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from agglpy.errors import ImgDataSetBufferError

from .cases import Case
from .legacy_adapter import legacy_distribution, legacy_psd_bins, run_legacy
from .result import Result

Adapter = Callable[..., Result]

# Runs a circle case: (case, tmp_path, pixel_size=...) -> Result
ADAPTERS: dict[str, Adapter] = {"legacy": run_legacy}

# Each check, with the roadmap item that brings it to the new core.
CHECKS: dict[str, str] = {
    "grouping": "2.3",
    "idj": "2.3",
    "classify": "2.4",
    "properties": "2.4",
    "psd_bins": "2.5",
    "distribution": "2.5",
    "summary": "2.5",
}

# What each implementation can do so far; tests skip the other checks.
SUPPORTED_CHECKS: dict[str, frozenset[str]] = {"legacy": frozenset(CHECKS)}


def skip_unless_supported(adapter: str, check: str) -> None:
    """Skip the running test if ``adapter`` can't do ``check`` yet.

    A feature that is not built yet is not a bug, so this is a skip,
    not an xfail.
    """
    if check not in CHECKS:
        raise ValueError(f"unknown check {check!r}")
    if check not in SUPPORTED_CHECKS[adapter]:
        pytest.skip(f"{adapter}: {check} comes in roadmap {CHECKS[check]}")


@dataclass(frozen=True)
class StatsFunctions:
    """Binning functions of one implementation, called directly.

    ``psd_bins(start, end, periods, log=False, step=False)`` returns the
    bin edges; ``distribution(values, bins)`` returns one row per bin
    with ``left, right, mid, width, count, cumulative, count_norm,
    cumulative_norm, volume, volume_norm, volume_cumulative_norm``.
    """

    psd_bins: Callable[..., np.ndarray]
    distribution: Callable[[list[float], np.ndarray], pd.DataFrame]


# Binning functions per implementation; an implementation without them
# skips the psd_bins and distribution checks (skip_unless_supported).
STATS: dict[str, StatsFunctions] = {
    "legacy": StatsFunctions(legacy_psd_bins, legacy_distribution)
}

# Realistic SEM pixel size in metres. Runs at 1.0 keep coordinates in px.
REAL_PIXEL_SIZE = 2.5e-9


@dataclass(frozen=True)
class KnownFailure:
    reason: str
    raises: type[BaseException] | None = None


_SCALING = (
    "the legacy code scales to metres before comparing, which can change "
    "an exact result by one ulp; the new core compares in px (2.3, 2.4)"
)

# (adapter, check, case name, pixel size) -> why it fails
KNOWN_FAILURES: dict[tuple[str, str, str, float], KnownFailure] = {
    # 14 * 2.5e-9 = 3.5e-08 > 10 * 2.5e-9 + 4 * 2.5e-9
    ("legacy", "grouping", "doublet_unequal_tangent", REAL_PIXEL_SIZE): (
        KnownFailure(_SCALING)
    ),
    # 6 * 2.5e-9 = 1.5000000000000002e-08 > 10 * 2.5e-9 - 4 * 2.5e-9
    ("legacy", "idj", "doublet_internally_tangent", REAL_PIXEL_SIZE): (
        KnownFailure(_SCALING)
    ),
    # 14 * 2.5e-9 / (20 * 2.5e-9) = 0.7000000000000001 > 0.7
    ("legacy", "classify", "ratio_at_0.7", REAL_PIXEL_SIZE): (
        KnownFailure(_SCALING)
    ),
    # np.arange(start, end + step, step) can keep end + step as well
    ("legacy", "psd_bins", "lin_step_metres", 1.0): KnownFailure(
        "linear step bins get an extra edge beyond end through float "
        "rounding in np.arange (fix in 2.5)"
    ),
    ("legacy", "psd_bins", "lin_step_from_0.1um", 1.0): KnownFailure(
        "linear step bins get an extra edge beyond end through float "
        "rounding in np.arange (fix in 2.5)"
    ),
    # the constructor rejects an empty particle table (D-022)
    ("legacy", "summary", "no_particles", 1.0): KnownFailure(
        "an image without particles raises instead of giving an empty "
        "summary (2.5)",
        raises=ImgDataSetBufferError,
    ),
    ("legacy", "summary", "no_particles", REAL_PIXEL_SIZE): KnownFailure(
        "an image without particles raises instead of giving an empty "
        "summary (2.5)",
        raises=ImgDataSetBufferError,
    ),
    ("legacy", "grouping", "chain_2500", 1.0): KnownFailure(
        "recursive search exceeds the recursion limit (D-015, fixed in 2.3)",
        raises=RecursionError,
    ),
}


def expect_known_failure(
    request: pytest.FixtureRequest,
    adapter: str,
    check: str,
    case: Case,
    pixel_size: float = 1.0,
) -> None:
    """Mark the running test as a strict xfail if it is a known failure."""
    known = KNOWN_FAILURES.get((adapter, check, case.name, pixel_size))
    if known is not None:
        request.applymarker(
            pytest.mark.xfail(
                strict=True, reason=known.reason, raises=known.raises
            )
        )
