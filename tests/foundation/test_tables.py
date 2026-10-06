"""Particle table: schema, coercion, validation, duplicate warning."""

import warnings

import numpy as np
import pandas as pd
import pytest

from agglpy.errors import DuplicateParticlesWarning, ParticleTableError
from agglpy.tables import (
    PARTICLE_COLUMNS,
    find_duplicates,
    make_particles,
    validate_particles,
)


def _table(**columns: list) -> pd.DataFrame:
    base = {
        "id": [1, 2],
        "x": [10.0, 50.0],
        "y": [10.0, 50.0],
        "r": [5.0, 5.0],
        "source": ["hct", "hct"],
    }
    base.update(columns)
    return pd.DataFrame(base)


def test_make_particles_numbers_ids_from_one() -> None:
    t = make_particles([10, 50], [10, 50], [5, 6])
    assert list(t.columns) == list(PARTICLE_COLUMNS)
    assert t["id"].tolist() == [1, 2]
    assert t["source"].tolist() == ["hct", "hct"]
    assert t["id"].dtype == np.int64 and t["r"].dtype == np.float64


def test_empty_table_is_valid() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning either
        t = make_particles([], [], [])
    assert len(t) == 0
    assert list(t.columns) == list(PARTICLE_COLUMNS)
    assert t["id"].dtype == np.int64 and t["x"].dtype == np.float64


def test_integral_float_ids_are_coerced() -> None:
    # CSV readers often give ids as 1.0, 2.0
    t = validate_particles(_table(id=[1.0, 2.0]))
    assert t["id"].tolist() == [1, 2] and t["id"].dtype == np.int64


def test_fractional_ids_are_rejected() -> None:
    with pytest.raises(ParticleTableError, match="id"):
        validate_particles(_table(id=[1.5, 2.0]))


def test_duplicate_ids_are_rejected() -> None:
    with pytest.raises(ParticleTableError, match=r"unique.*\[1\]"):
        validate_particles(_table(id=[1, 1]))


def test_missing_column_is_named() -> None:
    with pytest.raises(ParticleTableError, match="source"):
        validate_particles(_table().drop(columns="source"))


def test_non_numeric_column_is_rejected() -> None:
    with pytest.raises(ParticleTableError, match="x"):
        validate_particles(_table(x=["a", "b"]))


def test_numeric_text_is_coerced() -> None:
    t = validate_particles(_table(x=["10.5", "50"]))
    assert t["x"].tolist() == [10.5, 50.0]


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_coordinates_are_rejected(bad: float) -> None:
    with pytest.raises(ParticleTableError, match="y"):
        validate_particles(_table(y=[10.0, bad]))


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_radius_must_be_positive(bad: float) -> None:
    with pytest.raises(ParticleTableError, match="r"):
        validate_particles(_table(r=[5.0, bad]))


def test_missing_source_is_rejected() -> None:
    with pytest.raises(ParticleTableError, match="source"):
        validate_particles(_table(source=["hct", None]))


def test_extra_columns_are_kept_after_the_schema() -> None:
    t = _table()
    t.insert(0, "note", ["a", "b"])
    out = validate_particles(t)
    assert list(out.columns) == [*PARTICLE_COLUMNS, "note"]


def test_input_is_not_modified_and_index_is_fresh() -> None:
    t = _table(id=[1.0, 2.0])
    t.index = [7, 9]
    out = validate_particles(t)
    assert t["id"].dtype == np.float64  # untouched
    assert out.index.tolist() == [0, 1]


def test_real_duplicate_from_the_golden_image_is_found() -> None:
    # D7-019 HCT output: same centre, R = 29.4 and R = 30.5 (item 2.1)
    with pytest.warns(DuplicateParticlesWarning):
        t = make_particles(
            [43.5, 43.5, 200.0], [109.5, 109.5, 200.0], [29.4, 30.5, 10.0]
        )
    assert find_duplicates(t) == [(1, 2)]


def test_concentric_small_inside_large_is_not_a_duplicate() -> None:
    # An enclosed particle is real (the dsom model counts it); only
    # near-identical circles are a detection mistake.
    t = make_particles([100.0, 100.0], [100.0, 100.0], [30.0, 5.0])
    assert find_duplicates(t) == []


def test_tolerances_are_inclusive_and_adjustable() -> None:
    with pytest.warns(DuplicateParticlesWarning):
        t = make_particles([0.0, 2.0], [0.0, 0.0], [10.0, 11.0])
    assert find_duplicates(t) == [(1, 2)]  # 2 px, 10 %
    assert find_duplicates(t, xy_tol=1.0) == []
    assert find_duplicates(t, r_rel_tol=0.05) == []


def test_validation_warns_about_duplicates() -> None:
    t = _table(x=[10.0, 10.5], y=[10.0, 10.0])
    with pytest.warns(DuplicateParticlesWarning, match=r"\(1, 2\)"):
        validate_particles(t)


def test_duplicate_warning_can_be_switched_off() -> None:
    t = _table(x=[10.0, 10.5], y=[10.0, 10.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_particles(t, warn_duplicates=False)


@pytest.mark.parametrize(
    ("args", "ids"),
    [
        (([10, 50], [10], [5, 5]), None),  # lengths differ
        (([10, 50], [10, 50], [5, 5]), [1]),  # ids too short
        (([[10, 50]], [[10, 50]], [[5, 5]]), None),  # 2-D
        ((10.0, 10.0, 5.0), None),  # scalars
        ((["a"], [10], [5]), None),  # text
    ],
)
def test_make_particles_bad_input_is_a_table_error(
    args: tuple, ids: list | None
) -> None:
    # A frontend catching AgglpyError must catch these too.
    with pytest.raises(ParticleTableError):
        make_particles(*args, ids=ids)
