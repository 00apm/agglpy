"""The tables ``detect`` and ``import_particles`` return."""

import warnings
from collections.abc import Iterable, Mapping, Sequence

import pandas as pd

__all__ = ["REPORT_COLUMNS", "class_names", "reemit", "report", "warning_text"]

REPORT_COLUMNS = ("image", "outcome", "particle_count", "message", "warnings")


def report(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """One row per image; ``particle_count`` empty unless written."""
    table = pd.DataFrame(list(rows), columns=list(REPORT_COLUMNS))
    return table.astype(
        {
            "image": object,
            "outcome": object,
            "particle_count": "Int64",
            "message": object,
            "warnings": object,
        }
    )


def warning_text(w: warnings.WarningMessage) -> str:
    """A caught warning as stored in a record: ``"<Class>: <message>"``."""
    return f"{w.category.__name__}: {w.message}"


def class_names(texts: Iterable[str]) -> str:
    """The classes of record warning texts, once each, in order."""
    names = dict.fromkeys(text.split(":", 1)[0] for text in texts)
    return ", ".join(names)


def reemit(caught: Iterable[warnings.WarningMessage], stacklevel: int) -> None:
    """Warn the caught warnings again, as they are.

    ``stacklevel`` counts from this function: 3 points at the line
    that called the public method calling ``reemit``.
    """
    for w in caught:
        warnings.warn(w.message, w.category, stacklevel=stacklevel)
