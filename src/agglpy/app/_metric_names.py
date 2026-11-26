"""The summary columns: a quick reference for a notebook."""

import pandas as pd

from agglpy.core.metrics import _METRICS

# SI unit of each length dimension the summary columns have.
_UNITS = {0: "", 1: "m", -2: "m⁻²"}


def metric_names() -> pd.DataFrame:
    """List the summary columns with family, unit and a short meaning.

    A quick reference: the full definitions and formulas are in
    *Results* (``docs/results.md``), section *Metrics*.

    Returns
    -------
    DataFrame
        One row per summary column, in the default column order:
        ``name``, ``family``, ``unit`` (SI, as in `Results`; empty for
        no unit) and ``meaning`` (one line).

    See Also
    --------
    Results.summary : The metrics of each group of images.
    """
    return pd.DataFrame(
        {
            "name": list(_METRICS),
            "family": [m.family for m in _METRICS.values()],
            "unit": [_UNITS[m.dimension] for m in _METRICS.values()],
            "meaning": [m.meaning for m in _METRICS.values()],
        }
    )
