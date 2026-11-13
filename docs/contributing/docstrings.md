# Docstring standard

agglpy's docstrings follow the [numpydoc format](https://numpydoc.readthedocs.io/en/latest/format.html) and
[PEP 257](https://peps.python.org/pep-0257/), like NumPy, SciPy and pandas. This page lists what we use of them and
the few rules we add. ruff (`D`, numpy convention) and numpydoc's validator check them in CI; the exact checks are
in `pyproject.toml`. Locally: `ruff check src` and `python -X utf8 -m numpydoc lint $(git ls-files 'src/agglpy/*.py')`
(`-X utf8`: numpydoc reads files in the system encoding, which is not UTF-8 on Windows).

## Which objects

- **Public API**: names a user imports (in an `__all__`, or a public class's methods and properties without a
  leading underscore, wherever the module lives: `Collection` is public although it lives in `app/_collection.py`).
  Full numpydoc.
- **Private code** (`_name`, other names inside private modules): PEP 257 only. A one-line imperative summary,
  prose after it when the reason is not obvious, no sections required.
- **Tests**: no docstrings on test functions (pytest prints the name); a comment when the name does not say what
  is checked. Helpers in `tests/support/`: as private code.
- **Recipes** (`examples/`): the markdown cells explain; functions get a one-line summary.

## Layout

Sections in this order, each left out when empty:

```
Summary line.

Extended summary.

Parameters      Returns (or Yields)      Raises      Warns
See Also        Notes                    References
```

Each section header is underlined with dashes of its length. Everything is wrapped at 72 columns.

**Summary**: one line ending with a period. Functions and methods: imperative ("Return …", "Group …", "Draw …"),
not "Returns …" or a noun phrase. Classes, modules, properties, constants: a noun phrase ("Particle table of one
image."). Errors and warnings: when they are raised ("An image file cannot be read.").

**Extended summary**: at most about three sentences on what the object does. Not how it is computed, not why: that
goes to Notes or a docs page.

**Parameters**: one entry per argument, in signature order, one or two lines each.

```
particles : DataFrame
    Particle table with ``id``, ``x``, ``y``, ``r`` in px.
by : str or list of str, optional
    Images-table columns to group by; None gives one row.
min_core_members : int, default 7
    Particles a group of triangles needs to be a core; 3 or more.
format : {"auto", "agglpy", "imagej"}, default "auto"
    The file format; ``"auto"`` recognises it from the header.
**fields
    New values of the section's fields.
```

- Types are short and for people: `DataFrame`, `Series of bool`, `str or Path`, `int`, `ImageParams`, a value set
  `{"a", "b"}`. They repeat the annotation in words; when a signature changes, its entry changes with it.
- `, optional` when the default is None ("not given"); `, default <value>` otherwise.
- Tables: the description names the columns the function needs and their unit.
- Physical values always state their unit: px in the core, pipeline and edges, SI (m, m², m³) in `Results`.
- An argument shared with another function: "As in `particles`." instead of repeating the text.
- Class constructors are documented in the class docstring's Parameters; `__init__` gets no docstring. Fields of
  a dataclass or NamedTuple are listed under **Attributes** (same entry format): the class has no `__init__` in
  the source for the validator to compare Parameters with.

**Returns**: the type on its own line, or `name : type` for each element of a returned tuple.

```
Returns
-------
particles : DataFrame
    A validated copy with ``agglomerate_id``, ``enclosed`` and
    ``coordination`` added.
contacts : DataFrame
    One row per possible contact.
```

**Raises / Warns**: one entry per class, then a sentence starting with "If". List agglpy's errors and warnings and
built-in ones a caller would catch (`FileNotFoundError`), not every `TypeError` a wrong call could cause.

**See Also**: the function a method wraps (`agglpy.core.metrics.summary`), close relatives, a recipe. Names outside
the module are written in full.

**Notes**: details a caller needs to use the result correctly (conversions, edge cases), kept short. Definitions,
the method and caveats shared by several functions live in a docs page; never the same explanation in two places.

**Docs pages**: a docstring stands alone for the call (what to pass, what comes back, what can fail). A page is
named only in Notes, only where an explanation lives there, by title, path and section: "See *Results*
(``docs/results.md``), section *Metrics*." The path stays valid when the pages become a docs site.

**References**: published sources a docstring relies on, as `.. [1] Author, Title, Journal (year).`

**Examples**: not used yet; the recipes in `examples/` are the tested examples, linked from See Also.

## Modules and packages

A summary line, at most one short paragraph, and the docs page that explains more. No list of the module's
functions: `__all__` and the API reference give it.

## Properties and constants

Properties: a noun-phrase summary starting with *The*, *A*, *An* or *Whether* ("The particle table in use."), no
sections; the validator recognises a property by that start, since it cannot see the decorator. Public constants:
a docstring right after the assignment.

```python
SEGMENT_TYPES = ("core", "small core", "link", "loop", "dangling end", "no core")
"""The values of the ``segment_type`` column."""
```

## Markup and text

- Single backticks for parameter names and agglpy objects (`particles`, `find_agglomerates`); double backticks for
  literal code, values, column and file names (``agglomerate_id``, ``"auto"``, ``agglpy.yml``).
- Plain text formulas: ``N_aggl / N_primary``. ASCII, plus `² ³ · → √ ± ° ≤ ≥ ≠ ∞` and Greek letters.
- Written for collaborators: state the reason itself, no internal planning ids or private notes.

## Example

```python
def with_structure(self, *, min_core_members: int = 7) -> "Results":
    """Return new results with the structure columns added.

    Particles get ``PARTICLE_COLUMNS`` and agglomerates
    ``AGGLOMERATE_COLUMNS`` of `agglpy.core.structure`, image by
    image; lengths are in m. The results it is called on stay as
    they are.

    Parameters
    ----------
    min_core_members : int, default 7
        Particles a group of triangles needs to be a core; 3 or more.

    Returns
    -------
    Results
        New results with the structure columns.

    Raises
    ------
    ParamsError
        If `min_core_members` is not an integer of 3 or more.

    See Also
    --------
    agglpy.core.structure.structure_properties : The computation.

    Notes
    -----
    `subset` keeps whole agglomerates, so the columns stay valid
    after it. See *Structure of agglomerates*
    (``docs/agglomerate-structure.md``).
    """
```
