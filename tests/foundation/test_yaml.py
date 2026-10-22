"""YAML module: round trip keeps comments, written files read the same
with PyYAML, removing a key keeps the comments that describe others."""

from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from agglpy import _yaml
from agglpy.errors import FileFormatError

TEMPLATE_LIKE = """\
# top
defaults:
  preprocess:
    median_blur: 3          # kernel
    # before rb
    rolling_ball:
      radius: 50            # px
    # before clahe
    clahe: null             # off
  hct:
    ranges: []    # none yet
                  # - {d_min: 8, d_max: 30}

# before images
images:
  A:
    preprocess:
      median_blur: 5      # A blur
      # before A rb
      rolling_ball: {radius: 30}   # thin
  # before B
  B:
    region: {bottom_fraction: 0.1}   # bar
  # before C
  C:
    hct:
      ranges:
      - {d_min: 1, d_max: 4}   # tiny
  # end of images
"""

A_BLOCK = """\
  A:
    preprocess:
      median_blur: 5      # A blur
      # before A rb
      rolling_ball: {radius: 30}   # thin
"""
B_BLOCK = """\
  B:
    region: {bottom_fraction: 0.1}   # bar
"""
C_BLOCK = """\
  C:
    hct:
      ranges:
      - {d_min: 1, d_max: 4}   # tiny
"""


def _doc() -> Any:
    return _yaml.loads(TEMPLATE_LIKE, "t")


def _without(*parts: str) -> str:
    text = TEMPLATE_LIKE
    for part in parts:
        assert text.count(part) == 1
        text = text.replace(part, "")
    return text


def test_round_trip_is_byte_identical() -> None:
    assert _yaml.dumps(_doc()) == TEMPLATE_LIKE


def test_null_is_written_as_null() -> None:
    assert _yaml.dumps({"a": None}) == "a: null\n"


def test_exponent_float_and_yes_text() -> None:
    assert _yaml.loads("a: 1e-6\nb: yes\n", "t") == {"a": 1e-06, "b": "yes"}


def test_yaml11_words_are_quoted() -> None:
    data = {"on": "yes", "No": "OFF", "x": "y", "n": "Null", "t": "~"}
    text = _yaml.dumps(data)
    # PyYAML (YAML 1.1) reads these words as bools / None unless quoted.
    assert yaml.safe_load(text) == data
    assert _yaml.loads(text, "t") == data


def test_unicode_is_written_as_text() -> None:
    assert "zł: ąę" in _yaml.dumps({"zł": "ąę"})


def test_long_text_is_not_wrapped() -> None:
    assert _yaml.dumps({"a": "x " * 100}).count("\n") == 1


def test_syntax_error_names_source_and_line() -> None:
    with pytest.raises(FileFormatError, match=r"agglpy\.yml, line 2"):
        _yaml.loads("a: [1,\n", "agglpy.yml")


def test_repeated_key_is_an_error() -> None:
    with pytest.raises(FileFormatError, match="t, line 2"):
        _yaml.loads("a: 1\na: 2\n", "t")


def test_load_reads_utf8_file(tmp_path: Path) -> None:
    path = tmp_path / "agglpy.yml"
    path.write_bytes("zł: 1\n".encode())
    assert _yaml.load(path) == {"zł": 1}


def test_load_error_names_the_path(tmp_path: Path) -> None:
    path = tmp_path / "agglpy.yml"
    path.write_text("a: [1,\n", encoding="utf-8")
    with pytest.raises(FileFormatError, match="line 2") as info:
        _yaml.load(path)
    assert str(path) in str(info.value)


def test_generated_comments() -> None:
    top = _yaml.new_mapping()
    _yaml.set_top_comment(top, "agglpy")
    defaults = _yaml.new_mapping()
    defaults["ranges"] = []
    top["defaults"] = defaults
    _yaml.set_comment(
        defaults, "ranges", "required, e.g.\n- {d_min: 8, d_max: 30}", 30
    )
    assert _yaml.dumps(top) == (
        "# agglpy\n"
        "defaults:\n"
        "  ranges: []                  # required, e.g.\n"
        "                              # - {d_min: 8, d_max: 30}\n"
    )


def _remove(mapping: MutableMapping[str, Any], key: str) -> None:
    _yaml.remove(mapping, key)


def test_remove_middle_scalar_keeps_next_keys_comment() -> None:
    d = _doc()
    _remove(d["defaults"]["preprocess"], "median_blur")
    assert _yaml.dumps(d) == _without("    median_blur: 3          # kernel\n")


def test_remove_last_scalar() -> None:
    d = _doc()
    _remove(d["defaults"]["preprocess"], "clahe")
    assert _yaml.dumps(d) == _without("    clahe: null             # off\n")


def test_remove_block_value() -> None:
    d = _doc()
    _remove(d["defaults"]["preprocess"], "rolling_ball")
    assert _yaml.dumps(d) == _without(
        "    rolling_ball:\n      radius: 50            # px\n"
    )


def test_remove_first_image() -> None:
    d = _doc()
    _remove(d["images"], "A")
    assert _yaml.dumps(d) == _without(A_BLOCK)


def test_remove_middle_image() -> None:
    d = _doc()
    _remove(d["images"], "B")
    assert _yaml.dumps(d) == _without(B_BLOCK)


def test_remove_last_image_with_flow_map_in_sequence() -> None:
    d = _doc()
    _remove(d["images"], "C")
    assert _yaml.dumps(d) == _without(C_BLOCK)


def test_remove_first_key_of_nested_map() -> None:
    d = _doc()
    _remove(d["images"]["A"]["preprocess"], "median_blur")
    assert _yaml.dumps(d) == _without("      median_blur: 5      # A blur\n")


def test_remove_all_images_gives_empty_mapping() -> None:
    d = _doc()
    for name in ("A", "B", "C"):
        _remove(d["images"], name)
    text = _yaml.dumps(d)
    head = TEMPLATE_LIKE[: TEMPLATE_LIKE.index("images:")]
    assert text == head + "images: {}\n"
    assert text.endswith("# before images\nimages: {}\n")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a: 1\nb: 2  # b\n# about c\nc: 3\n", "a: 1\n# about c\nc: 3\n"),
        (
            "a:\n  x: 1\nb: 2  # b\n# about c\nc: 3\n",
            "a:\n  x: 1\n# about c\nc: 3\n",
        ),
    ],
)
def test_remove_after_key_without_comment(text: str, expected: str) -> None:
    d = _yaml.loads(text, "t")
    _yaml.remove(d, "b")
    assert _yaml.dumps(d) == expected


def test_replace_block_value_keeps_next_keys_comment() -> None:
    d = _doc()
    _yaml.replace(d["defaults"]["preprocess"], "rolling_ball", None)
    assert _yaml.dumps(d) == TEMPLATE_LIKE.replace(
        "    rolling_ball:\n      radius: 50            # px\n",
        "    rolling_ball: null\n",
    )


def test_replace_scalar_keeps_its_comment() -> None:
    d = _doc()
    _yaml.replace(d["defaults"]["preprocess"], "median_blur", 5)
    assert _yaml.dumps(d) == TEMPLATE_LIKE.replace(
        "median_blur: 3          # kernel", "median_blur: 5          # kernel"
    )


def test_replace_appends_a_new_key() -> None:
    d = _yaml.loads("a: 1\n", "t")
    _yaml.replace(d, "b", {"c": [1]})
    assert _yaml.dumps(d) == "a: 1\nb:\n  c:\n  - 1\n"
