"""agglpy.yml: per-image params, shape checks, problem reports."""

from pathlib import Path

import pytest

from agglpy.app._config import NEUTRAL, template
from agglpy.app._config_file import CONFIG_NAME, ConfigFile
from agglpy.errors import FileFormatError, ParamsError
from agglpy.params import RollingBallParams

YML = """\
defaults:
  preprocess:
    rolling_ball: {radius: 50, light_background: true}
  hct:
    ranges: [{d_min: 8, d_max: 30}]
images:
  D7-019:
    preprocess:
      rolling_ball: {radius: 30}
  D7-021:
    preprocess:
      rolling_ball: null
"""

IMAGES = ["D7-017", "D7-019", "D7-021"]


@pytest.fixture
def cfg(tmp_path: Path):
    def read(text: str) -> ConfigFile:
        path = tmp_path / CONFIG_NAME
        path.write_text(text, encoding="utf-8")
        return ConfigFile.read(path)

    return read


def test_config_name() -> None:
    assert CONFIG_NAME == "agglpy.yml"


def test_params_of_defaults_and_images(cfg) -> None:
    c = cfg(YML)
    assert c.params(None).preprocess.rolling_ball == RollingBallParams(
        50.0, light_background=True
    )
    assert c.params("D7-019").preprocess.rolling_ball == RollingBallParams(
        30.0, light_background=True
    )
    assert c.params("D7-021").preprocess.rolling_ball is None
    assert c.params("D7-017") == c.params(None)
    c.check(IMAGES)


@pytest.mark.parametrize("text", ["{}\n", "defaults:\nimages:\n", ""])
def test_missing_sections_are_empty(cfg, text: str) -> None:
    assert cfg(text).params(None) == NEUTRAL


def test_unknown_top_level_key(cfg) -> None:
    with pytest.raises(ParamsError) as e:
        cfg("default: {}\n")
    assert str(e.value) == (
        "agglpy.yml has 1 problem:\n"
        "  default: unknown key (valid: defaults, images)"
    )


def test_shape_problems_are_reported_at_once(cfg) -> None:
    with pytest.raises(ParamsError) as e:
        cfg("defaults: 3\nimages:\n  A: [1]\n  B: {}\n")
    assert str(e.value) == (
        "agglpy.yml has 2 problems:\n"
        "  defaults: must be a mapping, got 3\n"
        "  images.A: must be a mapping, got [1]"
    )


def test_file_must_be_a_mapping(cfg) -> None:
    with pytest.raises(ParamsError, match="must be a mapping"):
        cfg("- 1\n")


def test_problems_report_layout(cfg) -> None:
    text = YML + "    hct:\n      ranges: [{d_min: 40, d_max: 30}]\n"
    text = text.replace(
        "rolling_ball: {radius: 50",
        "median_blr: 3\n    rolling_ball: {radius: 50",
    )
    with pytest.raises(ParamsError) as e:
        cfg(text).check(IMAGES)
    assert str(e.value) == (
        "agglpy.yml has 2 problems:\n"
        "  defaults.preprocess.median_blr: unknown key "
        "(valid: median_blur, rolling_ball, clahe)\n"
        "  images.D7-021.hct.ranges[0]: need 1 <= d_min < d_max, "
        "got d_min=40, d_max=30"
    )


def test_params_of_an_image_raise_with_the_same_layout(cfg) -> None:
    text = YML.replace("rolling_ball: {radius: 30}", "rolling_ball: 30")
    c = cfg(text)
    assert c.params("D7-021").preprocess.rolling_ball is None
    with pytest.raises(ParamsError) as e:
        c.params("D7-019")
    assert str(e.value) == (
        "agglpy.yml has 1 problem:\n"
        "  images.D7-019.preprocess.rolling_ball: "
        "must be a mapping or null, got 30"
    )


def test_default_problem_is_listed_once(cfg) -> None:
    text = "defaults:\n  preprocess:\n    rolling_ball: {radius: -1}\n"
    text += "images:\n  A:\n    preprocess:\n"
    text += "      rolling_ball: {light_background: true}\n"
    with pytest.raises(ParamsError) as e:
        cfg(text).check(["A", "B", "C"])
    message = str(e.value)
    assert message.startswith("agglpy.yml has 1 problem:\n")
    assert "  defaults.preprocess.rolling_ball: radius must be > 0" in message


def test_problem_from_image_mixed_with_default(cfg) -> None:
    text = "defaults:\n  preprocess:\n    rolling_ball: {radius: 50}\n"
    text += "images:\n  X:\n    preprocess:\n"
    text += "      rolling_ball: {light_background: 'yes'}\n"
    with pytest.raises(ParamsError) as e:
        cfg(text).check(["X"])
    message = str(e.value)
    assert message.startswith("agglpy.yml has 1 problem:\n")
    assert "  images.X.preprocess.rolling_ball: light_background" in message


def test_entry_for_unknown_image(cfg) -> None:
    c = cfg("images:\n  d7-019: {}\n  D7-091: {}\n")
    with pytest.raises(ParamsError) as e:
        c.check(["D7-019"])
    assert str(e.value) == (
        "agglpy.yml has 2 problems:\n"
        "  images.d7-019: no image of that name in the sheet "
        "(did you mean 'D7-019'?)\n"
        "  images.D7-091: no image of that name in the sheet"
    )


def test_unquoted_numeric_image_key_is_a_problem(cfg) -> None:
    with pytest.raises(ParamsError) as e:
        cfg("images:\n  007: {}\n")
    assert (
        "images: key 7 is not text; quote image names in agglpy.yml, "
        "e.g. '007'"
    ) in str(e.value)


def test_quoted_numeric_image_key_is_text(cfg) -> None:
    c = cfg("images:\n  '007':\n    region: {bottom_fraction: 0.1}\n")
    assert c.params("007").region.bottom_fraction == 0.1


def test_syntax_error_is_file_format_error(cfg) -> None:
    with pytest.raises(FileFormatError, match=r"agglpy\.yml, line \d"):
        cfg("defaults: [\n")


def test_text_keeps_the_file(cfg) -> None:
    assert cfg(template()).text() == template()
    assert cfg(YML).text() == YML


def test_path_is_kept(cfg, tmp_path: Path) -> None:
    assert cfg(YML).path == tmp_path / CONFIG_NAME
