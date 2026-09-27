from __future__ import annotations

import copy
import sys
from pathlib import Path

from fontTools.ttLib import TTFont
from fontTools.ttLib.tables._f_v_a_r import Axis
from fontTools.varLib.instancer import instantiateVariableFont

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PACKAGE_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from variable_gen.ital_merge import (  # noqa: E402
    _group_substitutions,
    _lift_glyph_surface,
    _point_program_compatible,
)

FIXTURE = PACKAGE_ROOT / "tests" / "fixtures" / "sample-vf.ttf"


def _coords(font: TTFont, glyph_name: str):
    coordinates, _ = font["glyf"]._getCoordinatesAndControls(glyph_name, font["hmtx"].metrics, None)
    return list(coordinates)


def test_surface_lift_preserves_both_endpoints(tmp_path) -> None:
    roman = TTFont(FIXTURE)
    _ = roman["gvar"].variations
    italic = copy.deepcopy(roman)
    _ = italic["gvar"].variations

    glyph = italic["glyf"]["A"]
    x, y = glyph.coordinates[0]
    glyph.coordinates[0] = (x + 23, y - 11)
    glyph.recalcBounds(italic["glyf"])

    assert _point_program_compatible(roman, italic, "A")

    candidate = copy.deepcopy(roman)
    for tag in ("HVAR", "MVAR", "STAT"):
        if tag in candidate:
            del candidate[tag]

    axis = Axis()
    axis.axisTag = "ital"
    axis.minValue = 0.0
    axis.defaultValue = 0.0
    axis.maxValue = 1.0
    axis.flags = 0
    axis.axisNameID = candidate["name"].addName("Italic")
    candidate["fvar"].axes.append(axis)

    _lift_glyph_surface(candidate, roman, italic, "A")
    path = tmp_path / "surface-lift.ttf"
    candidate.save(path)
    merged = TTFont(path)
    for weight in (100, 250, 400, 700, 900):
        roman_instance = instantiateVariableFont(
            copy.deepcopy(roman), {"wght": weight}, inplace=False
        )
        italic_instance = instantiateVariableFont(
            copy.deepcopy(italic), {"wght": weight}, inplace=False
        )
        merged_roman = instantiateVariableFont(
            copy.deepcopy(merged),
            {"wght": weight, "ital": 0},
            inplace=False,
        )
        merged_italic = instantiateVariableFont(
            copy.deepcopy(merged),
            {"wght": weight, "ital": 1},
            inplace=False,
        )

        assert _coords(merged_roman, "A") == _coords(roman_instance, "A")
        assert _coords(merged_italic, "A") == _coords(italic_instance, "A")


def test_substitution_group_uses_configured_default_threshold() -> None:
    strategies = {
        "B": ("substitute", None),
        "a": ("substitute", 0.55),
    }
    alternates = {
        "B": "B.ital",
        "a": "a.ital",
    }
    grouped = _group_substitutions(
        strategies,
        alternates,
        default_threshold=0.5,
    )

    assert grouped == [
        ([{"ital": (0.5, 1.0)}], {"B": "B.ital"}),
        ([{"ital": (0.55, 1.0)}], {"a": "a.ital"}),
    ]
