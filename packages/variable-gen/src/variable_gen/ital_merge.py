"""Merge Roman and Italic variable-font surfaces behind a registered ital axis.

The exact path operates on compiled TrueType VFs. Compatible glyphs retain the
Roman gvar surface and receive an ital-conditioned (Italic - Roman) correction.
Incompatible glyphs keep both complete wght/wdth surfaces and switch with rvrn.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import fontTools.varLib.varStore  # noqa: F401 -- installs VarIdx traversal methods
from fontTools.misc.roundTools import otRound
from fontTools.ttLib import TTFont
from fontTools.ttLib.reorderGlyphs import reorderGlyphs
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables._f_v_a_r import Axis, NamedInstance
from fontTools.ttLib.tables._g_l_y_f import GlyphCoordinates
from fontTools.ttLib.tables.TupleVariation import TupleVariation
from fontTools.varLib.featureVars import (
    addFeatureVariations,
    buildConditionTable,
    buildFeatureTableSubstitutionRecord,
    buildFeatureVariationRecord,
    findFeatureVariationRecord,
    remapFeatures,
)
from fontTools.varLib.instancer import instantiateVariableFont
from fontTools.varLib.iup import iup_delta

from .common import PipelineError
from .config import ConfigAxis, ProjectConfig


@dataclass
class ItalMergeReport:
    output: Path
    interpolated: list[str] = field(default_factory=list)
    substituted: list[str] = field(default_factory=list)
    review: list[str] = field(default_factory=list)
    alternates: dict[str, str] = field(default_factory=dict)
    cyclic_interpolated: dict[str, list[int]] = field(default_factory=dict)
    fallback_substituted: dict[str, str] = field(default_factory=dict)
    source_hashes: dict[str, str] = field(default_factory=dict)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _axis_signature(font: TTFont) -> tuple[tuple[str, float, float, float], ...]:
    return tuple(
        (axis.axisTag, axis.minValue, axis.defaultValue, axis.maxValue)
        for axis in font["fvar"].axes
    )


def _coordinates(font: TTFont, glyph_name: str):
    return font["glyf"]._getCoordinatesAndControls(glyph_name, font["hmtx"].metrics, None)


def _point_program_compatible(roman: TTFont, italic: TTFont, glyph_name: str) -> bool:
    try:
        roman_coords, roman_control = _coordinates(roman, glyph_name)
        italic_coords, italic_control = _coordinates(italic, glyph_name)
    except KeyError:
        return False
    if len(roman_coords) != len(italic_coords):
        return False
    return (
        roman_control.numberOfContours == italic_control.numberOfContours
        and roman_control.endPts == italic_control.endPts
        and roman_control.flags == italic_control.flags
        and roman_control.components == italic_control.components
    )


CYCLIC_PHASE_IMPROVEMENT_RATIO = 0.70
CYCLIC_PHASE_AMBIGUITY_RATIO = 0.85


def _normalised_glyf_points(points, flags):
    oncurve = [point for point, flag in zip(points, flags, strict=True) if flag & 1]
    basis = oncurve or points
    xs = [point[0] for point in basis]
    ys = [point[1] for point in basis]
    x_min, y_min = min(xs), min(ys)
    width = (max(xs) - x_min) or 1.0
    height = (max(ys) - y_min) or 1.0
    return [((point[0] - x_min) / width, (point[1] - y_min) / height) for point in points]


def _cyclic_cost(points, flags, reference_points, reference_flags):
    if [flag & 1 for flag in flags] != [flag & 1 for flag in reference_flags]:
        return float("inf")
    here = _normalised_glyf_points(points, flags)
    there = _normalised_glyf_points(reference_points, reference_flags)
    return sum(
        (point[0] - target[0]) ** 2 + (point[1] - target[1]) ** 2
        for point, target in zip(here, there, strict=True)
    )


def _cyclic_shift_plan(roman: TTFont, italic: TTFont, glyph_name: str) -> list[int] | None:
    roman_glyph = roman["glyf"][glyph_name]
    italic_glyph = italic["glyf"][glyph_name]
    if roman_glyph.isComposite() or italic_glyph.isComposite():
        return None
    if roman_glyph.numberOfContours <= 0:
        return None
    if roman_glyph.numberOfContours != italic_glyph.numberOfContours:
        return None
    if len(roman_glyph.coordinates) != len(italic_glyph.coordinates):
        return None
    if roman_glyph.endPtsOfContours != italic_glyph.endPtsOfContours:
        return None
    if any(flag & 0x80 for flag in roman_glyph.flags + italic_glyph.flags):
        return None
    for glyph in (roman_glyph, italic_glyph):
        program = getattr(glyph, "program", None)
        if program is not None and program.getBytecode():
            return None

    shifts: list[int] = []
    start = 0
    for end in roman_glyph.endPtsOfContours:
        stop = end + 1
        reference_points = list(roman_glyph.coordinates[start:stop])
        reference_flags = list(roman_glyph.flags[start:stop])
        points = list(italic_glyph.coordinates[start:stop])
        flags = list(italic_glyph.flags[start:stop])
        candidates = []
        for shift in range(len(points)):
            rotated_points = points[shift:] + points[:shift]
            rotated_flags = flags[shift:] + flags[:shift]
            value = _cyclic_cost(
                rotated_points,
                rotated_flags,
                reference_points,
                reference_flags,
            )
            if value != float("inf"):
                candidates.append((value, shift))
        if not candidates:
            return None
        candidates.sort()
        best_cost, best_shift = candidates[0]
        current = next(
            (cost for cost, shift in candidates if shift == 0),
            float("inf"),
        )
        second = candidates[1][0] if len(candidates) > 1 else float("inf")
        if best_shift:
            if (
                current <= 1e-12
                or best_cost >= current * CYCLIC_PHASE_IMPROVEMENT_RATIO
                or (second <= 1e-12 or best_cost >= second * CYCLIC_PHASE_AMBIGUITY_RATIO)
            ):
                return None
        shifts.append(best_shift)
        start = stop
    return shifts if any(shifts) else None


def _rotate_contour_chunks(values, end_points, shifts):
    result = []
    start = 0
    for end, shift in zip(end_points, shifts, strict=True):
        chunk = list(values[start : end + 1])
        if chunk:
            shift %= len(chunk)
            chunk = chunk[shift:] + chunk[:shift]
        result.extend(chunk)
        start = end + 1
    return result


def _apply_cyclic_rotation(font: TTFont, glyph_name: str, shifts: list[int]) -> None:
    glyph = font["glyf"][glyph_name]
    point_count = len(glyph.coordinates)
    overlap_simple = bool(glyph.flags and glyph.flags[0] & 0x40)
    glyph.coordinates = GlyphCoordinates(
        _rotate_contour_chunks(glyph.coordinates, glyph.endPtsOfContours, shifts)
    )
    rotated_flags = _rotate_contour_chunks(glyph.flags, glyph.endPtsOfContours, shifts)
    glyph.flags = bytearray(flag & 1 for flag in rotated_flags)
    if overlap_simple and glyph.flags:
        glyph.flags[0] |= 0x40
    for variation in font["gvar"].variations.get(glyph_name, []):
        points = variation.coordinates[:point_count]
        phantoms = variation.coordinates[point_count:]
        variation.coordinates = _rotate_contour_chunks(
            points, glyph.endPtsOfContours, shifts
        ) + list(phantoms)


def _negate(coordinates):
    return [None if point is None else (-point[0], -point[1]) for point in coordinates]


def _ital_axes(axes: dict[str, tuple[float, float, float]]):
    return {**axes, "ital": (0.0, 1.0, 1.0)}


def _default_difference(roman: TTFont, italic: TTFont, glyph_name: str):
    roman_coords, _ = _coordinates(roman, glyph_name)
    italic_coords, _ = _coordinates(italic, glyph_name)
    if len(roman_coords) != len(italic_coords):
        raise PipelineError(f"{glyph_name}: endpoint point counts differ")
    return [
        (int(italic_point[0] - roman_point[0]), int(italic_point[1] - roman_point[1]))
        for roman_point, italic_point in zip(roman_coords, italic_coords, strict=True)
    ]


def _expanded_variation_coordinates(font: TTFont, glyph_name: str, variation: TupleVariation):
    if None not in variation.coordinates:
        return copy.deepcopy(variation.coordinates)
    base_coords, control = _coordinates(font, glyph_name)
    end_points = (
        control.endPts if control.numberOfContours >= 1 else list(range(len(control.endPts)))
    )
    expanded = list(iup_delta(variation.coordinates, base_coords, end_points))
    rounded = [(otRound(x), otRound(y)) for x, y in expanded]
    if any(
        abs(x - rx) > 1e-9 or abs(y - ry) > 1e-9
        for (x, y), (rx, ry) in zip(expanded, rounded, strict=True)
    ):
        raise PipelineError(
            f"{glyph_name}: Italic IUP field is fractional and cannot be "
            "lifted exactly into explicit gvar deltas"
        )
    return rounded


def _lift_glyph_surface(candidate: TTFont, roman: TTFont, italic: TTFont, glyph_name: str) -> None:
    if not _point_program_compatible(roman, italic, glyph_name):
        raise PipelineError(
            f"{glyph_name}: interpolate requested without exact point compatibility"
        )
    roman_vars = copy.deepcopy(roman["gvar"].variations.get(glyph_name, []))
    italic_vars = copy.deepcopy(italic["gvar"].variations.get(glyph_name, []))
    variations = list(roman_vars)
    variations.append(
        TupleVariation(
            {"ital": (0.0, 1.0, 1.0)},
            _default_difference(roman, italic, glyph_name),
        )
    )
    variations.extend(
        TupleVariation(_ital_axes(var.axes), _negate(var.coordinates)) for var in roman_vars
    )
    variations.extend(
        TupleVariation(
            _ital_axes(var.axes),
            _expanded_variation_coordinates(italic, glyph_name, var),
        )
        for var in italic_vars
    )
    candidate["gvar"].variations[glyph_name] = variations


def _unique_alternate_name(name: str, occupied: set[str]) -> str:
    base = f"{name}.ital"
    if base not in occupied:
        return base
    index = 2
    while f"{base}{index}" in occupied:
        index += 1
    return f"{base}{index}"


def _copy_italic_alternates(
    candidate: TTFont,
    italic: TTFont,
    glyph_names: list[str],
) -> dict[str, str]:
    occupied = set(candidate.getGlyphOrder())
    mapping: dict[str, str] = {}
    for name in glyph_names:
        if name not in italic.getGlyphOrder():
            continue
        alt_name = _unique_alternate_name(name, occupied)
        mapping[name] = alt_name
        occupied.add(alt_name)

    order = list(candidate.getGlyphOrder())
    for name, alt_name in mapping.items():
        candidate["glyf"].glyphs[alt_name] = copy.deepcopy(italic["glyf"][name])
        candidate["hmtx"].metrics[alt_name] = tuple(italic["hmtx"].metrics[name])
        candidate["gvar"].variations[alt_name] = copy.deepcopy(
            italic["gvar"].variations.get(name, [])
        )
        order.append(alt_name)

    candidate.setGlyphOrder(order)
    for _name, alt_name in mapping.items():
        glyph = candidate["glyf"][alt_name]
        if not glyph.isComposite():
            continue
        for component in glyph.components:
            component.glyphName = mapping.get(component.glyphName, component.glyphName)
    return mapping


def _region_signature(region, *, ital_conditioned: bool):
    axes = [(axis.StartCoord, axis.PeakCoord, axis.EndCoord) for axis in region.VarRegionAxis]
    axes.append((0.0, 1.0, 1.0) if ital_conditioned else (0.0, 0.0, 0.0))
    return tuple(axes)


def _neutral_ital_signature(axis_count: int, *, conditioned: bool):
    support = [(0.0, 0.0, 0.0)] * axis_count
    support.append((0.0, 1.0, 1.0) if conditioned else (0.0, 0.0, 0.0))
    return tuple(support)


def _hvar_deltas(font: TTFont, glyph_name: str):
    table = font["HVAR"].table
    if table.AdvWidthMap is None:
        raise PipelineError("HVAR without AdvWidthMap is not yet supported")
    varidx = table.AdvWidthMap.mapping[glyph_name]
    outer = varidx >> 16
    inner = varidx & 0xFFFF
    var_data = table.VarStore.VarData[outer]
    deltas = var_data.Item[inner]
    return [
        (table.VarStore.VarRegionList.Region[region_index], delta)
        for region_index, delta in zip(var_data.VarRegionIndex, deltas, strict=True)
        if delta
    ]


def _build_var_region(signature):
    region = ot.VarRegion()
    region.VarRegionAxis = []
    for start, peak, end in signature:
        axis = ot.VarRegionAxis()
        axis.StartCoord = start
        axis.PeakCoord = peak
        axis.EndCoord = end
        region.VarRegionAxis.append(axis)
    return region


def _rebuild_hvar(
    candidate: TTFont,
    roman: TTFont,
    italic: TTFont,
    resolved: dict[str, tuple[str, float | None]],
    alternates: dict[str, str],
) -> None:
    if "HVAR" not in roman or "HVAR" not in italic:
        raise PipelineError("Both source fonts need HVAR for exact advance merging")
    for source in (roman, italic):
        table = source["HVAR"].table
        if table.LsbMap is not None or table.RsbMap is not None:
            raise PipelineError("HVAR LSB/RSB maps are not yet supported")

    region_map: dict[tuple, int] = {}
    regions: list[ot.VarRegion] = []
    var_data_rows: list[ot.VarData] = []
    mapping: dict[str, int] = {}
    italic_by_alt = {alternate: source for source, alternate in alternates.items()}
    source_axis_count = len(roman["fvar"].axes)

    def add_region(signature) -> int:
        index = region_map.get(signature)
        if index is None:
            index = len(regions)
            region_map[signature] = index
            regions.append(_build_var_region(signature))
        return index

    def add_deltas(accumulator, source, glyph_name, *, conditioned, sign=1):
        for region, delta in _hvar_deltas(source, glyph_name):
            signature = _region_signature(region, ital_conditioned=conditioned)
            accumulator[signature] += sign * int(delta)

    for glyph_name in candidate.getGlyphOrder():
        deltas: dict[tuple, int] = defaultdict(int)
        if glyph_name in italic_by_alt:
            source_name = italic_by_alt[glyph_name]
            add_deltas(deltas, italic, source_name, conditioned=False)
        elif glyph_name in roman.getGlyphOrder():
            strategy = resolved.get(glyph_name, ("review", None))[0]
            add_deltas(deltas, roman, glyph_name, conditioned=False)
            if strategy == "interpolate" and glyph_name in italic.getGlyphOrder():
                base_delta = (
                    italic["hmtx"].metrics[glyph_name][0] - roman["hmtx"].metrics[glyph_name][0]
                )
                if base_delta:
                    signature = _neutral_ital_signature(source_axis_count, conditioned=True)
                    deltas[signature] += int(base_delta)
                add_deltas(deltas, roman, glyph_name, conditioned=True, sign=-1)
                add_deltas(deltas, italic, glyph_name, conditioned=True)

        active = [(signature, delta) for signature, delta in deltas.items() if delta]
        active.sort(key=lambda item: item[0])
        region_indices = [add_region(signature) for signature, _ in active]
        row_deltas = [delta for _, delta in active]

        var_data = ot.VarData()
        var_data.ItemCount = 1
        var_data.NumShorts = len(region_indices)
        var_data.VarRegionCount = len(region_indices)
        var_data.VarRegionIndex = region_indices
        var_data.Item = [row_deltas]
        outer = len(var_data_rows)
        var_data_rows.append(var_data)
        mapping[glyph_name] = outer << 16

    region_list = ot.VarRegionList()
    region_list.RegionAxisCount = len(candidate["fvar"].axes)
    region_list.RegionCount = len(regions)
    region_list.Region = regions

    store = ot.VarStore()
    store.Format = 1
    store.VarRegionList = region_list
    store.VarDataCount = len(var_data_rows)
    store.VarData = var_data_rows

    table = candidate["HVAR"].table
    table.VarStore = store
    table.AdvWidthMap.mapping = mapping
    table.LsbMap = None
    table.RsbMap = None


def _rebuild_hvar_preserving_sources(
    candidate: TTFont,
    roman: TTFont,
    italic: TTFont,
    resolved: dict[str, tuple[str, float | None]],
    alternates: dict[str, str],
) -> None:
    roman_table = candidate["HVAR"].table
    italic_table = italic["HVAR"].table
    if (
        roman_table.AdvWidthMap is None
        or italic_table.AdvWidthMap is None
        or roman_table.LsbMap is not None
        or roman_table.RsbMap is not None
        or italic_table.LsbMap is not None
        or italic_table.RsbMap is not None
    ):
        raise PipelineError("HVAR map shape is not supported for exact ital merge")

    store = roman_table.VarStore
    source_axis_count = len(roman["fvar"].axes)
    _extend_varstore_axis(store, source_axis_count + 1)
    regions = store.VarRegionList.Region

    def append_region(source_region, *, conditioned):
        region = copy.deepcopy(source_region)
        if len(region.VarRegionAxis) == source_axis_count + 1:
            axis = region.VarRegionAxis[-1]
        else:
            axis = ot.VarRegionAxis()
            region.VarRegionAxis.append(axis)
        axis.StartCoord = 0.0
        axis.PeakCoord = 1.0 if conditioned else 0.0
        axis.EndCoord = 1.0 if conditioned else 0.0
        regions.append(region)
        return len(regions) - 1

    roman_conditioned = [append_region(region, conditioned=True) for region in list(regions)]
    italic_neutral = [
        append_region(region, conditioned=False)
        for region in italic_table.VarStore.VarRegionList.Region
    ]
    italic_conditioned = [
        append_region(region, conditioned=True)
        for region in italic_table.VarStore.VarRegionList.Region
    ]

    base_region = ot.VarRegion()
    base_region.VarRegionAxis = []
    for _ in range(source_axis_count):
        axis = ot.VarRegionAxis()
        axis.StartCoord = axis.PeakCoord = axis.EndCoord = 0.0
        base_region.VarRegionAxis.append(axis)
    ital_axis = ot.VarRegionAxis()
    ital_axis.StartCoord = 0.0
    ital_axis.PeakCoord = 1.0
    ital_axis.EndCoord = 1.0
    base_region.VarRegionAxis.append(ital_axis)
    regions.append(base_region)
    base_region_index = len(regions) - 1
    store.VarRegionList.RegionCount = len(regions)

    roman_data_count = len(store.VarData)
    for source_data in italic_table.VarStore.VarData:
        var_data = copy.deepcopy(source_data)
        var_data.VarRegionIndex = [italic_neutral[index] for index in var_data.VarRegionIndex]
        store.VarData.append(var_data)
    italic_data_offset = roman_data_count

    mapping = dict(roman_table.AdvWidthMap.mapping)
    reverse_alternates = {alternate: source for source, alternate in alternates.items()}
    for alternate, source_name in reverse_alternates.items():
        source_varidx = italic_table.AdvWidthMap.mapping[source_name]
        mapping[alternate] = (((source_varidx >> 16) + italic_data_offset) << 16) | (
            source_varidx & 0xFFFF
        )

    for glyph_name, (strategy, _) in resolved.items():
        if strategy != "interpolate":
            continue
        roman_varidx = roman_table.AdvWidthMap.mapping[glyph_name]
        italic_varidx = italic_table.AdvWidthMap.mapping[glyph_name]
        roman_outer, roman_inner = roman_varidx >> 16, roman_varidx & 0xFFFF
        italic_outer, italic_inner = italic_varidx >> 16, italic_varidx & 0xFFFF
        roman_data = store.VarData[roman_outer]
        italic_data = italic_table.VarStore.VarData[italic_outer]

        region_indices = list(roman_data.VarRegionIndex)
        row = list(roman_data.Item[roman_inner])
        region_indices.append(base_region_index)
        row.append(italic["hmtx"].metrics[glyph_name][0] - roman["hmtx"].metrics[glyph_name][0])
        region_indices.extend(roman_conditioned[index] for index in roman_data.VarRegionIndex)
        row.extend(-value for value in roman_data.Item[roman_inner])
        region_indices.extend(italic_conditioned[index] for index in italic_data.VarRegionIndex)
        row.extend(italic_data.Item[italic_inner])

        var_data = ot.VarData()
        var_data.ItemCount = 1
        var_data.NumShorts = len(region_indices)
        var_data.VarRegionCount = len(region_indices)
        var_data.VarRegionIndex = region_indices
        var_data.Item = [row]
        new_outer = len(store.VarData)
        store.VarData.append(var_data)
        mapping[glyph_name] = new_outer << 16

    store.VarDataCount = len(store.VarData)
    roman_table.AdvWidthMap.mapping = mapping


def _axis_extreme_locations(font: TTFont) -> list[dict[str, float]]:
    axes = list(font["fvar"].axes)
    values = [sorted({axis.minValue, axis.defaultValue, axis.maxValue}) for axis in axes]
    locations = [
        dict(zip((axis.axisTag for axis in axes), row, strict=True))
        for row in itertools.product(*values)
    ]
    locations.extend(dict(instance.coordinates) for instance in font["fvar"].instances)
    unique = {}
    for location in locations:
        unique[tuple(sorted(location.items()))] = location
    return list(unique.values())


def _hvar_is_redundant(font: TTFont) -> bool:
    if "HVAR" not in font:
        return True
    without_hvar = copy.deepcopy(font)
    del without_hvar["HVAR"]
    for location in _axis_extreme_locations(font):
        with_instance = instantiateVariableFont(copy.deepcopy(font), location, inplace=False)
        without_instance = instantiateVariableFont(
            copy.deepcopy(without_hvar), location, inplace=False
        )
        if with_instance["hmtx"].metrics != without_instance["hmtx"].metrics:
            return False
    return True


def _add_ital_axis(
    candidate: TTFont,
    italic: TTFont,
    axis_config: ConfigAxis,
) -> None:
    axis = Axis()
    axis.axisTag = axis_config.tag
    axis.minValue = axis_config.minimum
    axis.defaultValue = axis_config.default
    axis.maxValue = axis_config.maximum
    axis.flags = 0
    axis.axisNameID = candidate["name"].addName(axis_config.name)
    candidate["fvar"].axes.append(axis)

    for instance in candidate["fvar"].instances:
        instance.coordinates[axis_config.tag] = axis_config.default

    italic_instances: list[NamedInstance] = []
    for source in italic["fvar"].instances:
        instance = NamedInstance()
        source_name = italic["name"].getDebugName(source.subfamilyNameID) or "Italic"
        instance.subfamilyNameID = candidate["name"].addName(source_name)
        instance.flags = source.flags
        instance.coordinates = {
            **{tag: float(value) for tag, value in source.coordinates.items()},
            axis_config.tag: axis_config.maximum,
        }
        if source.postscriptNameID != 0xFFFF:
            ps_name = italic["name"].getDebugName(source.postscriptNameID)
            if ps_name:
                instance.postscriptNameID = candidate["name"].addName(ps_name)
        italic_instances.append(instance)
    candidate["fvar"].instances.extend(italic_instances)


def _add_ital_stat(candidate: TTFont, axis_config: ConfigAxis) -> None:
    if "STAT" not in candidate:
        raise PipelineError("Roman source has no STAT table")
    stat = candidate["STAT"].table
    existing_tags = [axis.AxisTag for axis in stat.DesignAxisRecord.Axis]
    if axis_config.tag in existing_tags:
        return

    axis_index = len(stat.DesignAxisRecord.Axis)
    fvar_axis = next(axis for axis in candidate["fvar"].axes if axis.axisTag == axis_config.tag)
    axis_record = ot.AxisRecord()
    axis_record.AxisTag = axis_config.tag
    axis_record.AxisNameID = fvar_axis.axisNameID
    axis_record.AxisOrdering = (
        max(
            (axis.AxisOrdering for axis in stat.DesignAxisRecord.Axis),
            default=-1,
        )
        + 1
    )
    stat.DesignAxisRecord.Axis.append(axis_record)
    stat.DesignAxisCount = len(stat.DesignAxisRecord.Axis)

    if stat.AxisValueArray is None:
        stat.AxisValueArray = ot.AxisValueArray()
        stat.AxisValueArray.AxisValue = []

    roman_value = ot.AxisValue()
    roman_value.Format = 3
    roman_value.AxisIndex = axis_index
    roman_value.Flags = 0x2
    roman_value.ValueNameID = stat.ElidedFallbackNameID
    roman_value.Value = axis_config.default
    roman_value.LinkedValue = axis_config.maximum

    italic_value = ot.AxisValue()
    italic_value.Format = 1
    italic_value.AxisIndex = axis_index
    italic_value.Flags = 0
    italic_value.ValueNameID = fvar_axis.axisNameID
    italic_value.Value = axis_config.maximum

    stat.AxisValueArray.AxisValue.extend([roman_value, italic_value])
    stat.AxisValueCount = len(stat.AxisValueArray.AxisValue)
    stat.Version = max(stat.Version, 0x00010001)


def _strategy_for(config: ProjectConfig, glyph_name: str) -> tuple[str, float | None]:
    override = config.glyphs.ital_strategies.get(glyph_name)
    if override is not None:
        return override.strategy, override.threshold
    return config.glyphs.ital_default_strategy, None


def _group_substitutions(
    strategies: dict[str, tuple[str, float | None]],
    alternates: dict[str, str],
    *,
    default_threshold: float,
) -> list[tuple[list[dict[str, tuple[float, float]]], dict[str, str]]]:
    grouped: dict[float, dict[str, str]] = {}
    for name, alt_name in alternates.items():
        _, threshold = strategies[name]
        switch = default_threshold if threshold is None else threshold
        grouped.setdefault(switch, {})[name] = alt_name
    return [
        ([{"ital": (threshold, 1.0)}], substitution)
        for threshold, substitution in sorted(grouped.items())
    ]


def _load_renamed_layout_font(path: Path, alternates: dict[str, str]) -> TTFont:
    font = TTFont(str(path))
    font.setGlyphOrder([alternates.get(name, name) for name in font.getGlyphOrder()])
    for tag in ("GDEF", "GSUB", "GPOS"):
        if tag in font:
            _ = font[tag].table
    return font


def _extend_varstore_axis(store, target_axis_count: int) -> None:
    current = store.VarRegionList.RegionAxisCount
    if current > target_axis_count:
        raise PipelineError("VarStore has more axes than the merged font")
    for region in store.VarRegionList.Region:
        while len(region.VarRegionAxis) < target_axis_count:
            axis = ot.VarRegionAxis()
            axis.StartCoord = 0.0
            axis.PeakCoord = 0.0
            axis.EndCoord = 0.0
            region.VarRegionAxis.append(axis)
    store.VarRegionList.RegionAxisCount = target_axis_count


def _merge_gdef_varstore(
    candidate: TTFont,
    italic_layout: TTFont,
    alternate_targets: set[str],
) -> int:
    candidate_gdef = candidate["GDEF"].table
    italic_gdef = italic_layout["GDEF"].table

    base_classes = candidate_gdef.GlyphClassDef.classDefs
    italic_classes = italic_gdef.GlyphClassDef.classDefs
    for name, glyph_class in italic_classes.items():
        current = base_classes.get(name)
        if name in alternate_targets:
            base_classes[name] = glyph_class
        elif current is None:
            raise PipelineError(
                f"GDEF class differs on shared glyph {name!r}; cannot preserve Roman endpoint"
            )
        elif current != glyph_class:
            raise PipelineError(
                f"GDEF class conflict on shared glyph {name!r}: "
                f"Roman={current}, Italic={glyph_class}"
            )

    roman_store = candidate_gdef.VarStore
    italic_store = italic_gdef.VarStore
    if roman_store is None or italic_store is None:
        raise PipelineError("Both source GDEF tables must carry a VarStore")
    target_axis_count = len(candidate["fvar"].axes)
    _extend_varstore_axis(roman_store, target_axis_count)
    _extend_varstore_axis(italic_store, target_axis_count)
    if roman_store.VarRegionList.RegionAxisCount != italic_store.VarRegionList.RegionAxisCount:
        raise PipelineError("Roman and Italic GDEF VarStore axis counts differ")

    region_offset = len(roman_store.VarRegionList.Region)
    data_offset = len(roman_store.VarData)
    roman_store.VarRegionList.Region.extend(copy.deepcopy(italic_store.VarRegionList.Region))
    roman_store.VarRegionList.RegionCount = len(roman_store.VarRegionList.Region)
    for source_data in italic_store.VarData:
        var_data = copy.deepcopy(source_data)
        var_data.VarRegionIndex = [index + region_offset for index in var_data.VarRegionIndex]
        roman_store.VarData.append(var_data)
    roman_store.VarDataCount = len(roman_store.VarData)
    return data_offset


def _remap_italic_gpos_varidxes(italic_layout: TTFont, var_data_offset: int) -> None:
    if "GPOS" not in italic_layout or var_data_offset == 0:
        return
    table = italic_layout["GPOS"].table
    used: set[int] = set()
    table.collect_device_varidxes(used)
    mapping = {
        varidx: (((varidx >> 16) + var_data_offset) << 16) | (varidx & 0xFFFF) for varidx in used
    }
    if mapping:
        table.remap_device_varidxes(mapping)


def _append_lookups(base_table, italic_table) -> int:
    offset = len(base_table.LookupList.Lookup)
    base_table.LookupList.Lookup.extend(copy.deepcopy(italic_table.LookupList.Lookup))
    base_table.LookupList.LookupCount = len(base_table.LookupList.Lookup)
    return offset


def _layout_contexts(table) -> dict[tuple[str, str], ot.LangSys]:
    contexts: dict[tuple[str, str], ot.LangSys] = {}
    for script_record in table.ScriptList.ScriptRecord:
        script = script_record.Script
        if script.DefaultLangSys is not None:
            contexts[(script_record.ScriptTag, "dflt")] = script.DefaultLangSys
        for language_record in script.LangSysRecord:
            contexts[(script_record.ScriptTag, language_record.LangSysTag)] = (
                language_record.LangSys
            )
    return contexts


def _feature_replacement_plan(base_table, italic_table):
    base_features = base_table.FeatureList.FeatureRecord
    italic_features = italic_table.FeatureList.FeatureRecord
    base_contexts = _layout_contexts(base_table)
    italic_contexts = _layout_contexts(italic_table)
    usage: dict[int, list[tuple[tuple[str, str], int | None]]] = defaultdict(list)

    for context, base_langsys in base_contexts.items():
        italic_langsys = italic_contexts.get(context)
        if italic_langsys is None:
            raise PipelineError(f"Italic layout is missing context {context}")
        italic_by_tag: dict[str, list[int]] = defaultdict(list)
        for feature_index in italic_langsys.FeatureIndex:
            record = italic_features[feature_index]
            italic_by_tag[record.FeatureTag].append(feature_index)

        base_tags = {
            base_features[index].FeatureTag
            for index in base_langsys.FeatureIndex
            if base_features[index].FeatureTag != "rvrn"
        }
        italic_tags = {italic_features[index].FeatureTag for index in italic_langsys.FeatureIndex}
        extra = italic_tags - base_tags
        if extra:
            raise PipelineError(
                f"Italic layout context {context} has features absent from Roman: {sorted(extra)}"
            )

        for feature_index in list(base_langsys.FeatureIndex):
            tag = base_features[feature_index].FeatureTag
            if tag == "rvrn":
                continue
            matches = italic_by_tag.get(tag, [])
            if len(matches) > 1:
                raise PipelineError(
                    f"Italic layout context {context} has duplicate feature {tag!r}"
                )
            usage[feature_index].append((context, matches[0] if matches else None))

    replacement: dict[int, int | None] = {}
    for feature_index, entries in sorted(usage.items()):
        groups: dict[int | None, list[tuple[str, str]]] = defaultdict(list)
        for context, italic_index in entries:
            groups[italic_index].append(context)
        first = True
        for italic_index, contexts in sorted(
            groups.items(), key=lambda item: (item[0] is None, item[0] or -1)
        ):
            target_index = feature_index
            if not first:
                target_index = len(base_features)
                base_features.append(copy.deepcopy(base_features[feature_index]))
                base_table.FeatureList.FeatureCount = len(base_features)
                for context in contexts:
                    langsys = base_contexts[context]
                    langsys.FeatureIndex = [
                        target_index if index == feature_index else index
                        for index in langsys.FeatureIndex
                    ]
                    langsys.FeatureCount = len(langsys.FeatureIndex)
            replacement[target_index] = italic_index
            first = False

    return replacement


def _sort_features_with_replacement(base_table, replacement):
    records = list(base_table.FeatureList.FeatureRecord)
    decorated = sorted((record.FeatureTag, index, record) for index, record in enumerate(records))
    remap = {old_index: new_index for new_index, (_, old_index, _) in enumerate(decorated)}
    base_table.FeatureList.FeatureRecord = [record for _, _, record in decorated]
    base_table.FeatureList.FeatureCount = len(decorated)
    remapFeatures(base_table, remap)
    return {remap[index]: value for index, value in replacement.items()}


def _install_feature_replacements(
    font: TTFont,
    table_tag: str,
    italic_table,
    replacement: dict[int, int | None],
    lookup_offset: int,
    threshold: float,
) -> None:
    table = font[table_tag].table
    replacement = _sort_features_with_replacement(table, replacement)
    records = []
    for feature_index, italic_index in sorted(replacement.items()):
        if italic_index is None:
            lookups: list[int] = []
        else:
            italic_feature = italic_table.FeatureList.FeatureRecord[italic_index].Feature
            if italic_feature.FeatureParams is not None:
                raise PipelineError(f"{table_tag} feature parameters are not yet supported")
            lookups = [lookup_offset + index for index in italic_feature.LookupListIndex]
        records.append(buildFeatureTableSubstitutionRecord(feature_index, lookups))

    axis_index = next(
        index for index, axis in enumerate(font["fvar"].axes) if axis.axisTag == "ital"
    )
    condition_table = [buildConditionTable(axis_index, threshold, 1.0)]
    if table.Version < 0x00010001:
        table.Version = 0x00010001

    existing = None
    feature_variations = getattr(table, "FeatureVariations", None)
    if feature_variations is not None:
        existing = findFeatureVariationRecord(feature_variations, condition_table)
    if existing is not None:
        existing.FeatureTableSubstitution.SubstitutionRecord.extend(records)
        existing.FeatureTableSubstitution.SubstitutionCount = len(
            existing.FeatureTableSubstitution.SubstitutionRecord
        )
        return

    if table_tag == "GSUB" and feature_variations is not None:
        # rvrn may already partition [min substitution threshold, 1] into
        # disjoint first-match records. A later broad layout record would never
        # win inside those ranges, so copy the layout replacements into every
        # earlier rvrn record that can match at/above the layout threshold.
        for variation_record in feature_variations.FeatureVariationRecord:
            condition_set = variation_record.ConditionSet
            conditions = condition_set.ConditionTable if condition_set is not None else []
            if len(conditions) != 1 or conditions[0].AxisIndex != axis_index:
                continue
            condition = conditions[0]
            if condition.FilterRangeMaxValue < threshold:
                continue
            if condition.FilterRangeMinValue < threshold:
                raise PipelineError(
                    "GSUB FeatureVariations cross the layout threshold; "
                    "choose a layout threshold at or below every rvrn threshold"
                )
            substitution = variation_record.FeatureTableSubstitution
            substitution.SubstitutionRecord.extend(copy.deepcopy(records))
            substitution.SubstitutionCount = len(substitution.SubstitutionRecord)

    feature_record = buildFeatureVariationRecord(condition_table, records)
    if getattr(table, "FeatureVariations", None) is None:
        table.FeatureVariations = ot.FeatureVariations()
        table.FeatureVariations.Version = 0x00010000
        table.FeatureVariations.FeatureVariationRecord = []
    table.FeatureVariations.FeatureVariationRecord.append(feature_record)
    table.FeatureVariations.FeatureVariationCount = len(
        table.FeatureVariations.FeatureVariationRecord
    )


def _merge_layout(
    candidate: TTFont,
    italic_path: Path,
    alternates: dict[str, str],
    threshold: float,
) -> None:
    italic_layout = _load_renamed_layout_font(italic_path, alternates)
    var_data_offset = _merge_gdef_varstore(candidate, italic_layout, set(alternates.values()))
    _remap_italic_gpos_varidxes(italic_layout, var_data_offset)

    for table_tag in ("GSUB", "GPOS"):
        if table_tag not in candidate or table_tag not in italic_layout:
            raise PipelineError(f"Both sources need {table_tag} for exact merge")
        base_table = candidate[table_tag].table
        italic_table = italic_layout[table_tag].table
        replacement = _feature_replacement_plan(base_table, italic_table)
        lookup_offset = _append_lookups(base_table, italic_table)
        _install_feature_replacements(
            candidate,
            table_tag,
            italic_table,
            replacement,
            lookup_offset,
            threshold,
        )


def merge_variable_fonts(
    roman_path: Path,
    italic_path: Path,
    output_path: Path,
    axis: ConfigAxis,
    strategies: dict[str, tuple[str, float | None]],
    *,
    default_strategy: str = "auto",
    layout_threshold: float = 0.5,
) -> ItalMergeReport:
    roman = TTFont(str(roman_path))
    italic = TTFont(str(italic_path))
    if _axis_signature(roman) != _axis_signature(italic):
        raise PipelineError("Roman and Italic source axes do not match")
    if roman["head"].unitsPerEm != italic["head"].unitsPerEm:
        raise PipelineError("Roman and Italic unitsPerEm do not match")
    if set(roman.getBestCmap()) != set(italic.getBestCmap()):
        raise PipelineError("Roman and Italic encoded character coverage differs")
    if "HVAR" not in roman or "HVAR" not in italic:
        raise PipelineError("Both source fonts need HVAR for exact advance merging")

    # Decompile variation tables while fvar still has only the source axes.
    _ = roman["gvar"].variations
    _ = italic["gvar"].variations
    _ = roman["HVAR"].table
    _ = italic["HVAR"].table
    candidate = copy.deepcopy(roman)
    interpolation_italic = copy.deepcopy(italic)
    _ = candidate["gvar"].variations
    _ = candidate["HVAR"].table
    _ = interpolation_italic["gvar"].variations

    roman_names = set(roman.getGlyphOrder())
    italic_names = set(italic.getGlyphOrder())
    union_order = roman.getGlyphOrder() + [
        name for name in italic.getGlyphOrder() if name not in roman_names
    ]
    report = ItalMergeReport(
        output=output_path,
        source_hashes={
            "roman": _sha256(roman_path),
            "italic": _sha256(italic_path),
        },
    )
    resolved: dict[str, tuple[str, float | None]] = {}
    substitutes: list[str] = []
    for glyph_name in union_order:
        strategy, threshold = strategies.get(glyph_name, (default_strategy, None))
        if strategy == "auto":
            strategy = (
                "interpolate"
                if glyph_name in roman_names
                and glyph_name in italic_names
                and _point_program_compatible(roman, italic, glyph_name)
                else "substitute"
            )
        if glyph_name not in roman_names or glyph_name not in italic_names:
            strategy = "review"
        resolved[glyph_name] = (strategy, threshold)
        if strategy in {"interpolate", "cyclic_interpolate"}:
            lift_source = italic
            cyclic_shifts = None
            if strategy == "cyclic_interpolate":
                cyclic_shifts = _cyclic_shift_plan(roman, interpolation_italic, glyph_name)
                if cyclic_shifts is None:
                    raise PipelineError(f"{glyph_name}: no confident exact cyclic phase repair")
                _apply_cyclic_rotation(interpolation_italic, glyph_name, cyclic_shifts)
                if not _point_program_compatible(roman, interpolation_italic, glyph_name):
                    raise PipelineError(
                        f"{glyph_name}: cyclic phase repair did not align point programs"
                    )
                lift_source = interpolation_italic
            try:
                _lift_glyph_surface(candidate, roman, lift_source, glyph_name)
            except PipelineError as exc:
                if "IUP field is fractional" not in str(exc):
                    raise
                strategy = "substitute"
                resolved[glyph_name] = (strategy, threshold)
                substitutes.append(glyph_name)
                report.substituted.append(glyph_name)
                report.fallback_substituted[glyph_name] = str(exc)
            else:
                resolved[glyph_name] = ("interpolate", threshold)
                report.interpolated.append(glyph_name)
                if cyclic_shifts is not None:
                    report.cyclic_interpolated[glyph_name] = cyclic_shifts
        elif strategy == "substitute":
            substitutes.append(glyph_name)
            report.substituted.append(glyph_name)
        elif strategy == "review":
            report.review.append(glyph_name)
        else:
            raise PipelineError(f"{glyph_name}: unknown ital strategy {strategy!r}")

    alternates = _copy_italic_alternates(candidate, italic, substitutes)
    report.alternates.update(alternates)
    _add_ital_axis(candidate, italic, axis)
    _add_ital_stat(candidate, axis)
    _rebuild_hvar_preserving_sources(candidate, roman, italic, resolved, alternates)
    conditional_substitutions = _group_substitutions(
        resolved,
        alternates,
        default_threshold=layout_threshold,
    )
    if conditional_substitutions:
        addFeatureVariations(candidate, conditional_substitutions, featureTag="rvrn")
    _merge_layout(candidate, italic_path, alternates, layout_threshold)
    reorderGlyphs(candidate, candidate.getGlyphOrder())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate.save(output_path)
    return report


def merge_ital(config: ProjectConfig) -> ItalMergeReport:
    merge = config.ital_merge
    if merge is None:
        raise PipelineError("config has no italMerge block")
    roman_style = config.styles[merge.roman_style]
    italic_style = config.styles[merge.italic_style]
    if not roman_style.output.is_file():
        raise PipelineError(f"Roman VF does not exist: {roman_style.output}")
    if not italic_style.output.is_file():
        raise PipelineError(f"Italic VF does not exist: {italic_style.output}")
    output = merge.output
    if output is None:
        output = Path(config.output.dir)
        if not output.is_absolute():
            output = config.repo_root / output
        output = output / f"{config.id}-ital-merged.ttf"

    strategies = {
        name: (strategy.strategy, strategy.threshold)
        for name, strategy in config.glyphs.ital_strategies.items()
    }
    return merge_variable_fonts(
        roman_style.output,
        italic_style.output,
        output,
        merge.axis,
        strategies,
        default_strategy=config.glyphs.ital_default_strategy,
        layout_threshold=merge.layout_threshold,
    )
