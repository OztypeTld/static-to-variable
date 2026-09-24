"""Conservative repair of cyclic start-point correspondence.

All fixtures are small synthetic contours.  The repair may rotate an existing
closed path, but it must never move a point, force a topology change, or accept
an unsafe interpolation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import variable_gen.reconstruct_compatible as reconstruction  # noqa: E402
from variable_gen.reconstruct_compatible import (  # noqa: E402
    _contour_node_points,
    _repair_cyclic_phase_correspondence,
    _rotate_contour,
)


def _polygon(points):
    contour = [("moveTo", [points[0]])]
    contour.extend(("lineTo", [point]) for point in points[1:])
    contour.append(("lineTo", [points[0]]))
    contour.append(("closePath", []))
    return contour


def _curved_polygon(points, bulges):
    contour = [("moveTo", [points[0]])]
    for index, start in enumerate(points):
        end = points[(index + 1) % len(points)]
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = (dx * dx + dy * dy) ** 0.5 or 1.0
        nx, ny = -dy / length, dx / length
        bulge = bulges[index]
        contour.append(
            (
                "curveTo",
                [
                    (start[0] + dx / 3 + nx * bulge, start[1] + dy / 3 + ny * bulge),
                    (start[0] + 2 * dx / 3 + nx * bulge, start[1] + 2 * dy / 3 + ny * bulge),
                    end,
                ],
            )
        )
    contour.append(("closePath", []))
    return contour


def _point_set(contour):
    return {(round(x, 7), round(y, 7)) for _op, args in contour for x, y in args}


def test_wrong_cyclic_phase_is_repaired_without_moving_points() -> None:
    light_points = [(0.0, 0.0), (330.0, 20.0), (285.0, 215.0), (95.0, 270.0), (-35.0, 130.0)]
    bold_points = [(0.0, 0.0), (350.0, 5.0), (310.0, 235.0), (80.0, 300.0), (-55.0, 145.0)]
    light = _polygon(light_points)
    bold = _rotate_contour(_polygon(bold_points), 2)
    assert bold is not None
    outlines = {400.0: [light], 700.0: [bold]}

    repaired = _repair_cyclic_phase_correspondence(outlines, 400.0, outlines)

    assert repaired is not outlines
    assert _contour_node_points(repaired[700.0][0])[0] == bold_points[0]
    assert _point_set(repaired[700.0][0]) == _point_set(bold)


def test_repeated_curve_structure_uses_geometry_to_choose_phase() -> None:
    # Every segment is a cubic, and three pairs have deliberately similar
    # lengths.  Segment structure alone therefore accepts every phase; the
    # unequal handles and modest node asymmetry provide the geometric evidence.
    nodes_a = [
        (0.0, 0.0),
        (160.0, -5.0),
        (310.0, 25.0),
        (330.0, 190.0),
        (165.0, 220.0),
        (-15.0, 175.0),
    ]
    nodes_b = [
        (0.0, 0.0),
        (170.0, -8.0),
        (325.0, 18.0),
        (350.0, 205.0),
        (160.0, 245.0),
        (-25.0, 185.0),
    ]
    first = _curved_polygon(nodes_a, [8.0, 28.0, 10.0, 42.0, 13.0, 31.0])
    second = _rotate_contour(_curved_polygon(nodes_b, [10.0, 32.0, 12.0, 48.0, 15.0, 35.0]), 3)
    assert second is not None
    outlines = {300.0: [first], 600.0: [second]}

    repaired = _repair_cyclic_phase_correspondence(outlines, 300.0, outlines)

    assert repaired is not outlines
    assert _contour_node_points(repaired[600.0][0])[0] == nodes_b[0]
    assert _point_set(repaired[600.0][0]) == _point_set(second)


def test_existing_correct_correspondence_is_left_unchanged() -> None:
    first = _polygon([(0.0, 0.0), (300.0, 10.0), (260.0, 210.0), (70.0, 260.0), (-20.0, 125.0)])
    second = _polygon([(0.0, 0.0), (325.0, 0.0), (290.0, 235.0), (55.0, 285.0), (-35.0, 140.0)])
    outlines = {400.0: [first], 700.0: [second]}

    repaired = _repair_cyclic_phase_correspondence(outlines, 400.0, outlines)

    assert repaired is outlines


def test_repair_that_would_self_intersect_is_rejected() -> None:
    first = _polygon(
        [
            (495.57, 356.67),
            (466.55, 356.59),
            (334.97, 388.83),
            (174.14, 412.12),
            (396.76, 238.90),
        ]
    )
    second_base = _polygon(
        [
            (285.08, 415.68),
            (112.10, 247.48),
            (123.19, 114.23),
            (266.31, 232.54),
            (317.90, 118.00),
        ]
    )
    second = _rotate_contour(second_base, 2)
    assert second is not None
    outlines = {400.0: [first], 700.0: [second]}

    proposed = reconstruction._confident_cyclic_phase(second, first)
    assert proposed is not second
    assert reconstruction._has_interpolated_self_intersection({400.0: [first], 700.0: [proposed]})

    repaired = _repair_cyclic_phase_correspondence(outlines, 400.0, outlines)

    assert repaired is outlines


def test_differing_donor_contour_topology_is_not_forced() -> None:
    first = _polygon([(0.0, 0.0), (300.0, 5.0), (260.0, 220.0), (60.0, 270.0), (-30.0, 125.0)])
    second_base = _polygon(
        [(0.0, 0.0), (325.0, 0.0), (290.0, 240.0), (45.0, 295.0), (-45.0, 140.0)]
    )
    second = _rotate_contour(second_base, 2)
    assert second is not None
    compatible = {400.0: [first], 700.0: [second]}
    topology_changed_donors = {
        400.0: [first],
        700.0: [second_base, _polygon([(500.0, 0.0), (560.0, 0.0), (530.0, 60.0)])],
    }

    repaired = _repair_cyclic_phase_correspondence(compatible, 400.0, topology_changed_donors)

    assert repaired is compatible
