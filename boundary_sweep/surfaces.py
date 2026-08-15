"""Explicit GEO-0.5 surface model.

Plane support points are observations used to fit a plane.  They are kept
separate from physical_boundary, which describes the target facade extent.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .geometry import surface_coordinate_to_world_point


BOUNDARIES = ("LEFT", "RIGHT", "TOP", "BOTTOM")


def load_surface(path: str | Path) -> dict:
    surface = json.loads(Path(path).read_text())
    required = ("plane_support_points", "physical_boundary", "plane_origin",
                "plane_normal", "horizontal_axis", "vertical_axis")
    missing = [key for key in required if key not in surface]
    if missing:
        raise ValueError(f"surface v2 missing fields: {missing}")
    if set(surface["physical_boundary"]) != set(BOUNDARIES):
        raise ValueError("physical_boundary must contain LEFT/RIGHT/TOP/BOTTOM")
    return surface


def physical_corners(surface: Mapping) -> np.ndarray:
    boundary = surface["physical_boundary"]
    tl = np.asarray(boundary["TOP"]["start"], dtype=float)
    tr = np.asarray(boundary["TOP"]["end"], dtype=float)
    bl = np.asarray(boundary["BOTTOM"]["start"], dtype=float)
    br = np.asarray(boundary["BOTTOM"]["end"], dtype=float)
    corners = np.asarray([tl, tr, bl, br])
    if not np.isfinite(corners).all():
        raise ValueError("physical boundary contains non-finite points")
    return corners


def boundary_line(surface: Mapping, name: str) -> np.ndarray:
    item = surface["physical_boundary"][name]
    return np.asarray([item["start"], item["end"]], dtype=float)


def sample_surface_points(surface: Mapping, nx: int = 9, ny: int = 7, margin: float = 0.02):
    """Uniformly sample the physical rectangle in its 2D surface coordinates."""
    corners = physical_corners(surface)
    tl, tr, bl, br = corners
    points, coords = [], []
    xs = np.linspace(margin, 1.0 - margin, nx)
    ys = np.linspace(margin, 1.0 - margin, ny)
    for y in ys:
        for x in xs:
            top = tl * (1.0 - x) + tr * x
            bottom = bl * (1.0 - x) + br * x
            points.append(top * (1.0 - y) + bottom * y)
            width = float(surface.get("width_m", np.linalg.norm(tr - tl)))
            height = float(surface.get("height_m", np.linalg.norm(tl - bl)))
            coords.append([x * width, (1.0 - y) * height])
    return np.asarray(points), np.asarray(coords)


def surface_axes(surface: Mapping):
    h = np.asarray(surface["horizontal_axis"], dtype=float)
    v = np.asarray(surface["vertical_axis"], dtype=float)
    n = np.asarray(surface["plane_normal"], dtype=float)
    return h / np.linalg.norm(h), v / np.linalg.norm(v), n / np.linalg.norm(n)


def look_direction_to_rotation(carla, forward: Sequence[float]):
    d = np.asarray(forward, dtype=float)
    d /= np.linalg.norm(d)
    horizontal = max(math.hypot(d[0], d[1]), 1e-12)
    return carla.Rotation(pitch=float(math.degrees(math.atan2(d[2], horizontal))),
                          yaw=float(math.degrees(math.atan2(d[1], d[0]))), roll=0.0)


def normal_lock_transform(carla, surface: Mapping, distance_m: float, lateral_offset=None):
    """Return a pose whose orientation is fixed to the surface normal."""
    h, _v, n = surface_axes(surface)
    center = np.mean(physical_corners(surface), axis=0)
    position = center + n * float(distance_m)
    if lateral_offset is not None:
        position = position + h * float(lateral_offset)
    rotation = look_direction_to_rotation(carla, -n)
    return carla.Transform(carla.Location(x=float(position[0]), y=float(position[1]), z=float(position[2])), rotation)


def surface_summary(surface: Mapping) -> dict:
    corners = physical_corners(surface)
    return {
        "surface_id": surface.get("surface_id"),
        "width_m": float(surface.get("width_m", np.linalg.norm(corners[1] - corners[0]))),
        "height_m": float(surface.get("height_m", np.linalg.norm(corners[0] - corners[2]))),
        "plane_support_count": len(surface["plane_support_points"]),
        "physical_boundary": surface["physical_boundary"],
    }

