"""Coordinate and projective geometry used by GEO-0.

CARLA uses a left-handed UE convention (X forward, Y right, Z up).  The
camera model in this module uses the conventional CV frame (x right, y down,
z forward).  The conversion is explicit so depth semantics cannot be hidden
inside a projection helper.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


def intrinsics_from_fov(width: int, height: int, horizontal_fov_deg: float) -> np.ndarray:
    """Return a pinhole K for CARLA's horizontal field of view."""
    if width <= 0 or height <= 0 or not (0.0 < horizontal_fov_deg < 180.0):
        raise ValueError("invalid image size or horizontal FOV")
    fx = width / (2.0 * math.tan(math.radians(horizontal_fov_deg) / 2.0))
    return np.array([[fx, 0.0, (width - 1.0) / 2.0],
                     [0.0, fx, (height - 1.0) / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _rotation_matrix(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """CARLA/UE rotation matrix, with angles in degrees."""
    p, y, r = np.radians([pitch, yaw, roll])
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    cr, sr = math.cos(r), math.sin(r)
    return np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
        [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
        [sp, -cp * sr, cp * cr],
    ], dtype=np.float64)


def transform_matrix(transform) -> np.ndarray:
    """Return a 4x4 CARLA UE-local-to-world matrix.

    ``transform`` may be a carla.Transform or a small duck-typed object with
    ``location`` and ``rotation`` fields.  A 4x4 array is accepted unchanged.
    """
    arr = np.asarray(transform)
    if arr.shape == (4, 4):
        return arr.astype(np.float64, copy=True)
    loc, rot = transform.location, transform.rotation
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = _rotation_matrix(float(rot.pitch), float(rot.yaw), float(rot.roll))
    out[:3, 3] = [float(loc.x), float(loc.y), float(loc.z)]
    return out


def _as_points(points: Sequence[Sequence[float]]) -> Tuple[np.ndarray, bool]:
    arr = np.asarray(points, dtype=np.float64)
    one = arr.ndim == 1
    if one:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("points must have shape (3,) or (N,3)")
    return arr, one


def ue_to_cv(points_ue: Sequence[Sequence[float]]) -> np.ndarray:
    """Convert UE camera-local [X,Y,Z] to CV [x,y,z]."""
    arr, one = _as_points(points_ue)
    out = arr[:, [1, 2, 0]].copy()
    out[:, 1] *= -1.0
    return out[0] if one else out


def cv_to_ue(points_cv: Sequence[Sequence[float]]) -> np.ndarray:
    """Convert CV camera-local [x,y,z] to UE camera-local [X,Y,Z]."""
    arr, one = _as_points(points_cv)
    out = np.column_stack([arr[:, 2], arr[:, 0], -arr[:, 1]])
    return out[0] if one else out


def camera_to_world(points_camera: Sequence[Sequence[float]], camera_transform) -> np.ndarray:
    arr, one = _as_points(points_camera)
    T = transform_matrix(camera_transform)
    ue = cv_to_ue(arr)
    world = (T[:3, :3] @ ue.T).T + T[:3, 3]
    return world[0] if one else world


def world_to_camera(points_world: Sequence[Sequence[float]], camera_transform) -> np.ndarray:
    arr, one = _as_points(points_world)
    T = transform_matrix(camera_transform)
    ue = (T[:3, :3].T @ (arr - T[:3, 3]).T).T
    cv = ue_to_cv(ue)
    return cv[0] if one else cv


def pixel_to_camera_ray(pixel: Sequence[float], K: np.ndarray) -> np.ndarray:
    uv1 = np.array([float(pixel[0]), float(pixel[1]), 1.0], dtype=np.float64)
    ray = np.linalg.solve(np.asarray(K, dtype=np.float64), uv1)
    return ray / np.linalg.norm(ray)


def pixel_depth_to_camera_point(pixel: Sequence[float], depth: float, K: np.ndarray,
                                mode: str = "z-depth") -> np.ndarray:
    if not np.isfinite(depth) or depth <= 0.0:
        raise ValueError("depth must be a positive finite value")
    ray = pixel_to_camera_ray(pixel, K)
    if mode in ("ray-range", "range"):
        return ray * float(depth)
    if mode in ("z-depth", "z"):
        xy = np.linalg.solve(np.asarray(K, dtype=np.float64),
                             np.array([float(pixel[0]), float(pixel[1]), 1.0]))
        return xy * float(depth)
    raise ValueError("mode must be 'z-depth' or 'ray-range'")


def world_to_pixel(points_world: Sequence[Sequence[float]], camera_transform,
                   K: np.ndarray, behind_value: float = np.nan) -> np.ndarray:
    """Project world points; result is [u,v,z_cv], NaN for points behind camera."""
    cam, one = _as_points(world_to_camera(points_world, camera_transform))
    K = np.asarray(K, dtype=np.float64)
    out = np.full((len(cam), 3), behind_value, dtype=np.float64)
    good = cam[:, 2] > 1e-9
    homog = (K @ cam[good].T).T
    out[good, 0] = homog[:, 0] / homog[:, 2]
    out[good, 1] = homog[:, 1] / homog[:, 2]
    out[good, 2] = cam[good, 2]
    return out[0] if one else out


def ray_plane_intersection(ray_origin: Sequence[float], ray_direction: Sequence[float],
                           plane_point: Sequence[float], plane_normal: Sequence[float],
                           epsilon: float = 1e-9) -> Optional[np.ndarray]:
    origin = np.asarray(ray_origin, dtype=np.float64)
    direction = np.asarray(ray_direction, dtype=np.float64)
    point = np.asarray(plane_point, dtype=np.float64)
    normal = np.asarray(plane_normal, dtype=np.float64)
    denom = float(np.dot(direction, normal))
    if abs(denom) < epsilon:
        return None
    t = float(np.dot(point - origin, normal) / denom)
    if t < -epsilon:
        return None
    return origin + max(t, 0.0) * direction


def world_point_to_surface_coordinate(world_point: Sequence[float], origin: Sequence[float],
                                      horizontal_axis: Sequence[float],
                                      vertical_axis: Sequence[float],
                                      normal: Optional[Sequence[float]] = None) -> np.ndarray:
    delta = np.asarray(world_point, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    u = np.asarray(horizontal_axis, dtype=np.float64)
    v = np.asarray(vertical_axis, dtype=np.float64)
    return np.array([np.dot(delta, u), np.dot(delta, v)], dtype=np.float64)


def surface_coordinate_to_world_point(surface_coordinate: Sequence[float], origin: Sequence[float],
                                      horizontal_axis: Sequence[float],
                                      vertical_axis: Sequence[float]) -> np.ndarray:
    uv = np.asarray(surface_coordinate, dtype=np.float64)
    return np.asarray(origin, dtype=np.float64) + uv[0] * np.asarray(horizontal_axis) + uv[1] * np.asarray(vertical_axis)


def fit_plane(points: Iterable[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray, float]:
    pts = np.asarray(list(points), dtype=np.float64)
    if pts.shape[0] < 3 or pts.shape[1] != 3:
        raise ValueError("at least three 3D points are required")
    origin = pts.mean(axis=0)
    _, values, vh = np.linalg.svd(pts - origin, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    residual = np.abs((pts - origin) @ normal)
    return origin, normal, float(np.max(residual))


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    pts = np.asarray(points, dtype=np.float64)
    return 0.5 * abs(float(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - pts[:, 1] * np.roll(pts[:, 0], -1))))

