#!/usr/bin/env python3
"""Fit GEO-0.6 facade planes and terminal edges from CARLA raycasts.

Building bboxes are used only to bound the search.  Every support point and
every physical boundary line in the output comes from collision raycasts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from boundary_sweep.carla_utils import discover_carla_root, import_carla, transform_to_dict
from boundary_sweep.geometry import fit_plane


SELECTED = {48389: "surface_omega", 48418: "surface_sigma"}


def _point(corners, u, v):
    tl, tr, bl, br = corners
    return (tl * (1.0 - u) + tr * u) * (1.0 - v) + (bl * (1.0 - u) + br * u) * v


def _ray_hit(world, carla, origin, target, center, normal):
    hits = world.cast_ray(carla.Location(*origin.tolist()), carla.Location(*target.tolist()))
    if not hits:
        return None
    hit = hits[0]
    for hit in hits:
        if str(hit.label) == "Buildings":
            return np.array([hit.location.x, hit.location.y, hit.location.z], dtype=float)
    return None


def _fit_candidate(world, carla, candidate, views=3):
    corners = np.asarray(candidate["corners_world_TL_TR_BL_BR"], dtype=float)
    center = np.asarray(candidate["center"], dtype=float)
    normal = np.asarray(candidate["normal"], dtype=float)
    normal /= np.linalg.norm(normal)
    h = np.asarray(candidate["horizontal_axis"], dtype=float)
    h /= np.linalg.norm(h)
    v = np.asarray(candidate["vertical_axis"], dtype=float)
    v /= np.linalg.norm(v)
    distance = max(30.0, 0.75 * float(candidate["width_m"]))
    support = []
    occupancy = []
    for vv in np.linspace(0.08, 0.92, 11):
        for uu in np.linspace(0.08, 0.92, 17):
            target = _point(corners, uu, vv)
            hit = _ray_hit(world, carla, center + normal * distance, target, center, normal)
            occupancy.append(hit is not None)
            if hit is not None:
                support.append(hit)
    if len(support) < 40:
        raise RuntimeError(f"insufficient building ray hits for bbox {candidate['bbox_id']}: {len(support)}")
    plane_origin, plane_normal, residual = fit_plane(support)
    if np.dot(plane_normal, normal) < 0:
        plane_normal = -plane_normal
    h = h - plane_normal * np.dot(h, plane_normal)
    h /= np.linalg.norm(h)
    v = np.cross(plane_normal, h)
    v /= np.linalg.norm(v)
    if np.dot(v, np.asarray(candidate["vertical_axis"], dtype=float)) < 0:
        v = -v

    edge_points = {"LEFT": [], "RIGHT": [], "TOP": [], "BOTTOM": []}
    # A binary search from the central hit toward each candidate bound refines
    # the terminal location without treating the bbox edge as ground truth.
    for vv in np.linspace(0.08, 0.92, 9):
        for side, inner, outer in (("LEFT", 0.5, 0.0), ("RIGHT", 0.5, 1.0)):
            inner_hit = _ray_hit(world, carla, center + normal * distance, _point(corners, inner, vv), center, normal)
            outer_hit = _ray_hit(world, carla, center + normal * distance, _point(corners, outer, vv), center, normal)
            if inner_hit is None:
                continue
            lo, hi = (outer, inner) if side == "LEFT" else (inner, outer)
            if outer_hit is not None:
                edge_points[side].append(outer_hit)
                continue
            for _ in range(9):
                mid = (lo + hi) / 2.0
                hit = _ray_hit(world, carla, center + normal * distance, _point(corners, mid, vv), center, normal)
                if hit is not None:
                    if side == "LEFT": hi = mid
                    else: lo = mid
                    edge_points[side].append(hit)
                elif side == "LEFT":
                    lo = mid
                else:
                    hi = mid
        for side, inner, outer in (("BOTTOM", 0.5, 0.0), ("TOP", 0.5, 1.0)):
            inner_hit = _ray_hit(world, carla, center + normal * distance, _point(corners, 0.5, inner), center, normal)
            outer_hit = _ray_hit(world, carla, center + normal * distance, _point(corners, 0.5, outer), center, normal)
            if inner_hit is None:
                continue
            lo, hi = (outer, inner) if side == "BOTTOM" else (inner, outer)
            if outer_hit is not None:
                edge_points[side].append(outer_hit)
                continue
            for _ in range(9):
                mid = (lo + hi) / 2.0
                hit = _ray_hit(world, carla, center + normal * distance, _point(corners, 0.5, mid), center, normal)
                if hit is not None:
                    if side == "BOTTOM": hi = mid
                    else: lo = mid
                    edge_points[side].append(hit)
                elif side == "BOTTOM":
                    lo = mid
                else:
                    hi = mid
    if any(len(points) < 4 for points in edge_points.values()):
        raise RuntimeError(f"incomplete raycast terminal edges for bbox {candidate['bbox_id']}: { {k: len(v) for k,v in edge_points.items()} }")
    support = np.asarray(support, dtype=float)
    coords = (support - plane_origin) @ np.column_stack([h, v])
    u_min, u_max = np.percentile(coords[:, 0], [1, 99])
    v_min, v_max = np.percentile(coords[:, 1], [1, 99])
    # Edge extrema are measured from the refined terminal ray hits.
    for name, points in edge_points.items():
        arr = (np.asarray(points) - plane_origin) @ np.column_stack([h, v])
        if name == "LEFT": u_min = min(u_min, float(np.percentile(arr[:, 0], 50)))
        elif name == "RIGHT": u_max = max(u_max, float(np.percentile(arr[:, 0], 50)))
        elif name == "BOTTOM": v_min = min(v_min, float(np.percentile(arr[:, 1], 50)))
        else: v_max = max(v_max, float(np.percentile(arr[:, 1], 50)))
    def world_uv(u, vv): return (plane_origin + h * u + v * vv).tolist()
    physical = {
        "LEFT": {"start": world_uv(u_min, v_max), "end": world_uv(u_min, v_min)},
        "RIGHT": {"start": world_uv(u_max, v_max), "end": world_uv(u_max, v_min)},
        "TOP": {"start": world_uv(u_min, v_max), "end": world_uv(u_max, v_max)},
        "BOTTOM": {"start": world_uv(u_min, v_min), "end": world_uv(u_max, v_min)},
    }
    annotation_transforms = {}
    for index, offset in enumerate(np.linspace(-0.28, 0.28, views)):
        pos = center + normal * distance + h * (offset * float(candidate["width_m"]))
        annotation_transforms[f"view_{index:02d}"] = {"location": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])}, "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}}
    return {
        "schema": "geo06.surface.v1", "surface_id": SELECTED[candidate["bbox_id"]], "bbox_id": candidate["bbox_id"],
        "bbox_used_for": "candidate_search_bounds_only", "candidate_search_bounds": candidate["corners_world_TL_TR_BL_BR"],
        "plane_support_points": support.tolist(), "plane_origin": plane_origin.tolist(), "plane_normal": plane_normal.tolist(),
        "horizontal_axis": h.tolist(), "vertical_axis": v.tolist(), "physical_boundary": physical,
        "width_m": float(u_max - u_min), "height_m": float(v_max - v_min),
        "raycast_support_count": len(support), "raycast_terminal_sample_count": {k: len(vv) for k, vv in edge_points.items()},
        "plane_fit_max_residual_m": residual, "annotation_camera_transforms": annotation_transforms,
        "boundary_method": "collision_raycast_terminal_extrema_and_plane_fit", "operator_visual_review_status": "PENDING",
        "manual_confirmation_status": "pending_operator_visual_review",
        "source": {"bbox_id": candidate["bbox_id"], "bbox_used_for": "candidate filtering and ray placement only", "boundary_truth": "CARLA collision raycast hits; no bbox rectangle used as physical boundary"},
    }


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--candidates", default="results/geo06/facade_candidates.json"); parser.add_argument("--output", default="results/geo06/surfaces"); parser.add_argument("--carla-root", default=None)
    args = parser.parse_args()
    carla = import_carla(str(discover_carla_root(args.carla_root)) if discover_carla_root(args.carla_root) else args.carla_root)
    client = carla.Client("localhost", 2000); client.set_timeout(10.0); world = client.get_world()
    candidates = json.loads(Path(args.candidates).read_text())["candidates"]
    selected = []
    for bbox_id, surface_id in SELECTED.items():
        required_surface = "-X" if bbox_id == 48418 else "+Y"
        candidate = next(c for c in candidates if c["bbox_id"] == bbox_id and c["surface"] == required_surface)
        try:
            result = _fit_candidate(world, carla, candidate)
            path = Path(args.output) / f"{surface_id}.json"; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(result, indent=2) + "\n")
            selected.append({"surface_id": surface_id, "bbox_id": bbox_id, "path": str(path), "status": "SELECTED", "plane_fit_max_residual_m": result["plane_fit_max_residual_m"]})
        except Exception as exc:
            selected.append({"surface_id": surface_id, "bbox_id": bbox_id, "status": "REJECTED", "reason": str(exc)})
    print(json.dumps({"selected": selected}, indent=2))


if __name__ == "__main__": main()
