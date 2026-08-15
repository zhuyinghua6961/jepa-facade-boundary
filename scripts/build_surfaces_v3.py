#!/usr/bin/env python3
"""Fit surface-v3 physical boundary lines from multi-view RGB-D clicks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from boundary_sweep.geometry import camera_to_world, fit_plane, pixel_depth_to_camera_point, ray_plane_intersection, pixel_to_camera_ray, world_to_pixel


def depth_at(path, pixel):
    depth = np.load(path)
    x, y = np.rint(pixel).astype(int)
    if not (0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]):
        return None
    patch = depth[max(0, y-1):min(depth.shape[0], y+2), max(0, x-1):min(depth.shape[1], x+2)]
    valid = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 1000.0)]
    return float(np.median(valid)) if valid.size else None


def click_world(view, point_name):
    pixel = np.asarray(view["raw_clicks_2d"][point_name], dtype=float)
    # Always use the inner facade support plane for boundary clicks. A raw
    # corner pixel can belong to the adjacent face/background at a jump.
    plane_origin, plane_normal, _ = fit_plane(view["plane_support_points"])
    transform = type("T", (), {
        "location": type("L", (), view["camera_transform"]["location"])(),
        "rotation": type("R", (), view["camera_transform"]["rotation"])(),
    })()
    camera_origin = camera_to_world([0.0, 0.0, 0.0], transform)
    ray_camera = pixel_to_camera_ray(pixel, np.asarray(view["K"], dtype=float))
    ray_world = camera_to_world(ray_camera, transform) - camera_origin
    point_world = ray_plane_intersection(camera_origin, ray_world, plane_origin, plane_normal)
    if point_world is None:
        raise ValueError(f"no plane intersection at {view['view_id']} {point_name}")
    point_world = np.asarray(point_world, dtype=float)
    depth = float(world_to_pixel(point_world, transform, np.asarray(view["K"], dtype=float))[2])
    return point_world, depth


def make_surface(annotation):
    views = annotation["views"]
    endpoint_world = {name: [] for name in ("TL", "TR", "BL", "BR")}
    click_depths = {}
    for view in views:
        click_depths[view["view_id"]] = {}
        for name in endpoint_world:
            point, depth = click_world(view, name)
            endpoint_world[name].append(point)
            click_depths[view["view_id"]][name] = depth
    fitted = {name: np.mean(points, axis=0) for name, points in endpoint_world.items()}
    support = [point for view in views for point in view["plane_support_points"]]
    plane_origin, plane_normal, plane_residual = fit_plane(support)
    horizontal = fitted["BR"] - fitted["BL"]
    vertical = fitted["TL"] - fitted["BL"]
    horizontal /= np.linalg.norm(horizontal)
    vertical /= np.linalg.norm(vertical)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    normal_hint = np.asarray(annotation["normal_hint_for_view_setup"], dtype=float)
    if np.dot(plane_normal, normal_hint) < 0:
        plane_normal = -plane_normal
    reprojection_errors = {}
    error_values = []
    for view in views:
        transform = type("T", (), {
            "location": type("L", (), view["camera_transform"]["location"])(),
            "rotation": type("R", (), view["camera_transform"]["rotation"])(),
        })()
        K = np.asarray(view["K"], dtype=float)
        errors = {}
        for name in endpoint_world:
            expected = np.asarray(annotation["raw_clicks_2d"][view["view_id"]][name], dtype=float)
            actual = world_to_pixel(fitted[name], transform, K)[:2]
            err = float(np.linalg.norm(actual - expected))
            errors[name] = err; error_values.append(err)
        reprojection_errors[view["view_id"]] = errors
    boundary = {
        "LEFT": {"start": fitted["TL"].tolist(), "end": fitted["BL"].tolist()},
        "RIGHT": {"start": fitted["TR"].tolist(), "end": fitted["BR"].tolist()},
        "TOP": {"start": fitted["TL"].tolist(), "end": fitted["TR"].tolist()},
        "BOTTOM": {"start": fitted["BL"].tolist(), "end": fitted["BR"].tolist()},
    }
    return {
        "schema": "geo0.5r2.surface.v3", "surface_id": annotation["surface_id"], "map": annotation["map"],
        "bbox_id": annotation["bbox_id"], "bbox_used_for": "candidate_filter_only",
        "raw_clicks_2d": annotation["raw_clicks_2d"],
        "annotation_camera_transforms": annotation["annotation_camera_transforms"],
        "fitted_boundary_world_line": boundary,
        "physical_boundary": boundary,
        "plane_support_points": [np.asarray(x).tolist() for x in support],
        "plane_origin": plane_origin.tolist(), "plane_normal": plane_normal.tolist(),
        "horizontal_axis": horizontal.tolist(), "vertical_axis": vertical.tolist(),
        "width_m": float(np.linalg.norm(fitted["TR"] - fitted["TL"])),
        "height_m": float(np.linalg.norm(fitted["TL"] - fitted["BL"])),
        "reprojection_errors_px": {"per_view": reprojection_errors, "median": float(np.median(error_values)), "max": float(np.max(error_values))},
        "click_depths_m": click_depths,
        "annotation_method": annotation["calibration_method"],
        "manual_confirmation_status": annotation["manual_confirmation_status"],
        "manual_confirmation_required": annotation["manual_confirmation_required"],
        "source": {"bbox_id": annotation["bbox_id"], "bbox_used_for": "candidate filtering and camera placement only",
                   "boundary_truth": "multi-view RGB-D back-projected operator clicks; no hard-coded rectangle"},
    }


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--annotations", default="data/boundary_annotations"); ap.add_argument("--output", default="data/surfaces_v3")
    args = ap.parse_args(); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(Path(args.annotations).glob("*/annotation.json")):
        annotation = json.loads(path.read_text()); surface = make_surface(annotation)
        destination = out / f"{surface['surface_id']}.json"; destination.write_text(json.dumps(surface, indent=2) + "\n")
        rows.append({"surface_id": surface["surface_id"], "reprojection_errors_px": surface["reprojection_errors_px"]})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
