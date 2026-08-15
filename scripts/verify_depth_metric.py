#!/usr/bin/env python3
"""Validate CARLA depth semantics against an analytic opaque-plane equation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from boundary_sweep.carla_utils import discover_carla_root, import_carla
from boundary_sweep.geometry import (camera_to_world, pixel_to_camera_ray, ray_plane_intersection,
                                     world_to_camera, world_to_pixel)
from boundary_sweep.sensors import SynchronousRGBD
from boundary_sweep.surfaces import load_surface, physical_corners, surface_axes, look_direction_to_rotation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carla-root", default=None)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--surface", required=True)
    ap.add_argument("--output", default="data/depth_metric_v2.json")
    ap.add_argument("--distances", nargs="+", type=float, default=[5.0, 10.0, 20.0])
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fov", type=float, default=90.0)
    args = ap.parse_args()
    surface = load_surface(args.surface)
    root = discover_carla_root(args.carla_root)
    carla = import_carla(str(root) if root else args.carla_root)
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    corners = physical_corners(surface)
    center = corners.mean(axis=0)
    h, v, n = surface_axes(surface)
    plane_origin = np.asarray(surface["plane_origin"], dtype=float)
    K = None
    rows = []
    for distance in args.distances:
        location = center + n * float(distance)
        rotation = look_direction_to_rotation(carla, -n)
        transform = carla.Transform(carla.Location(x=float(location[0]), y=float(location[1]), z=float(location[2])), rotation)
        with SynchronousRGBD(world, carla, transform, args.width, args.height, args.fov, output_dir="data/depth_metric_capture") as rig:
            sample = rig.capture({"type": "depth_metric", "distance_m": distance})
            K = sample["K"]
            # Center plus four image-internal off-axis pixels. Intersect each
            # pixel ray with the known opaque plane before reading CARLA depth.
            offsets = [(0.0, 0.0), (0.25 * rig.width, 0.0), (-0.25 * rig.width, 0.0),
                       (0.0, 0.25 * rig.height), (0.0, -0.25 * rig.height)]
            camera_origin = camera_to_world([0.0, 0.0, 0.0], sample["camera_transform"])
            for du, dv in offsets:
                uv = np.array([(rig.width - 1.0) / 2.0 + du, (rig.height - 1.0) / 2.0 + dv], dtype=float)
                ray_camera = pixel_to_camera_ray(uv, K)
                ray_world = camera_to_world(ray_camera, sample["camera_transform"]) - camera_origin
                world_target = ray_plane_intersection(camera_origin, ray_world, plane_origin, n)
                if world_target is None:
                    continue
                uvz = world_to_pixel(world_target, sample["camera_transform"], K)
                x, y = np.rint(uv).astype(int)
                measured = float(sample["depth_m"][y, x])
                camera_point = world_to_camera(world_target, sample["camera_transform"])
                camera_z = float(camera_point[2])
                ray_range = float(np.linalg.norm(world_target - camera_to_world([0.0, 0.0, 0.0], sample["camera_transform"])))
                rows.append({"distance_m": float(distance), "pixel_offset": [float(du), float(dv)],
                             "pixel": uv.tolist(), "sensor_depth_m": measured,
                             "analytic_camera_z_m": camera_z, "analytic_ray_range_m": ray_range,
                             "abs_error_z_m": abs(measured - camera_z), "abs_error_range_m": abs(measured - ray_range),
                             "rel_error_z": abs(measured - camera_z) / max(camera_z, 1e-9),
                             "rel_error_range": abs(measured - ray_range) / max(ray_range, 1e-9)})
    z = np.asarray([row["rel_error_z"] for row in rows])
    r = np.asarray([row["rel_error_range"] for row in rows])
    result = {"surface_id": surface["surface_id"], "plane_source": "physical_boundary_plane_equation",
              "distances_m": args.distances, "sample_count": len(rows),
              "z_depth_median_abs_error_m": float(np.median([row["abs_error_z_m"] for row in rows])),
              "ray_range_median_abs_error_m": float(np.median([row["abs_error_range_m"] for row in rows])),
              "z_depth_median_relative_error": float(np.median(z)),
              "ray_range_median_relative_error": float(np.median(r)),
              "z_depth_pass": bool(np.median(z) < 0.01 or np.median([row["abs_error_z_m"] for row in rows]) < 0.1),
              "ray_range_pass": bool(np.median(r) < 0.01 or np.median([row["abs_error_range_m"] for row in rows]) < 0.1),
              "samples": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
