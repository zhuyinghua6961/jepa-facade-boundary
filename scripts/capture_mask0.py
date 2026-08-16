#!/usr/bin/env python3
"""Capture a small synchronized MASK-0 sensor audit, not a sweep."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from boundary_sweep.carla_utils import discover_carla_root, import_carla
from boundary_sweep.sensors import SynchronousRGBDSeg
from boundary_sweep.surfaces import load_surface, physical_corners, surface_axes, look_direction_to_rotation


POSES = ("CENTER", "LEFT_NEAR_BOUNDARY", "RIGHT_NEAR_BOUNDARY", "TOP_NEAR_BOUNDARY", "BOTTOM_SAFE_VIEW")


def _dimensions(surface, width, height, fov):
    corners = physical_corners(surface)
    facade_width = float(np.linalg.norm(corners[1] - corners[0]))
    facade_height = float(np.linalg.norm(corners[0] - corners[2]))
    vfov = 2.0 * math.atan(math.tan(math.radians(fov) / 2.0) * height / width)
    distance = 0.42 * min(facade_width / (2.0 * math.tan(math.radians(fov) / 2.0)),
                           facade_height / (2.0 * math.tan(vfov / 2.0)))
    return facade_width, facade_height, distance, 2.0 * distance * math.tan(math.radians(fov) / 2.0), 2.0 * distance * math.tan(vfov / 2.0)


def _agl(world, carla, position):
    if not hasattr(world, "cast_ray"):
        return None, "cast_ray_unavailable"
    start = carla.Location(x=float(position[0]), y=float(position[1]), z=float(position[2] + 0.2))
    end = carla.Location(x=float(position[0]), y=float(position[1]), z=float(position[2] - 1000.0))
    try:
        hits = world.cast_ray(start, end)
    except Exception as exc:
        return None, f"cast_ray_error:{type(exc).__name__}"
    points = [float(hit.location.z) for hit in hits if float(hit.location.z) < float(position[2])]
    if not points:
        return None, "no_ground_hit"
    return float(position[2] - max(points)), "raycast_ground"


def pose_transform(surface, pose, world, carla, width, height, fov, min_agl):
    corners = physical_corners(surface)
    center = corners.mean(axis=0)
    h, v, n = surface_axes(surface)
    facade_width, facade_height, distance, footprint_width, footprint_height = _dimensions(surface, width, height, fov)
    offset = 0.0
    if pose == "LEFT_NEAR_BOUNDARY":
        offset = -(facade_width / 2.0 - footprint_width * 0.35)
        axis = "horizontal"
    elif pose == "RIGHT_NEAR_BOUNDARY":
        offset = facade_width / 2.0 - footprint_width * 0.35
        axis = "horizontal"
    elif pose == "TOP_NEAR_BOUNDARY":
        offset = facade_height / 2.0 - footprint_height * 0.35
        axis = "vertical"
    elif pose == "BOTTOM_SAFE_VIEW":
        offset = -(facade_height / 2.0 - footprint_height * 0.45)
        axis = "vertical"
    else:
        axis = "none"
    position = center + n * distance
    if axis == "horizontal":
        position = position + h * offset
    elif axis == "vertical":
        position = position + v * offset
    agl, agl_method = _agl(world, carla, position)
    safety_adjustment = 0.0
    if pose == "BOTTOM_SAFE_VIEW" and agl is not None and agl < min_agl:
        safety_adjustment = float(min_agl - agl)
        position = position + np.array([0.0, 0.0, safety_adjustment])
        agl, agl_method = _agl(world, carla, position)
    rotation = look_direction_to_rotation(carla, -n)
    transform = carla.Transform(carla.Location(x=float(position[0]), y=float(position[1]), z=float(position[2])), rotation)
    return transform, {"pose": pose, "axis": axis, "offset_m": offset, "distance_m": distance,
                       "facade_width_m": facade_width, "facade_height_m": facade_height,
                       "image_footprint_width_m": footprint_width, "image_footprint_height_m": footprint_height,
                       "agl_m": agl, "agl_method": agl_method, "min_agl_m": min_agl,
                       "safety_vertical_adjustment_m": safety_adjustment,
                       "bottom_action_feasible": bool(pose != "BOTTOM_SAFE_VIEW" or (agl is not None and agl >= min_agl))}


def preflight(world):
    library = world.get_blueprint_library()
    names = ["sensor.camera.rgb", "sensor.camera.depth", "sensor.camera.semantic_segmentation", "sensor.camera.instance_segmentation"]
    result = {}
    for name in names:
        try:
            result[name] = {"present": True, "id": library.find(name).id}
        except Exception as exc:
            result[name] = {"present": False, "error": type(exc).__name__}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--carla-root", default=None)
    parser.add_argument("--surfaces", default="results/geo06/surfaces")
    parser.add_argument("--output-root", default="results/mask0/raw")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fov", type=float, default=90.0)
    parser.add_argument("--fixed-delta-seconds", type=float, default=0.05)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--min-agl-m", type=float, default=2.0)
    args = parser.parse_args(argv)
    root = discover_carla_root(args.carla_root)
    carla = import_carla(str(root) if root else args.carla_root)
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    if "Town10" not in world.get_map().name:
        world = client.load_world("Town10")
    surfaces = [load_surface(path) for path in sorted(Path(args.surfaces).glob("*.json"))]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "mask0.capture.v1", "map": world.get_map().name,
                "sensor": {"width": args.width, "height": args.height, "horizontal_fov_deg": args.fov,
                            "fixed_delta_seconds": args.fixed_delta_seconds, "repeats_per_pose": args.repeats,
                            "min_agl_m": args.min_agl_m}, "blueprints": preflight(world), "frames": []}
    if not all(row.get("present") for row in manifest["blueprints"].values()):
        raise SystemExit("MASK-0 requires all four sensor blueprints")
    for surface in surfaces:
        for pose in POSES:
            transform, plan = pose_transform(surface, pose, world, carla, args.width, args.height, args.fov, args.min_agl_m)
            pose_dir = output_root / surface["surface_id"] / pose
            with SynchronousRGBDSeg(world, carla, transform, args.width, args.height, args.fov, args.fixed_delta_seconds) as rig:
                for repeat in range(args.repeats):
                    action = {"mode": "MASK-0_KEY_POSE", "surface_id": surface["surface_id"], "pose": pose,
                              "repeat_index": repeat, "plan": plan}
                    sample = rig.capture(action)
                    stem = f"{pose.lower()}_{repeat:02d}"
                    metadata = rig.save(sample, pose_dir, stem)
                    record = {"surface_id": surface["surface_id"], "pose": pose, "repeat_index": repeat,
                              "frame_id": sample["frame_id"], "timestamp": sample["timestamp"], "plan": plan,
                              "metadata_path": str((pose_dir / f"{stem}.json").relative_to(output_root)),
                              "files": metadata["files"], "camera_transform": metadata["camera_transform"],
                              "K": metadata["K"], "T_world_camera": metadata["T_world_camera"],
                              "sensor_frames": metadata["sensor_frames"], "sensor_timestamps": metadata["sensor_timestamps"],
                              "commanded_action": action, "executed_delta_pose": metadata["executed_delta_pose"]}
                    manifest["frames"].append(record)
    (output_root.parent / "capture_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"map": manifest["map"], "frame_count": len(manifest["frames"]), "output": str(output_root)}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
