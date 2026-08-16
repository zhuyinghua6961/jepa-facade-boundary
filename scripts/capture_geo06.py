#!/usr/bin/env python3
"""Capture the minimal GEO-0.6 outward sweeps with paired RGB-z-depth."""

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
from boundary_sweep.geo06 import DIRECTIONS, plan_sweep, required_distance
from boundary_sweep.labels import generate_frame_label
from boundary_sweep.sensors import SynchronousRGBD
from boundary_sweep.surfaces import load_surface, physical_corners, surface_axes


def _rotation(carla, normal):
    forward = -np.asarray(normal, dtype=float); forward /= np.linalg.norm(forward)
    horizontal = max(math.hypot(float(forward[0]), float(forward[1])), 1e-12)
    return carla.Rotation(pitch=float(math.degrees(math.atan2(float(forward[2]), horizontal))),
                          yaw=float(math.degrees(math.atan2(float(forward[1]), float(forward[0])))), roll=0.0)


def capture_trajectory(surface, direction, world, carla, args):
    corners = physical_corners(surface); center = corners.mean(axis=0); h, v, n = surface_axes(surface)
    distance = required_distance(surface, args.width, args.height, args.fov, args.step_m, 5)
    plan = plan_sweep(surface, direction, distance, args.width, args.height, args.fov, args.step_m, args.max_steps)
    sequence_id = f"{surface['surface_id']}/NORMAL_LOCK/{direction}"
    out_dir = Path(args.output_root) / surface["surface_id"] / "NORMAL_LOCK" / direction
    out_dir.mkdir(parents=True, exist_ok=True)
    rotation = _rotation(carla, n)
    base = center + n * distance
    records = []
    initial = carla.Transform(carla.Location(x=float(base[0]), y=float(base[1]), z=float(base[2])), rotation)
    with SynchronousRGBD(world, carla, initial, args.width, args.height, args.fov,
                         fixed_delta_seconds=args.fixed_delta_seconds, output_dir=str(out_dir)) as rig:
        for step in range(plan["steps"]):
            offset = DIRECTIONS[direction][1] * step * plan["step_m"]
            position = base + (h * offset if plan["axis"] == "horizontal" else v * offset)
            transform = carla.Transform(carla.Location(x=float(position[0]), y=float(position[1]), z=float(position[2])), rotation)
            rig.set_transform(transform)
            action = {"mode": "NORMAL_LOCK", "sequence_id": sequence_id, "facade_id": surface["surface_id"],
                      "direction": direction, "active_boundary": plan["active_boundary"], "step_index": step,
                      "step_m": plan["step_m"], "offset_m": float(offset), "distance_m": distance}
            sample = rig.capture(action)
            stem = f"{direction.lower()}_{step:04d}"
            meta = rig.save(sample, stem, out_dir)
            labels = generate_frame_label(surface, plan["active_boundary"], sample["camera_transform"], sample["K"],
                                          sample["depth_m"], args.width, args.height, pixel_step=args.pixel_step)
            record = {"frame_id": sample["frame_id"], "timestamp": sample["timestamp"], "sequence_id": sequence_id,
                      "facade_id": surface["surface_id"], "camera_transform": meta["camera_transform"],
                      "T_world_camera": meta["T_world_camera"], "K": meta["K"], "commanded_action": action,
                      "executed_delta_pose": sample["executed_delta_pose"].tolist(), "labels": labels,
                      "rgb_path": str((out_dir / f"{stem}_rgb.png").relative_to(Path(args.output_root))),
                      "depth_path": str((out_dir / f"{stem}_depth_m.npy").relative_to(Path(args.output_root))),
                      "depth_metric": "z-depth"}
            (out_dir / f"{stem}_labels.json").write_text(json.dumps(record, indent=2) + "\n")
            records.append(record)
    summary = {"schema": "geo06.trajectory.v1", "sequence_id": sequence_id, "surface_id": surface["surface_id"],
               "direction": direction, "active_boundary": plan["active_boundary"], "plan": plan, "frames": len(records),
               "mode": "NORMAL_LOCK", "orientation_locked": True, "frames_data": records}
    (out_dir / "trajectory.json").write_text(json.dumps(summary, indent=2) + "\n")
    return {key: value for key, value in summary.items() if key != "frames_data"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--carla-root", default=None); parser.add_argument("--surfaces", default="results/geo06/surfaces")
    parser.add_argument("--output-root", default="results/geo06/raw"); parser.add_argument("--distances", nargs="*", type=float)
    parser.add_argument("--directions", nargs="+", default=["LEFT", "RIGHT", "UP", "DOWN"])
    parser.add_argument("--width", type=int, default=640); parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fov", type=float, default=90.0); parser.add_argument("--step-m", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=80); parser.add_argument("--pixel-step", type=int, default=4)
    parser.add_argument("--fixed-delta-seconds", type=float, default=0.05)
    args = parser.parse_args()
    if any(direction not in DIRECTIONS for direction in args.directions):
        raise SystemExit("directions must be LEFT RIGHT UP DOWN")
    root = discover_carla_root(args.carla_root); carla = import_carla(str(root) if root else args.carla_root)
    client = carla.Client("localhost", 2000); client.set_timeout(10.0); world = client.get_world()
    if "Town10" not in world.get_map().name: world = client.load_world("Town10")
    surfaces = [load_surface(path) for path in sorted(Path(args.surfaces).glob("*.json"))]
    summaries = []
    for surface in surfaces:
        for direction in args.directions:
            summaries.append(capture_trajectory(surface, direction, world, carla, args))
    output = Path(args.output_root).parent / "capture_summary.json"; output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"schema": "geo06.capture.v1", "map": world.get_map().name, "sensor": {"width": args.width, "height": args.height, "fov": args.fov, "fixed_delta_seconds": args.fixed_delta_seconds, "depth_metric": "z-depth"}, "trajectories": summaries}, indent=2) + "\n")
    print(json.dumps({"map": world.get_map().name, "trajectory_count": len(summaries), "output": str(output)}, indent=2))


if __name__ == "__main__": main()
