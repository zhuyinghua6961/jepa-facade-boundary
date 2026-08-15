#!/usr/bin/env python3
"""Capture independent GEO-0.5R2 NORMAL_LOCK trajectories."""

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
from boundary_sweep.labels import generate_frame_label
from boundary_sweep.sensors import SynchronousRGBD
from boundary_sweep.surfaces import load_surface, physical_corners, surface_axes


def normal_rotation(carla, normal):
    d = -np.asarray(normal, dtype=float); d /= np.linalg.norm(d)
    return carla.Rotation(pitch=float(math.degrees(math.atan2(d[2], max(math.hypot(d[0], d[1]), 1e-12)))),
                          yaw=float(math.degrees(math.atan2(d[1], d[0]))), roll=0.0)


def capture_surface(surface, world, carla, distances, frames_per_direction, step_m, output_root, overlay_root):
    corners = physical_corners(surface); center = corners.mean(axis=0); h, _, n = surface_axes(surface)
    width = float(np.linalg.norm(corners[1] - corners[0]))
    summaries = []
    for distance in distances:
        for direction, sign in (("LEFT", -1.0), ("RIGHT", 1.0)):
            sequence_id = f"{surface['surface_id']}/NORMAL_LOCK/{distance:g}m/{direction}"
            out_dir = Path(output_root) / surface["surface_id"] / "NORMAL_LOCK" / f"{distance:g}m" / direction
            overlay_dir = Path(overlay_root) / surface["surface_id"] / "NORMAL_LOCK" / f"{distance:g}m" / direction
            out_dir.mkdir(parents=True, exist_ok=True); overlay_dir.mkdir(parents=True, exist_ok=True)
            base = center + n * float(distance); rotation = normal_rotation(carla, n)
            initial = carla.Transform(carla.Location(x=float(base[0]), y=float(base[1]), z=float(base[2])), rotation)
            records = []
            with SynchronousRGBD(world, carla, initial, 640, 480, 90.0, output_dir=str(out_dir)) as rig:
                for step in range(frames_per_direction):
                    offset = sign * float(step) * float(step_m)
                    location = base + h * offset
                    transform = carla.Transform(carla.Location(x=float(location[0]), y=float(location[1]), z=float(location[2])), rotation)
                    rig.set_transform(transform)
                    action = {"mode": "NORMAL_LOCK", "sequence_id": sequence_id, "facade_id": surface["surface_id"],
                              "direction": direction, "distance_m": float(distance), "step_index": step,
                              "step_m": float(step_m), "lateral_offset_m": offset}
                    sample = rig.capture(action); stem = f"{direction.lower()}_{step:04d}"
                    meta = rig.save(sample, stem, out_dir)
                    labels = generate_frame_label(surface, direction, sample["camera_transform"], sample["K"], sample["depth_m"],
                                                   640, 480, pixel_step=4,
                                                   overlay_path=overlay_dir / f"{stem}_overlay.png",
                                                   rgb_path=out_dir / f"{stem}_rgb.png")
                    record = {"frame_id": sample["frame_id"], "timestamp": sample["timestamp"], "sequence_id": sequence_id,
                              "facade_id": surface["surface_id"], "camera_transform": meta["camera_transform"],
                              "commanded_action": action, "executed_delta_pose": sample["executed_delta_pose"].tolist(),
                              "labels": labels}
                    (out_dir / f"{stem}_labels.json").write_text(json.dumps(record, indent=2) + "\n")
                    records.append(record)
            states = [r["labels"]["label"] for r in records]
            compressed = []
            for state in states:
                if not compressed or compressed[-1] != state: compressed.append(state)
            summary = {"surface_id": surface["surface_id"], "facade_id": surface["surface_id"], "mode": "NORMAL_LOCK",
                       "distance_m": float(distance), "direction": direction, "frames": len(records),
                       "state_counts": {s: states.count(s) for s in ("IN", "STRADDLE", "OUT", "UNKNOWN")},
                       "compressed_state_sequence": compressed, "orientation_locked": True,
                       "step_m": float(step_m), "physical_boundary_source": surface.get("source", {})}
            (out_dir / "trajectory.json").write_text(json.dumps({"summary": summary, "frames": records}, indent=2) + "\n")
            summaries.append(summary)
    return summaries


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--carla-root", required=True); ap.add_argument("--surfaces", default="data/surfaces_v3"); ap.add_argument("--distances", nargs="+", type=float, default=[5,10,15]); ap.add_argument("--frames-per-direction", type=int, default=80); ap.add_argument("--step-m", type=float, default=0.5); ap.add_argument("--output-root", default="data/sweeps_geo05r2"); ap.add_argument("--overlay-root", default="data/overlays_geo05r2")
    args = ap.parse_args(); root = discover_carla_root(args.carla_root); carla = import_carla(str(root)); client = carla.Client("localhost", 2000); client.set_timeout(10.0); world = client.get_world()
    if "Town10" not in world.get_map().name: world = client.load_world("Town10")
    surfaces = [load_surface(p) for p in sorted(Path(args.surfaces).glob("*.json"))]
    if len(surfaces) != 2: raise SystemExit("R2 expects exactly two surfaces_v3 files")
    summaries = []
    for surface in surfaces:
        summaries.extend(capture_surface(surface, world, carla, args.distances, args.frames_per_direction, args.step_m, args.output_root, args.overlay_root))
    print(json.dumps({"map": world.get_map().name, "mode": "NORMAL_LOCK", "summaries": summaries}, indent=2))


if __name__ == "__main__": main()
