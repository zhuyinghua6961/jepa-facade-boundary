#!/usr/bin/env python3
"""Enumerate Building bounding-box vertical surface candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from boundary_sweep.carla_utils import discover_carla_root, import_carla
from boundary_sweep.geometry import transform_matrix


def _vec(v):
    return np.array([float(v.x), float(v.y), float(v.z)], dtype=float)


def candidate_corners(center, h_axis, v_axis, width, height):
    h, v = np.asarray(h_axis), np.asarray(v_axis)
    return np.asarray([center - h * width / 2 + v * height / 2,
                       center + h * width / 2 + v * height / 2,
                       center - h * width / 2 - v * height / 2,
                       center + h * width / 2 - v * height / 2])


def inspect(world, carla):
    label = getattr(getattr(carla, "CityObjectLabel", None), "Buildings", None)
    if label is None or not hasattr(world, "get_level_bbs"):
        return []
    boxes = world.get_level_bbs(label)
    candidates = []
    for bbox_id, bbox in enumerate(boxes):
        loc = _vec(bbox.location)
        ext = _vec(bbox.extent)
        fake = type("T", (), {"location": bbox.location, "rotation": bbox.rotation})
        R = transform_matrix(fake)[:3, :3]
        axes = (("+X", R[:, 0], R[:, 1], ext[0], ext[1]),
                ("-X", -R[:, 0], R[:, 1], ext[0], ext[1]),
                ("+Y", R[:, 1], R[:, 0], ext[1], ext[0]),
                ("-Y", -R[:, 1], R[:, 0], ext[1], ext[0]))
        for side, normal, h_axis, normal_extent, horizontal_extent in axes:
            center = loc + normal * normal_extent
            width, height = 2.0 * horizontal_extent, 2.0 * ext[2]
            corners = candidate_corners(center, h_axis, R[:, 2], width, height)
            candidates.append({
                "bbox_id": bbox_id,
                "bbox_location": {"x": float(bbox.location.x), "y": float(bbox.location.y), "z": float(bbox.location.z)},
                "bbox_extent": {"x": float(bbox.extent.x), "y": float(bbox.extent.y), "z": float(bbox.extent.z)},
                "bbox_rotation": {"pitch": float(bbox.rotation.pitch), "yaw": float(bbox.rotation.yaw), "roll": float(bbox.rotation.roll)},
                "surface": side,
                "center": center.tolist(),
                "normal": (normal / np.linalg.norm(normal)).tolist(),
                "horizontal_axis": (h_axis / np.linalg.norm(h_axis)).tolist(),
                "vertical_axis": (R[:, 2] / np.linalg.norm(R[:, 2])).tolist(),
                "width_m": width,
                "height_m": height,
                "area_m2": width * height,
                "corners_world_TL_TR_BL_BR": corners.tolist(),
            })
    return sorted(candidates, key=lambda x: x["area_m2"], reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carla-root", default=None)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--map", default="Town10")
    ap.add_argument("--output", default="data/facade_candidates.json")
    args = ap.parse_args()
    root = discover_carla_root(args.carla_root)
    carla = import_carla(str(root) if root else args.carla_root)
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    if args.map not in world.get_map().name:
        world = client.load_world(args.map)
    candidates = inspect(world, carla)
    output = {"map": world.get_map().name, "count": len(candidates), "candidates": candidates,
              "note": "Building bounding boxes are candidate filters, not pixel-accurate visual ground truth."}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"map": output["map"], "count": output["count"], "output": str(path)}, indent=2))


if __name__ == "__main__":
    main()
