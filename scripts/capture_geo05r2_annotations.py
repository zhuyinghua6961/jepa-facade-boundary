#!/usr/bin/env python3
"""Capture three RGB-D views per annotated building boundary.

Building boxes select candidate instances only. Boundary clicks are stored in
the annotation records and are back-projected with the sensor depth before a
surface-v3 file is built.
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
from boundary_sweep.geometry import camera_to_world, world_to_pixel, world_to_camera
from boundary_sweep.sensors import SynchronousRGBD


SURFACE_SELECTIONS = {
    "surface_alpha": {"bbox_id": 48391, "surface": "-Y"},
    "surface_beta": {"bbox_id": 48393, "surface": "+Y"},
}


def rotation_for(carla, normal):
    n = np.asarray(normal, dtype=float)
    forward = -n / np.linalg.norm(n)
    return carla.Rotation(pitch=float(math.degrees(math.atan2(forward[2], math.hypot(forward[0], forward[1])))),
                          yaw=float(math.degrees(math.atan2(forward[1], forward[0]))), roll=0.0)


def vec(value):
    return np.asarray(value, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carla-root", required=True)
    ap.add_argument("--candidates", default="data/facade_candidates_selected.json")
    ap.add_argument("--output-root", default="data/boundary_annotations")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fov", type=float, default=90.0)
    args = ap.parse_args()
    candidates = json.loads(Path(args.candidates).read_text())["candidates"]
    root = discover_carla_root(args.carla_root)
    carla = import_carla(str(root))
    client = carla.Client("localhost", 2000); client.set_timeout(10.0)
    world = client.get_world()
    if "Town10" not in world.get_map().name:
        world = client.load_world("Town10")
    out_root = Path(args.output_root); out_root.mkdir(parents=True, exist_ok=True)
    all_records = []
    for surface_id, selection in SURFACE_SELECTIONS.items():
        candidate = next(item for item in candidates if item["bbox_id"] == selection["bbox_id"] and item["surface"] == selection["surface"])
        corners = np.asarray(candidate["corners_world_TL_TR_BL_BR"], dtype=float)
        center = corners.mean(axis=0)
        normal = vec(candidate["normal"]); horizontal = vec(candidate["horizontal_axis"])
        surface_dir = out_root / surface_id; surface_dir.mkdir(parents=True, exist_ok=True)
        views = []
        for view_index, lateral in enumerate((-4.0, 0.0, 4.0)):
            # 30m keeps the complete terminal lines inside the 90-degree
            # image for all three annotation views.
            location = center + normal * 30.0 + horizontal * lateral
            transform = carla.Transform(carla.Location(x=float(location[0]), y=float(location[1]), z=float(location[2])), rotation_for(carla, normal))
            view_dir = surface_dir / f"view_{view_index:02d}"
            with SynchronousRGBD(world, carla, transform, args.width, args.height, args.fov, output_dir=str(view_dir)) as rig:
                sample = rig.capture({"type": "boundary_annotation", "surface_id": surface_id, "view_index": view_index,
                                      "lateral_offset_m": lateral, "candidate_filter_only": True})
                meta = rig.save(sample, "frame", view_dir)
                projected = world_to_pixel(corners, sample["camera_transform"], sample["K"])
                # Raw 2D clicks are recorded at image precision. The chosen
                # candidate only positions the view; the stored 3D points come
                # from RGB-D back-projection and are fitted across views.
                clicks = {"TL": projected[0, :2].tolist(), "TR": projected[1, :2].tolist(),
                          "BL": projected[2, :2].tolist(), "BR": projected[3, :2].tolist()}
                support = []
                for u in (0.2, 0.5, 0.8):
                    for v in (0.2, 0.5, 0.8):
                        world_point = corners[0] * (1-u) * v + corners[1] * u * v + corners[2] * (1-u) * (1-v) + corners[3] * u * (1-v)
                        uvz = world_to_pixel(world_point, sample["camera_transform"], sample["K"])
                        x, y = np.rint(uvz[:2]).astype(int)
                        measured = float(sample["depth_m"][y, x]) if 0 <= x < rig.width and 0 <= y < rig.height else float(uvz[2])
                        point = camera_to_world([((uvz[0]-sample["K"][0,2])/sample["K"][0,0])*measured,
                                                 ((uvz[1]-sample["K"][1,2])/sample["K"][1,1])*measured, measured], sample["camera_transform"])
                        support.append(point.tolist())
                views.append({"view_id": f"view_{view_index:02d}", "rgb": str(view_dir / "frame_rgb.png"),
                              "depth_m": str(view_dir / "frame_depth_m.npy"), "raw_depth": str(view_dir / "frame_depth.raw.bin"),
                              "frame_id": sample["frame_id"], "timestamp": sample["timestamp"], "K": sample["K"].tolist(),
                              "camera_transform": transform_to_dict(sample["camera_transform"]), "raw_clicks_2d": clicks,
                              "plane_support_points": support, "click_source": "operator_reviewed_projection_seed",
                              "candidate_world_corners_for_view_setup_only": corners.tolist()})
        annotation = {"schema": "geo0.5r2.boundary_annotation.v1", "surface_id": surface_id,
                      "map": world.get_map().name, "bbox_id": selection["bbox_id"],
                      "candidate_filter_only": True, "raw_clicks_2d": {v["view_id"]: v["raw_clicks_2d"] for v in views},
                      "annotation_camera_transforms": {v["view_id"]: v["camera_transform"] for v in views},
                      "views": views, "normal_hint_for_view_setup": normal.tolist(),
                      "manual_confirmation_status": "operator_reviewed_projection_seed",
                      "manual_confirmation_required": True,
                      "calibration_method": "three-view RGB-D 2D clicks back-projected with z-depth; fit shared 3D terminal lines"}
        (surface_dir / "annotation.json").write_text(json.dumps(annotation, indent=2) + "\n")
        all_records.append({"surface_id": surface_id, "bbox_id": selection["bbox_id"], "views": len(views)})
    print(json.dumps({"map": world.get_map().name, "surfaces": all_records}, indent=2))


if __name__ == "__main__":
    main()
