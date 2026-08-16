#!/usr/bin/env python3
"""Small adaptive MASK-1 pilot using only sigma LEFT/RIGHT contours."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from boundary_sweep.carla_utils import discover_carla_root, import_carla
from boundary_sweep.geometry import (camera_to_world, pixel_depth_to_camera_point,
                                     pixel_to_camera_ray, ray_plane_intersection,
                                     world_to_camera, world_point_to_surface_coordinate)
from boundary_sweep.labels import adaptive_instance_boundary_evidence
from boundary_sweep.segmentation import bgra_array, decode_instance_channels
from boundary_sweep.sensors import SynchronousRGBDSeg
from boundary_sweep.surfaces import load_surface, physical_corners, surface_axes


def normal_rotation(carla, normal):
    d = -np.asarray(normal, dtype=float); d /= np.linalg.norm(d)
    return carla.Rotation(pitch=float(math.degrees(math.atan2(d[2], max(math.hypot(d[0], d[1]), 1e-12)))),
                          yaw=float(math.degrees(math.atan2(d[1], d[0]))), roll=0.0)


def draw_overlay(rgb_path: Path, mask: np.ndarray, contour: np.ndarray, label: str,
                 action: dict, output: Path) -> None:
    image = Image.open(rgb_path).convert("RGBA")
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask] = (40, 210, 80, 75)
    rgba[contour] = (255, 190, 0, 230)
    image = Image.alpha_composite(image, Image.fromarray(rgba, mode="RGBA"))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 640, 34), fill=(0, 0, 0, 220))
    draw.text((6, 7), f"{action['direction']} step={action['step_index']} {label} offset={action['offset_m']:.2f}m", fill=(255, 235, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output, quality=84, optimize=True)


def contact_sheet(frame_rows: list[dict], output: Path, title: str) -> None:
    if not frame_rows:
        return
    count = min(12, len(frame_rows))
    indices = [round(i * (len(frame_rows) - 1) / max(count - 1, 1)) for i in range(count)]
    tiles = []
    for index in indices:
        row = frame_rows[index]
        image = Image.open(row["rgb_path"]).convert("RGB").resize((320, 240))
        draw = ImageDraw.Draw(image); draw.rectangle((0, 0, 320, 38), fill=(0, 0, 0))
        draw.text((5, 5), f"frame={row['frame_id']} {row['state']}", fill=(255, 235, 0))
        draw.text((5, 21), f"step={row['step_index']} cov={row['target_coverage']:.3f}", fill=(255, 235, 0))
        tiles.append(image)
    sheet = Image.new("RGB", (1280, 720), (20, 20, 20))
    for i, tile in enumerate(tiles): sheet.paste(tile, ((i % 4) * 320, (i // 4) * 240))
    sheet.save(output, quality=84, optimize=True)


def triptych(frame_rows: list[dict], output: Path) -> None:
    if not frame_rows:
        return
    selected = []
    for state in ("IN", "APPROACH", "STRADDLE"):
        selected.append(next((row for row in frame_rows if row["state"] == state), frame_rows[-1]))
    canvas = Image.new("RGB", (960, 480), (20, 20, 20))
    for i, row in enumerate(selected):
        image = Image.open(row["rgb_path"]).convert("RGB").resize((320, 480))
        draw = ImageDraw.Draw(image); draw.rectangle((0, 0, 320, 34), fill=(0, 0, 0))
        draw.text((5, 5), f"{row['state']} frame={row['frame_id']}", fill=(255, 235, 0)); canvas.paste(image, (i * 320, 0))
    canvas.save(output, quality=84, optimize=True)


def boundary_world_estimate(row: dict, surface: dict, K: np.ndarray, transform, depth_m: np.ndarray) -> dict:
    contour = np.asarray(row["contour_pixels"], dtype=float)
    points = []
    residuals = []
    camera_origin = camera_to_world([0.0, 0.0, 0.0], transform)
    plane_origin = np.asarray(surface["plane_origin"], dtype=float)
    plane_normal = np.asarray(surface["plane_normal"], dtype=float)
    h_axis = np.asarray(surface["horizontal_axis"], dtype=float)
    v_axis = np.asarray(surface["vertical_axis"], dtype=float)
    width_m = float(surface.get("width_m", 1e9)); height_m = float(surface.get("height_m", 1e9))
    for pixel in contour[:: max(1, len(contour) // 250)]:
        x, y = int(round(pixel[0])), int(round(pixel[1]))
        if 0 <= x < depth_m.shape[1] and 0 <= y < depth_m.shape[0] and np.isfinite(depth_m[y, x]) and depth_m[y, x] > 0.05:
            ray_camera = pixel_to_camera_ray(pixel, K)
            ray_world = camera_to_world(ray_camera, transform) - camera_origin
            plane_point = ray_plane_intersection(camera_origin, ray_world, plane_origin, plane_normal)
            measured_z = float(depth_m[y, x])
            if plane_point is None:
                continue
            predicted_z = float(world_to_camera(plane_point, transform)[2])
            coordinate = world_point_to_surface_coordinate(plane_point, plane_origin, h_axis, v_axis)
            residual = abs(measured_z - predicted_z)
            if predicted_z <= 0 or residual > max(1.0, 0.05 * predicted_z):
                continue
            if coordinate[0] < -width_m / 2.0 - 1.0 or coordinate[0] > width_m / 2.0 + 1.0 or coordinate[1] < -height_m / 2.0 - 1.0 or coordinate[1] > height_m / 2.0 + 1.0:
                continue
            camera = pixel_depth_to_camera_point(pixel, measured_z, K, mode="z-depth")
            points.append(np.asarray(camera_to_world(camera, transform), dtype=float)); residuals.append(residual)
    if not points:
        return {"frame_id": row["frame_id"], "point_count": 0, "rejected_point_count": int(len(contour)), "boundary_world_points": [], "median_world_coordinate": None, "mad_m": None, "depth_residual_median": None}
    arr = np.asarray(points); median = np.median(arr, axis=0); mad = np.median(np.abs(arr - median), axis=0)
    return {"frame_id": row["frame_id"], "point_count": len(arr), "rejected_point_count": int(len(contour) - len(arr)), "boundary_world_points": arr.tolist(), "median_world_coordinate": median.tolist(), "mad_m": mad.tolist(), "depth_residual_median": float(np.median(residuals))}


def capture_direction(surface, direction, world, carla, distance, step_m, max_steps, confirm_frames, output_root, assets_root, target_id):
    corners = physical_corners(surface); center = corners.mean(axis=0); h, _v, n = surface_axes(surface)
    sign = -1.0 if direction == "LEFT" else 1.0
    base = center + n * float(distance)
    rotation = normal_rotation(carla, n)
    sequence_id = f"{surface['surface_id']}/MASK-1/{distance:g}m/{direction}"
    out_dir = Path(output_root) / surface["surface_id"] / "NORMAL_LOCK" / f"{distance:g}m" / direction
    overlay_dir = Path(assets_root) / "overlays" / surface["surface_id"] / f"{distance:g}m" / direction
    out_dir.mkdir(parents=True, exist_ok=True); overlay_dir.mkdir(parents=True, exist_ok=True)
    rows, world_estimates = [], []
    consecutive_straddle = 0
    with SynchronousRGBDSeg(world, carla, carla.Transform(carla.Location(x=float(base[0]), y=float(base[1]), z=float(base[2])), rotation), 640, 480, 90.0, 0.05) as rig:
        for step in range(max_steps):
            offset = sign * float(step) * float(step_m)
            position = base + h * offset
            transform = carla.Transform(carla.Location(x=float(position[0]), y=float(position[1]), z=float(position[2])), rotation)
            rig.set_transform(transform)
            action = {"mode": "MASK-1_ADAPTIVE", "sequence_id": sequence_id, "facade_id": surface["surface_id"], "direction": direction, "distance_m": float(distance), "step_index": step, "step_m": float(step_m), "offset_m": offset}
            sample = rig.capture(action); stem = f"{direction.lower()}_{step:04d}"; metadata = rig.save(sample, out_dir, stem)
            instance = decode_instance_channels(bgra_array(sample["data"]["instance"]))
            semantic = decode_instance_channels(bgra_array(sample["data"]["semantic"]))["semantic_tag"]
            mask = (semantic == 3) & (instance["instance_id_16bit"] == int(target_id))
            evidence = adaptive_instance_boundary_evidence(mask, direction, min_side_fraction=0.05, min_span_fraction=0.70)
            if step < 3:
                state = "IN" if evidence["label"] == "IN" else "UNKNOWN"
            elif evidence["label"] == "STRADDLE":
                state = "STRADDLE"; consecutive_straddle += 1
            else:
                state = "APPROACH"; consecutive_straddle = 0
            rgb_path = out_dir / f"{stem}_rgb.png"; overlay_path = overlay_dir / f"{stem}.jpg"
            row = {"frame_id": int(sample["frame_id"]), "timestamp": float(sample["timestamp"]), "sequence_id": sequence_id, "surface_id": surface["surface_id"], "direction": direction, "distance_m": float(distance), "step_index": step, "offset_m": offset, "state": state, "target_coverage": float(mask.mean()), "contour_present": bool(evidence["contour_present"]), "contour_span_fraction": float(evidence["contour_span_fraction"]), "contour_centroid_px": evidence["contour_centroid_px"], "target_side_fraction": float(evidence["target_fraction"]), "external_side_fraction": float(evidence["external_fraction"]), "label_reason": evidence["reason"], "contour_pixels": np.column_stack(np.where(evidence["contour"])[::-1]).tolist(), "rgb_path": str(rgb_path), "files": metadata["files"], "camera_transform": metadata["camera_transform"], "T_world_camera": metadata["T_world_camera"], "K": metadata["K"], "commanded_action": action, "executed_delta_pose": sample["executed_delta_pose"].tolist(), "sensor_frames": metadata["sensor_frames"], "sensor_timestamps": metadata["sensor_timestamps"], "target_instance_id_16bit": int(target_id)}
            draw_overlay(rgb_path, mask, evidence["contour"], state, action, overlay_path); row["overlay_path"] = str(overlay_path); rows.append(row)
            if state == "STRADDLE" and consecutive_straddle >= confirm_frames:
                break
    for row in rows:
        if row["state"] == "STRADDLE": world_estimates.append(boundary_world_estimate(row, surface, np.asarray(row["K"]), np.asarray(row["T_world_camera"]), np.load(resolve_depth(row["files"]["depth_m"]["path"], out_dir))))
    states = [row["state"] for row in rows]; compressed = []
    for state in states:
        if not compressed or compressed[-1] != state: compressed.append(state)
    summary = {"sequence_id": sequence_id, "surface_id": surface["surface_id"], "direction": direction, "distance_m": float(distance), "step_m": float(step_m), "frames": len(rows), "state_counts": {state: states.count(state) for state in ("IN", "APPROACH", "STRADDLE", "UNKNOWN", "OUT")}, "compressed_state_sequence": compressed, "orientation_locked": True, "stop_after_confirmed_straddle": bool(rows and states[-1] == "STRADDLE" and states.count("STRADDLE") >= confirm_frames), "target_instance_id_16bit": int(target_id)}
    (out_dir / "trajectory.json").write_text(json.dumps({"summary": summary, "frames": rows}, indent=2) + "\n")
    contact_sheet(rows, Path(assets_root) / f"contact_{surface['surface_id']}_{direction}.jpg", sequence_id)
    triptych(rows, Path(assets_root) / f"triptych_{surface['surface_id']}_{direction}.jpg")
    return summary, rows, world_estimates


def resolve_depth(path: str, out_dir: Path) -> Path:
    candidate = PROJECT_ROOT / path
    return candidate if candidate.exists() else out_dir / Path(path).name


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--carla-root", required=True); ap.add_argument("--surface", default="results/geo06/surfaces/surface_sigma.json"); ap.add_argument("--distance", type=float, default=10.0); ap.add_argument("--step-m", type=float, default=0.5); ap.add_argument("--max-steps", type=int, default=80); ap.add_argument("--confirm-frames", type=int, default=3); ap.add_argument("--output-root", default="results/mask1/raw"); ap.add_argument("--assets-root", default="docs/assets/mask1")
    args = ap.parse_args(argv); root = discover_carla_root(args.carla_root); carla = import_carla(str(root)); client = carla.Client("localhost", 2000); client.set_timeout(10.0); world = client.get_world()
    surface = load_surface(args.surface); groups = json.loads((PROJECT_ROOT / "results/mask0/target_group_manifest_r1.json").read_text()); target_id = groups[surface["surface_id"]]["target_instance_ids_16bit"][0]
    summaries, all_rows, estimates = [], [], []
    for direction in ("LEFT", "RIGHT"):
        summary, rows, world_rows = capture_direction(surface, direction, world, carla, args.distance, args.step_m, args.max_steps, args.confirm_frames, args.output_root, args.assets_root, target_id); summaries.append(summary); all_rows.extend(rows); estimates.extend([{**item, "sequence_id": summary["sequence_id"], "direction": direction} for item in world_rows])
    result_root = PROJECT_ROOT / "results/mask1"; result_root.mkdir(parents=True, exist_ok=True)
    (result_root / "trajectories.json").write_text(json.dumps(summaries, indent=2) + "\n"); (result_root / "world_boundary_estimates.json").write_text(json.dumps(estimates, indent=2) + "\n")
    fields = ["frame_id", "timestamp", "sequence_id", "surface_id", "direction", "distance_m", "step_index", "offset_m", "state", "target_coverage", "contour_present", "contour_span_fraction", "contour_centroid_px", "target_side_fraction", "external_side_fraction", "label_reason", "rgb_path", "overlay_path", "target_instance_id_16bit"]
    with (result_root / "frame_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows({key: json.dumps(row[key]) if isinstance(row.get(key), (list, dict)) else row.get(key, "") for key in fields} for row in all_rows)
    print(json.dumps({"map": world.get_map().name, "summaries": summaries, "world_boundary_estimate_count": len(estimates)}, indent=2))


if __name__ == "__main__": raise SystemExit(main())
